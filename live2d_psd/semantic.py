"""Character parsing for anime, painted and shaded art (the "smart" pipeline).

Cues, from strongest to weakest:

* the character matte (transparency, or ISNet background removal);
* an anime face detector, plus MediaPipe body keypoints;
* MobileSAM, prompted with points placed from those detections ("click on
  the hair", "click along the left forearm"), and its segment-everything
  masks used to keep objects in one piece;
* colour models of skin and hair sampled around the face.

The result is a part label for every character pixel plus the parts' names,
folders and draw order.  Anything missing (no face, no pose, no SAM) degrades
gracefully to colour and position rules.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.color import rgb2lab

from . import models


@dataclass
class Scene:
    rgb: np.ndarray
    alpha: np.ndarray  # float matte 0..1
    fg: np.ndarray
    had_background: bool
    lab: np.ndarray = None
    face: tuple | None = None  # x, y, w, h
    face_angle: float = 0.0
    pose: dict | None = None
    notes: list = field(default_factory=list)


@dataclass(eq=False)
class SemPart:
    name: str
    path: list[str]
    z: float
    mask: np.ndarray
    sweep: bool = True
    fill: str = "inpaint"  # how hidden areas are painted: inpaint | color
    hidden_hint: np.ndarray | None = None  # extra area this part should cover underneath


# --------------------------------------------------------------------------- loading


def load_scene(path: str, log=print, matte: str = "auto") -> Scene:
    rgba = np.array(Image.open(path).convert("RGBA"))
    rgb = rgba[..., :3].copy()
    a = rgba[..., 3].astype(np.float32) / 255
    had_bg = (a < 0.5).mean() < 0.01
    if had_bg and matte != "none":
        log("removing background")
        m = models.remove_background(rgb)
        if m is None:
            from .segment import load_rgba

            log("  ISNet unavailable, using flood fill from the border")
            a = load_rgba(path)[..., 3].astype(np.float32) / 255
        else:
            a = _clean_matte(m, rgb)
    fg = a > 0.5
    sc = Scene(rgb=rgb, alpha=a, fg=fg, had_background=had_bg)
    sc.lab = rgb2lab(rgb.astype(np.float32) / 255)
    return sc


def _clean_matte(m: np.ndarray, rgb: np.ndarray | None = None) -> np.ndarray:
    hard = m > 0.5
    if rgb is not None:
        # dark outline pixels right next to the character belong to it (the
        # matte sometimes drops a stretch of outline, opening up enclosed patches)
        lum = rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
        hard |= (lum < 70) & ndi.binary_dilation(hard, iterations=6)
    lab, n = ndi.label(hard)
    if n == 0:
        return m
    sizes = np.bincount(lab.ravel())[1:]
    keep = np.zeros(n + 1, bool)
    keep[1:] = sizes >= 0.02 * sizes.max()
    hard = keep[lab]
    # small enclosed holes are part of the character (a patch of fur the
    # same colour as the backdrop); big ones may be real gaps, handled later
    holes = ndi.binary_fill_holes(hard) & ~hard
    hl, hn = ndi.label(holes)
    if hn:
        hs = np.bincount(hl.ravel())[1:]
        small = np.zeros(hn + 1, bool)
        small[1:] = hs < 0.10 * hard.sum()
        hard |= small[hl]
        m = np.where(small[hl], 1.0, m)
    hard = ndi.binary_fill_holes(hard) & (m > 0.15) | hard
    # keep the soft edge only around the kept pieces
    near = ndi.binary_dilation(hard, iterations=3)
    out = np.where(near, m, 0).astype(np.float32)
    out[hard & (out < 0.5)] = 0.5
    return out


# --------------------------------------------------------------------------- detection


def detect(sc: Scene, log=print):
    faces = models.detect_faces(sc.rgb, sc.fg)
    faces = [f for f in faces if sc.fg[int(min(sc.fg.shape[0] - 1, f[1] + f[3] / 2)), int(min(sc.fg.shape[1] - 1, f[0] + f[2] / 2))]]
    sc.pose = models.detect_pose(sc.rgb, sc.fg)
    if sc.pose:
        _orient_pose(sc.pose)
    if faces:
        x, y, w, h, ang = faces[0]
        sc.face, sc.face_angle = (x, y, w, h), ang
    elif sc.pose:
        sc.face = _face_from_pose(sc.pose)
        if sc.face:
            sc.notes.append("face box estimated from body keypoints")
    log(f"  face: {'found' if sc.face else 'not found'}, body keypoints: {'found' if sc.pose else 'not found'}")


def _orient_pose(p):
    # Live2D L/R is the character's own side, which for a character facing the
    # viewer is on the right of the picture.  Drawn characters often confuse the
    # detector, so trust the picture: whichever shoulder is further right is L.
    if p["shoulder_l"][0] < p["shoulder_r"][0]:
        for k in list(p):
            if k.endswith("_l"):
                r = k[:-2] + "_r"
                p[k], p[r] = p[r], p[k]


def _face_from_pose(p):
    pts = [p[k] for k in ("nose", "eye_l", "eye_r", "ear_l", "ear_r", "mouth_l", "mouth_r") if p[k][2] > 0.3]
    if len(pts) < 3:
        return None
    xs = np.array([q[0] for q in pts])
    ys = np.array([q[1] for q in pts])
    w = max(np.ptp(xs) * 1.5, 1.0)
    w = max(w, abs(p["shoulder_l"][0] - p["shoulder_r"][0]) * 0.45)
    cx, cy = xs.mean(), ys.mean()
    return (cx - w / 2, cy - w * 0.55, w, w * 1.05)


# --------------------------------------------------------------------------- helpers


def _ab_dist(lab, ref):
    return np.hypot(lab[..., 1] - ref[1], lab[..., 2] - ref[2])


def _sample(lab, mask):
    px = lab[mask]
    return np.median(px, 0) if len(px) else None


def _box_mask(shape, x0, y0, x1, y1):
    m = np.zeros(shape, bool)
    h, w = shape
    m[max(0, int(y0)) : min(h, int(y1)), max(0, int(x0)) : min(w, int(x1))] = True
    return m


def _keep_touching(mask, seed):
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    ids = np.unique(lab[seed & mask])
    ids = ids[ids > 0]
    return np.isin(lab, ids)


def _seg_dist(px, py, a, b):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L = dx * dx + dy * dy + 1e-6
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / L, 0, 1)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy))


class _Sam:
    """Prompted MobileSAM at two scales: the whole character and the head."""

    def __init__(self, sc: Scene, log):
        self.sam = models.MobileSAM.load()
        self.sc = sc
        self.emb = {}
        if self.sam is None:
            log("  MobileSAM not found (models/ folder); using colour rules only")

    def _emb(self, key, box):
        if key not in self.emb:
            x0, y0, x1, y1 = [int(v) for v in box]
            crop = self.sc.rgb[y0:y1, x0:x1].copy()
            crop[~self.sc.fg[y0:y1, x0:x1]] = 255
            self.emb[key] = (self.sam.embed(crop), (x0, y0))
        return self.emb[key]

    def mask(self, key, box, pos, neg=(), prefer="best"):
        if self.sam is None or not len(pos):
            return None
        info, (ox, oy) = self._emb(key, box)
        pts = np.array([(x - ox, y - oy) for x, y in list(pos) + list(neg)], np.float32)[None]
        lab = np.array([1] * len(pos) + [0] * len(neg), np.float32)[None]
        low, iou = self.sam.predict(info, pts, lab)
        k = int(np.argmax(iou[0])) if prefer == "best" else {"small": 0, "mid": 1, "large": 2}[prefer]
        m = self.sam.upscale(info, low[0, k]) > 0
        full = np.zeros(self.sc.fg.shape, bool)
        h, w = m.shape
        full[oy : oy + h, ox : ox + w] = m
        return full & self.sc.fg


# --------------------------------------------------------------------------- parsing

# Draw order (bigger = closer), following the official Live2D samples.
Z = {
    "Hair Front": 95, "Brow": 90, "Eye Line": 88, "Iris": 87, "Eye White": 86, "Mouth": 84, "Nose": 83,
    "Face": 80, "Ear": 75, "Hair Side": 72, "Prop": 70, "Hand": 62, "Forearm": 61, "Upper Arm": 60,
    "Upper Body": 50, "Lower Body": 40, "Neck": 30, "Thigh": 21, "Leg": 21, "Shin": 20, "Foot": 19, "Hair Back": 2,
}


class _Reg:
    """Part names <-> ids for the proposal map."""

    def __init__(self):
        self.parts: list[dict] = []

    def id(self, name, path, z, **opts):
        for i, p in enumerate(self.parts, start=1):
            if p["name"] == name and p["path"] == path:
                return i
        self.parts.append(dict(name=name, path=path, z=z, **opts))
        return len(self.parts)


def parse(sc: Scene, log=print) -> list[SemPart]:
    from . import partseg

    if partseg.available():
        from .semantic_seg import parse_learned

        parts = parse_learned(sc, log)
        if parts:
            return parts
    return parse_rules(sc, log)


def parse_rules(sc: Scene, log=print) -> list[SemPart]:
    from .atoms import background_atoms, make_atoms, vote

    H, W = sc.fg.shape
    lab = sc.lab
    L = lab[..., 0]
    yy, xx = np.mgrid[0:H, 0:W]

    log("  splitting into regions along the line art")
    atoms = make_atoms(sc.rgb, sc.fg, lab)
    sam = _Sam(sc, log)
    if sc.had_background:
        drop = background_atoms(atoms, lab, sc.alpha, protect=_person_mask(sc, sam))
        if drop.any():
            atoms.labels[drop] = 0
            keep = atoms.labels > 0
            ink_ok = atoms.line & ndi.binary_dilation(keep, iterations=int(2 * atoms.hw + 2))
            sc.fg = keep | ink_ok
            sc.alpha = np.where(sc.fg | (sc.alpha < 0.5), sc.alpha, 0).astype(np.float32)
            sc.alpha[~ndi.binary_dilation(sc.fg, iterations=2)] = 0
            atoms.line &= sc.fg
            log(f"  removed {int(drop.sum())} px of background the matte let through")
    log(f"  {atoms.n} regions")
    fg = sc.fg
    ys, xs = np.nonzero(fg)
    bbox = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    char_box = (max(0, bbox[0] - 4), max(0, bbox[1] - 4), min(W, bbox[2] + 4), min(H, bbox[3] + 4))
    reg = _Reg()
    P = np.zeros((H, W), np.int32)  # body-level proposal (voted per region)
    O = np.zeros((H, W), np.int32)  # pixel-exact overrides (eyes, brows, mouth)
    dark = fg & (L < 35)

    face_full = np.zeros((H, W), bool)
    hair = np.zeros((H, W), bool)
    face_skin = np.zeros((H, W), bool)
    skin_like = np.zeros((H, W), bool)
    creature = False
    if sc.face is None and sc.pose is None:
        creature = _creature(sc, atoms, sam, reg, P, O, log)
    if sc.face is not None and not creature:
        fx, fy, fw, fh = sc.face
        fcx = fx + fw / 2
        head_box = (max(0, fx - 0.9 * fw), max(0, fy - 0.9 * fh), min(W, fx + 1.9 * fw), min(H, fy + 1.6 * fh))
        cheek = _box_mask((H, W), fx + 0.25 * fw, fy + 0.55 * fh, fx + 0.75 * fw, fy + 0.85 * fh) & fg & (L > 45) & ~sc_line(atoms)
        skin_ref = _sample(lab, cheek)
        if skin_ref is None:
            skin_ref = np.array([80.0, 10.0, 15.0])
        skin_chroma = np.hypot(skin_ref[1], skin_ref[2])
        chroma = np.hypot(lab[..., 1], lab[..., 2])
        # pale anime skin is close to white: demand some of its warmth so eye
        # whites and white clothes don't count as skin
        skin_like = fg & (_ab_dist(lab, skin_ref) < 14) & (L > skin_ref[0] - 38) & (L < skin_ref[0] + 18)
        skin_like &= chroma > max(3.0, 0.45 * skin_chroma)

        face_zone = _box_mask((H, W), fx - 0.1 * fw, fy - 0.05 * fh, fx + 1.1 * fw, fy + 1.12 * fh)
        cheeks = [(fx + 0.3 * fw, fy + 0.7 * fh), (fx + 0.7 * fw, fy + 0.7 * fh), (fcx, fy + 0.88 * fh)]
        cheeks = [p for p in cheeks if skin_like[int(min(H - 1, p[1])), int(min(W - 1, p[0]))]] or cheeks
        face_skin = skin_like & face_zone
        seed = _box_mask((H, W), fx + 0.3 * fw, fy + 0.55 * fh, fx + 0.7 * fw, fy + 0.9 * fh)
        face_skin = _keep_touching(ndi.binary_closing(face_skin, iterations=2) & fg, seed)
        face_full = _convex(ndi.binary_fill_holes(face_skin))
        oval = np.zeros((H, W), np.uint8)
        cv2.ellipse(oval, (int(fcx), int(fy + 0.52 * fh)), (int(fw * 0.5), int(fh * 0.56)), sc.face_angle, 0, 360, 1, -1)
        face_full = (face_full | (oval.astype(bool) & (yy < fy + 0.5 * fh))) & fg

        # hair: dominant non-skin colour around the top and sides of the face
        ring = _box_mask((H, W), fx - 0.25 * fw, fy - 0.45 * fh, fx + 1.25 * fw, fy + 0.35 * fh) & fg & ~skin_like & ~face_skin
        ring &= ~(L > 92) & ~sc_line(atoms)
        hair_ref = _dominant(lab, ring)
        if hair_ref is not None:
            chroma = np.hypot(hair_ref[1], hair_ref[2])
            if chroma > 12:
                # same hue family as the hair; a yellow sweater next to orange hair is not hair
                hue = np.degrees(np.arctan2(lab[..., 2], lab[..., 1]))
                href = np.degrees(np.arctan2(hair_ref[2], hair_ref[1]))
                dh = np.abs((hue - href + 180) % 360 - 180)
                hair_like = fg & (_ab_dist(lab, hair_ref) < max(16, chroma * 0.4)) & (dh < 14) & ~skin_like
                hue_ok = dh < 25
            else:  # black / grey / white hair: go by lightness instead
                hair_like = fg & (np.abs(L - hair_ref[0]) < 18) & (np.hypot(lab[..., 1], lab[..., 2]) < 18) & ~skin_like
                hue_ok = np.hypot(lab[..., 1], lab[..., 2]) < 25
            hair_like = ndi.binary_closing(hair_like, iterations=2) & fg & ~face_skin
            hair = _keep_touching(hair_like, ring & hair_like)
            hp = _spread_points(hair & ring, 6)
            if len(hp):
                hs = sam.mask("head", head_box, hp, cheeks[:2])
                if hs is not None:
                    hair |= hs & hue_ok & ~face_skin & ~skin_like & ndi.binary_dilation(hair, iterations=int(0.1 * fw) + 3)
        hair &= ~face_skin

        # facial features: inside the face outline and not skin.  Hair inside
        # the face only counts as hair when it connects to hair outside the
        # face (bangs); an iris of the same colour as the hair stays an eye.
        bangs = _keep_touching(hair, hair & ~face_full)
        feats = face_full & ~face_skin & ~bangs & fg
        hair &= ~(feats & ~ndi.binary_dilation(bangs, iterations=1))
        flab, _ = ndi.label(ndi.binary_closing(feats, iterations=1) & feats)
        comps = []
        for i, sl in enumerate(ndi.find_objects(flab), start=1):
            if sl is None:
                continue
            m = flab == i
            a = int(m.sum())
            if a < max(12, 0.0004 * fw * fh):
                continue
            cy, cx = ndi.center_of_mass(m)
            comps.append(dict(m=m, a=a, u=(cx - fx) / fw, v=(cy - fy) / fh, bh=sl[0].stop - sl[0].start, bw=sl[1].stop - sl[1].start))
        eyes = sorted([c for c in comps if 0.2 < c["v"] < 0.8 and c["a"] > 0.005 * fw * fh], key=lambda c: -c["a"])[:2]
        sam_eyes = _iris_eyes(sam, head_box, sc, face_full, face_skin, (fx, fy, fw, fh), skin_ref)
        if len(sam_eyes) == 2:
            # SAM around the irises is more reliable than colour alone (painted
            # eyes, irises the colour of the hair, eyes touching the bangs)
            for e in sam_eyes:
                for c in comps:
                    if (c["m"] & e["m"]).sum() > 0.3 * c["a"]:
                        e["m"] = e["m"] | c["m"]
            eyes = sam_eyes
        elif len(eyes) < 2:
            eyes += _sam_eyes(sam, head_box, sc, eyes, face_full, (fx, fy, fw, fh))
        for e in eyes:
            hair &= ~e["m"]
        eyes.sort(key=lambda c: c["u"])
        for i, c in enumerate(eyes):
            side = ("R" if i == 0 else "L") if len(eyes) == 2 else ("R" if c["u"] < 0.5 else "L")
            _eye_parts(reg, O, c["m"], side, lab)
        eye_px = np.logical_or.reduce([e["m"] for e in eyes]) if eyes else np.zeros((H, W), bool)
        comps = [c for c in comps if not any(c is e for e in eyes) and (c["m"] & eye_px).sum() < 0.3 * c["a"]]
        eye_top = min((c["v"] for c in eyes), default=0.45) - 0.06
        for c in comps:
            if c["v"] < eye_top and c["bw"] > 1.5 * c["bh"]:
                side = "R" if c["u"] < 0.5 else "L"
                O[c["m"]] = reg.id(f"Brow {side}", ["Head"], Z["Brow"], sweep=False)
            elif c["v"] > 0.6 and abs(c["u"] - 0.5) < 0.3:
                O[c["m"]] = reg.id("Mouth", ["Head"], Z["Mouth"], sweep=False)
            elif abs(c["u"] - 0.5) < 0.2:
                O[c["m"]] = reg.id("Nose", ["Head"], Z["Nose"], sweep=False)
            else:
                O[c["m"] & ~dark] = reg.id("Face", ["Head"], Z["Face"], fill="color")
        face_id = reg.id("Face", ["Head"], Z["Face"], fill="color")
        P[face_skin] = face_id

        # hair: front (over the face), side (beside it), back (everything else)
        front = hair & ndi.binary_dilation(face_full, iterations=2) & (yy < fy + 0.85 * fh)
        side_band = hair & ~front & (yy < fy + 1.6 * fh) & ndi.binary_dilation(front | face_full, iterations=int(0.25 * fw) + 2)
        P[hair] = reg.id("Hair Back", ["Hair Back"], Z["Hair Back"])
        P[side_band & (xx < fcx)] = reg.id("Hair Side R", ["Head"], Z["Hair Side"])
        P[side_band & (xx >= fcx)] = reg.id("Hair Side L", ["Head"], Z["Hair Side"])
        P[front] = reg.id("Hair Front", ["Head"], Z["Hair Front"])

    # ---- body from the skeleton
    p = sc.pose
    bones = {}
    if p:
        def ok(k):  # limbs leaving the picture still count, they just end off-canvas
            q = p[k]
            return q[2] > 0.05 and -0.5 * W <= q[0] < 1.5 * W and -0.5 * H <= q[1] < 1.5 * H

        sl_, sr_ = np.array(p["shoulder_l"][:2]), np.array(p["shoulder_r"][:2])
        shoulder_w = max(np.linalg.norm(sl_ - sr_), 10)
        for s in ("L", "R"):
            k = s.lower()
            for name, a, b in (("Upper Arm", "shoulder", "elbow"), ("Forearm", "elbow", "wrist"), ("Hand", "wrist", "index"),
                               ("Thigh", "hip", "knee"), ("Shin", "knee", "ankle"), ("Foot", "ankle", "foot")):
                leg = name in ("Thigh", "Shin", "Foot")
                if ok(f"{a}_{k}") and (ok(f"{b}_{k}") or (leg and ok(f"hip_{k}"))):
                    bones[(name, s)] = (np.array(p[f"{a}_{k}"][:2]), np.array(p[f"{b}_{k}"][:2]))
            # knees the detector is unsure about (often off-picture): one piece per leg
            if p[f"knee_{k}"][2] < 0.3 and ("Thigh", s) in bones:
                A = bones.pop(("Thigh", s))[0]
                end = bones.pop(("Foot", s), None) or bones.pop(("Shin", s), None) or (None, np.array(p[f"knee_{k}"][:2]))
                bones.pop(("Shin", s), None)
                bones[("Leg", s)] = (A, end[1])
        hips_ok = ok("hip_l") and ok("hip_r")
        hip_y = (p["hip_l"][1] + p["hip_r"][1]) / 2 if hips_ok else None
        torso = np.array([p["shoulder_r"][:2], p["shoulder_l"][:2], p["hip_l"][:2], p["hip_r"][:2]], np.float32)
        tm = np.zeros((H, W), np.uint8)
        c = torso.mean(0)
        cv2.fillPoly(tm, [(c + (torso - c) * 1.15).astype(np.int32)], 1)
        torso_m = tm.astype(bool)
        # distance of every pixel to every bone; torso counts as distance 0 inside
        dists, names = [], []
        d_torso = ndi.distance_transform_edt(~torso_m)
        for (name, s), (A, B) in bones.items():
            r = shoulder_w * {"Hand": 0.16, "Foot": 0.16, "Upper Arm": 0.15, "Forearm": 0.13, "Thigh": 0.22, "Shin": 0.17, "Leg": 0.22}[name]
            dists.append(_seg_dist(xx, yy, A, B) / r)
            names.append((name, s))
        body_upper = reg.id("Upper Body", ["Body"], Z["Upper Body"])
        body_lower = reg.id("Lower Body", ["Body"], Z["Lower Body"])
        torso_scale = shoulder_w * 0.25
        dists.append(d_torso / torso_scale)
        names.append(("Torso", ""))
        D = np.stack(dists, 0)
        nearest = D.argmin(0)
        dmin = D.min(0)
        body_px = fg & (P == 0) & (O == 0)
        for i, (name, s) in enumerate(names):
            m = body_px & (nearest == i) & (dmin < 2.6)
            if name == "Torso":
                if hip_y is not None:
                    P[m & (yy < hip_y)] = body_upper
                    P[m & (yy >= hip_y)] = body_lower
                else:
                    P[m] = body_upper
            elif name in ("Upper Arm", "Forearm", "Hand"):
                P[m] = reg.id(f"{name} {s}", ["Body", f"Arm {s}"], Z[name])
            else:
                lid = reg.id(f"{name} {s}", ["Body", f"Leg {s}"], Z.get(name, Z["Thigh"]))
                if hip_y is not None and sc.face is not None:
                    # cloth just below the hips is the skirt / shorts, not the leg
                    A, B = bones[(name, s)]
                    reach = hip_y + 0.45 * max(1.0, abs(B[1] - A[1]))
                    cloth = m & ~skin_like & (yy < reach)
                    P[cloth] = body_lower
                    m = m & ~cloth
                P[m] = lid
        # SAM sharpens arms that stand out from the clothes behind them
        for (name, s), (A, B) in bones.items():
            if name not in ("Upper Arm", "Forearm", "Hand"):
                continue
            pts = [(A[0] + t * (B[0] - A[0]), A[1] + t * (B[1] - A[1])) for t in (0.35, 0.65)]
            pts = [q for q in pts if 0 <= q[0] < W and 0 <= q[1] < H and fg[int(q[1]), int(q[0])]]
            if not pts:
                continue
            m = sam.mask("body", char_box, pts, [tuple(torso.mean(0))])
            if m is None:
                continue
            tube = _seg_dist(xx, yy, A, B) < shoulder_w * 0.35
            if m.sum() < 2.5 * tube.sum() and (m & tube).sum() > 0.5 * m.sum():
                P[m & tube & (O == 0) & ~face_skin & ~hair] = reg.id(f"{name} {s}", ["Body", f"Arm {s}"], Z[name])
        # props: objects SAM sees as one thing that are long and thin (staff,
        # sword) or held in a hand, and lie mostly outside the body
        held = np.zeros((H, W), bool)
        for (name, s), (A, B) in bones.items():
            if name in ("Hand", "Forearm"):
                held |= _seg_dist(xx, yy, A, B) < shoulder_w * 0.45
        if sam.sam is not None:
            x0, y0, x1, y1 = [int(v) for v in char_box]
            crop = sc.rgb[y0:y1, x0:x1].copy()
            crop[~fg[y0:y1, x0:x1]] = 255
            body_core = (dmin < 1.8) | ndi.binary_dilation(face_full | hair, iterations=3)
            found = []
            for m, _ in sorted(sam.sam.everything(crop, fg[y0:y1, x0:x1], grid=20), key=lambda t: -t[0].sum()):
                full = np.zeros((H, W), bool)
                full[y0:y1, x0:x1] = m
                a = full.sum()
                if a < 0.002 * fg.sum() or a > 0.25 * fg.sum():
                    continue
                outside = (full & ~body_core).sum() / a
                skinny = _elongated(full)
                if outside > 0.7 and (skinny or (full & held).sum() > 0.05 * a) and (full & skin_like).sum() < 0.3 * a:
                    found.append(full)
            # pieces of one object split by a hand (a staff above and below the fist)
            groups = []
            for m in found:
                for g in groups:
                    if _same_object(sc.lab, g[0], m):
                        g.append(m)
                        break
                else:
                    groups.append([m])
            for i, g in enumerate(groups, start=1):
                O[np.logical_or.reduce(g) & (O == 0)] = reg.id(f"Prop {i}", ["Props"], Z["Prop"])
        # neck: skin between chin and shoulders
        if sc.face is not None:
            sh_y = (p["shoulder_l"][1] + p["shoulder_r"][1]) / 2
            neck = skin_like & ~face_skin & (yy > fy + 0.75 * fh) & (yy < sh_y + 0.1 * fh) & (np.abs(xx - fcx) < 0.45 * fw)
            P[_keep_touching(neck, ndi.binary_dilation(face_skin, iterations=4))] = reg.id("Neck", ["Body"], Z["Neck"])
        # props: things far from every bone (a staff, a sign)
        far = fg & (P == 0) & (O == 0) & (dmin >= 2.6) & ~hair
        # only things well away from the body are props; the rest (a belt end,
        # a skirt flare) joins the nearest body part
        lb, nlb = ndi.label(far)
        for i in range(1, nlb + 1):
            comp = lb == i
            if np.mean(dmin[comp] > 4.0) < 0.6:
                far[comp] = False  # filled from the neighbouring parts below
        lb, nlb = ndi.label(far)
        for i in range(1, nlb + 1):
            comp = lb == i
            if not (comp & held).any() and sc.had_background:
                sc.fg &= ~comp  # not held, not worn: background the matte let through
                sc.alpha[comp] = 0
                far[comp] = False
        if far.any():
            P[far] = reg.id("Prop", ["Props"], Z["Prop"])  # renamed per object below
    elif not creature:
        rest = fg & (P == 0) & (O == 0)
        if sc.face is not None:
            fx, fy, fw, fh = sc.face
            neck = rest & skin_like & ~face_skin & (yy > fy + 0.75 * fh) & (yy < fy + 1.5 * fh) & (np.abs(xx - (fx + fw / 2)) < 0.45 * fw)
            P[_keep_touching(neck, ndi.binary_dilation(face_skin, iterations=4))] = reg.id("Neck", ["Body"], Z["Neck"])
        rest = fg & (P == 0) & (O == 0)
        P[rest] = reg.id("Body", ["Body"], Z["Upper Body"])

    # ---- one label per region (edges follow the line art), exact overrides on top
    A = vote(atoms, P, len(reg.parts))
    A = np.where(O > 0, O, A)
    todo = fg & (A == 0)
    if todo.any() and (A > 0).any():
        _, idx = ndi.distance_transform_edt(A == 0, return_indices=True)
        A[todo] = A[idx[0][todo], idx[1][todo]]
    _split_props(A, reg)
    _ink_to_front(A, atoms, reg, fg)

    parts = []
    for i, d in enumerate(reg.parts, start=1):
        m = A == i
        if m.sum() < 20:
            continue
        hint = d.pop("hint", None)
        parts.append(SemPart(d["name"], d["path"], d["z"], m, sweep=d.get("sweep", True), fill=d.get("fill", "inpaint"),
                             hidden_hint=hint))
    _depth(sc, parts)
    return parts


def _iris_eyes(sam, head_box, sc, face_full, face_skin, box, skin_ref):
    """Two eyes from their irises: compact dark/saturated blobs at eye height,
    each cut out with MobileSAM."""
    if sam.sam is None:
        return []
    fx, fy, fw, fh = box
    H, W = face_full.shape
    lab = sc.lab
    yy, xx = np.mgrid[0:H, 0:W]
    v = (yy - fy) / fh
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    zone = ndi.binary_erosion(face_full, iterations=2) & (v > 0.15) & (v < 0.7)
    irisish = zone & ~face_skin & ((lab[..., 0] < 45) | (chroma > np.hypot(skin_ref[1], skin_ref[2]) + 18))
    lb, n = ndi.label(ndi.binary_opening(irisish, iterations=1))
    cands = []
    # second cue: upper lash lines, dark wide strokes (pale irises are common)
    lashes = []
    dlb, dn = ndi.label(zone & (lab[..., 0] < 40))
    for i, sl in enumerate(ndi.find_objects(dlb), start=1):
        if sl is None:
            continue
        m = dlb[sl] == i
        h_, w_ = m.shape
        if 0.12 * fw < w_ < 0.45 * fw and w_ > 1.4 * h_ and m.sum() > 0.001 * fw * fh:
            ys, xs = np.nonzero(m)
            lashes.append((float(m.sum()), xs.mean() + sl[1].start, ys.mean() + sl[0].start + 0.35 * w_))
    for i, sl in enumerate(ndi.find_objects(lb), start=1):
        if sl is None:
            continue
        m = lb[sl] == i
        a = m.sum()
        h_, w_ = m.shape
        if a < 0.002 * fw * fh or a > 0.06 * fw * fh:
            continue
        fill = a / (h_ * w_)
        aspect = max(h_, w_) / max(1, min(h_, w_))
        if fill < 0.35 or aspect > 3:
            continue
        ys, xs = np.nonzero(m)
        cands.append((a * fill, xs.mean() + sl[1].start, ys.mean() + sl[0].start))
    def pair(cs):
        # best pair: similar height, a plausible distance apart
        best, score = None, -1
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                a, b = cs[i], cs[j]
                dx, dy = abs(a[1] - b[1]), abs(a[2] - b[2])
                if dx < 0.2 * fw or dx > 0.8 * fw or dy > 0.25 * fh:
                    continue
                sc_ = min(a[0], b[0]) / (1 + dy / fh)
                if sc_ > score:
                    best, score = (a, b), sc_
        return best

    best = pair(cands) or pair(lashes)
    if best is None:
        return []
    out = []
    for _, x, y in best:
        masks = []
        for pref in ("small", "mid", "large"):
            m = sam.mask("head", head_box, [(x, y)], prefer=pref)
            if m is not None:
                m &= ndi.binary_dilation(face_full, iterations=3)
                if 0.004 * fw * fh < m.sum() < 0.07 * fw * fh:
                    masks.append(m)
        if not masks:
            return []
        m = max(masks, key=lambda q: q.sum())
        ys, xs = np.nonzero(m)
        out.append(dict(m=m, a=int(m.sum()), u=(xs.mean() - fx) / fw, v=(ys.mean() - fy) / fh, bh=np.ptp(ys) + 1, bw=np.ptp(xs) + 1))
    return out


def _creature(sc, atoms, sam, reg, P, O, log) -> bool:
    """Animals and mascots: find the eyes, get the head from SAM, then ears,
    mouth, brows, body and tail by shape.  Returns False when no eye pair is found."""
    H, W = sc.fg.shape
    fg = sc.fg
    lab = sc.lab
    L = lab[..., 0]
    yy, xx = np.mgrid[0:H, 0:W]
    ys, xs = np.nonzero(fg)
    bw, bh, area_fg = np.ptp(xs) + 1, np.ptp(ys) + 1, fg.sum()
    r = max(1, int(round(atoms.hw * 1.5)))
    solid = atoms.blob | ndi.binary_opening(fg & (L < 42), structure=np.ones((2 * r + 1, 2 * r + 1)))
    solid = ndi.binary_fill_holes(solid | (ndi.binary_fill_holes(solid) & (L > 80)))  # highlights inside pupils
    lb, n = ndi.label(solid)
    cands = []
    for i, sl in enumerate(ndi.find_objects(lb), start=1):
        if sl is None:
            continue
        m = lb[sl] == i
        a = m.sum()
        h_, w_ = m.shape
        if not (0.0008 * area_fg < a < 0.04 * area_fg):
            continue
        if a / (h_ * w_) < 0.5 or max(h_, w_) / min(h_, w_) > 1.8:
            continue
        cy, cx = sl[0].start + h_ / 2, sl[1].start + w_ / 2
        if cy > ys.min() + 0.75 * bh:
            continue
        cands.append((a, cx, cy, i))
    best, score = None, -1
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            a, b = cands[i], cands[j]
            dx, dy = abs(a[1] - b[1]), abs(a[2] - b[2])
            if max(a[0], b[0]) > 2.2 * min(a[0], b[0]) or dy > 0.15 * bh or not (0.08 * bw < dx < 0.6 * bw):
                continue
            sc_ = min(a[0], b[0]) / (1 + 3 * dy / bh) * (1.5 - (a[2] + b[2]) / 2 / H)
            if sc_ > score:
                best, score = (a, b), sc_
    if best is None:
        log("  no eyes found: the whole character is one body layer")
        return False
    e1, e2 = sorted(best, key=lambda c: c[1])
    d = float(np.hypot(e2[1] - e1[1], e2[2] - e1[2]))
    mid = np.array([(e1[1] + e2[1]) / 2, (e1[2] + e2[2]) / 2])
    sc.face = (mid[0] - 1.1 * d, mid[1] - 0.9 * d, 2.2 * d, 2.0 * d)
    log("  creature mode: found a pair of eyes")

    eye_masks = []
    for e in (e1, e2):
        base = ndi.binary_dilation(lb == e[3], iterations=max(1, int(atoms.hw)))
        m = None
        if sam.sam is not None:
            for pref in ("small", "mid"):
                q = sam.mask("full", (0, 0, W, H), [(e[1], e[2])], prefer=pref)
                if q is not None and 0.6 * base.sum() < q.sum() < 4 * base.sum():
                    m = q | base
                    break
        eye_masks.append(m if m is not None else base)

    # head: SAM clicked on both eyes and the nose, told the body below is not it
    # chibi animals: the eyes sit about mid-head and the chin is ~0.65 eye-distances below them
    chin = mid[1] + 0.65 * d
    prior = fg & (yy < chin) & (np.abs(xx - mid[0]) < 1.45 * d)
    pl, pn = ndi.label(prior)
    if pn:
        keep = np.unique(pl[int(mid[1]), max(0, int(e1[1])) : int(e2[1]) + 1])
        prior = np.isin(pl, keep[keep > 0])
    head = None
    if sam.sam is not None:
        pos = [(e1[1], e1[2]), (e2[1], e2[2]), (mid[0], mid[1] + 0.45 * d)]
        below = (mid[0], min(ys.max() - 0.08 * bh, mid[1] + 2.2 * d))
        neg = [below] if below[1] > mid[1] + 1.4 * d and fg[int(below[1]), int(below[0])] else []
        best_iou = 0.6
        for pref in ("small", "mid", "large"):
            q = sam.mask("full", (0, 0, W, H), pos, neg, prefer=pref)
            if q is None or not all(q[int(e[2]), int(e[1])] for e in (e1, e2)):
                continue
            # the head is the mask that best matches a head-sized oval around the eyes
            iou = (q & prior).sum() / max(1, (q | prior).sum())
            if iou > best_iou:
                head, best_iou = q, iou
    if head is None:
        head = prior
    head = ndi.binary_fill_holes(head) & fg
    # whiskers and strokes poking out of the head go with it
    near = ndi.binary_dilation(head, iterations=max(2, int(0.35 * d))) & fg & ~head & (L < 50) & (yy < mid[1] + 1.2 * d)
    head |= near & ~ndi.binary_dilation(fg & ~head & (L >= 50), iterations=1)

    # ears: what sticks out of the round skull above the eyes
    k = max(3, int(0.3 * d))
    disk = (np.add.outer(np.arange(-k, k + 1) ** 2, np.arange(-k, k + 1) ** 2) <= k * k)
    skull = ndi.binary_opening(head, structure=disk)
    ears = head & ~skull & (yy < mid[1] - 0.45 * d)
    ears = ndi.binary_opening(ears, iterations=2)
    ear_list = []
    for half in (xx < mid[0], xx >= mid[0]):  # one ear per side even if they touch
        el, en = ndi.label(ears & half)
        if en:
            sizes = np.bincount(el.ravel())[1:]
            m = el == sizes.argmax() + 1
            if m.sum() > 0.003 * area_fg:
                ear_list.append(m)
    for i, m in enumerate(ear_list[:2]):
        side = ("R" if i == 0 else "L") if len(ear_list) >= 2 else ("R" if np.nonzero(m)[1].mean() < mid[0] else "L")
        P[m] = reg.id(f"Ear {side}", ["Head"], Z["Ear"])

    # facial features: not fur-coloured, inside the head
    eyes_all = eye_masks[0] | eye_masks[1]
    fur_px = lab[head & (L > 45) & ~eyes_all]
    fur = np.median(fur_px, 0) if len(fur_px) else np.array([90.0, 0, 0])
    nonfur = head & ~eyes_all & (np.linalg.norm(lab - fur, axis=-1) > 18)
    fl, fn = ndi.label(nonfur)
    for i, sl in enumerate(ndi.find_objects(fl), start=1):
        if sl is None:
            continue
        m = fl == i
        a = m.sum()
        if a < 0.0003 * area_fg:
            continue
        cy, cx = ndi.center_of_mass(m)
        h_, w_ = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if max(h_, w_) > 0.9 * d:  # the outline network, not a feature
            continue
        is_dark = np.median(L[m]) < 50
        if cy < mid[1] - 0.15 * d and is_dark and w_ > 1.4 * h_ and abs(cx - mid[0]) < 1.1 * d:
            side = "R" if cx < mid[0] else "L"
            O[m] = reg.id(f"Brow {side}", ["Head"], Z["Brow"], sweep=False)
        elif mid[1] < cy < mid[1] + 1.3 * d and abs(cx - mid[0]) < 0.6 * d:
            O[m] = reg.id("Mouth", ["Head"], Z["Mouth"], sweep=False)
    for i, (m, side) in enumerate(zip(eye_masks, ("R", "L"))):
        O[m & (O == 0)] = reg.id(f"Eye White {side}", ["Head", f"Eye {side}"], Z["Eye White"], sweep=False, fill="color")
    P[head & (P == 0)] = reg.id("Face", ["Head"], Z["Face"], fill="color")

    # body and tail: the tail is what sticks out of the main body blob
    body = fg & ~head
    bys, bxs = np.nonzero(body)
    if len(bys):
        k = max(3, int(0.15 * np.sqrt(body.sum())))
        disk = (np.add.outer(np.arange(-k, k + 1) ** 2, np.arange(-k, k + 1) ** 2) <= k * k)
        core = ndi.binary_opening(body, structure=disk)
        tl, tn = ndi.label(body & ~ndi.binary_dilation(core, iterations=2))
        tails = [tl == i for i in range(1, tn + 1)]
        tails = [m for m in tails if 0.01 * area_fg < m.sum() < 0.25 * body.sum() and _elongated_soft(m)]
        if tails:
            t = max(tails, key=lambda m: m.sum())
            P[t] = reg.id("Tail", ["Body"], 10)
        P[body & (P == 0)] = reg.id("Body", ["Body"], Z["Upper Body"])
    return True


def _elongated_soft(m):
    ys, xs = np.nonzero(m)
    if len(ys) < 10:
        return False
    ev = np.linalg.eigvalsh(np.cov(np.stack([xs, ys]).astype(np.float32)))
    return ev[1] > 3 * max(ev[0], 1e-3)


def _sam_eyes(sam, head_box, sc, found, face_full, box):
    """Find eyes the colour rules missed (painted eyes, irises the colour of the hair).

    The missing eye is guessed by mirroring a found one across the face, or
    from where eyes sit in an anime face box, and MobileSAM cuts it out.
    """
    if sam.sam is None:
        return []
    fx, fy, fw, fh = box
    ang = np.radians(sc.face_angle)
    cx, cy = fx + fw / 2, fy + fh / 2

    def at(u, v):  # face-box coordinates -> pixels, following the head tilt
        dx, dy = (u - 0.5) * fw, (v - 0.5) * fh
        return (cx + dx * np.cos(ang) + dy * np.sin(ang), cy - dx * np.sin(ang) + dy * np.cos(ang))

    if found:
        u = 1.0 - found[0]["u"]
        guesses = [(u, found[0]["v"])]
    else:
        guesses = [(0.3, 0.5), (0.7, 0.5)]
    out = []
    for u, v in guesses:
        x, y = at(u, v)
        m = sam.mask("head", head_box, [(x, y)], prefer="small")
        if m is None:
            continue
        m &= ndi.binary_dilation(face_full, iterations=3)
        a = int(m.sum())
        if not (0.003 * fw * fh < a < 0.08 * fw * fh):
            continue
        yy, xx = np.nonzero(m)
        out.append(dict(m=m, a=a, u=(xx.mean() - fx) / fw, v=(yy.mean() - fy) / fh, bh=np.ptp(yy) + 1, bw=np.ptp(xx) + 1))
    return out


def _person_mask(sc, sam):
    """SAM's idea of the whole character, clicked on the body keypoints and face."""
    if sam.sam is None:
        return None
    H, W = sc.fg.shape
    pts = []
    if sc.pose:
        for k in ("nose", "shoulder_l", "shoulder_r", "elbow_l", "elbow_r", "wrist_l", "wrist_r", "hip_l", "hip_r"):
            x, y, v = sc.pose[k]
            if v > 0.5 and 0 <= x < W and 0 <= y < H:
                pts.append((x, y))
    if sc.face is not None:
        x, y, w, h = sc.face
        pts.append((x + w / 2, y + 0.7 * h))
    if len(pts) < 2:
        return None
    m = sam.mask("full", (0, 0, W, H), pts, prefer="large")
    return None if m is None else ndi.binary_dilation(m, iterations=3)


def sc_line(atoms):
    return atoms.line


def _eye_parts(reg, O, m, side, lab, prefix=(), dz=0.0):
    L = lab[..., 0]
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    path = list(prefix) + ["Head", f"Eye {side}"]
    m = ndi.binary_fill_holes(m)
    white = m & (L > 75) & (chroma < 20)
    darkp = m & (L < 32)
    ys, xs = np.nonzero(m)
    top, bot = ys.min(), ys.max()
    lab_d, nd = ndi.label(darkp)
    lash = np.zeros_like(m)
    for i in range(1, nd + 1):
        c = lab_d == i
        cy, cx = np.nonzero(c)
        if cy.min() <= top + 0.3 * (bot - top) and np.ptp(cx) > 0.35 * max(1, np.ptp(xs)):
            lash |= c
    iris = m & ~white & ~lash
    il, ni = ndi.label(iris)
    if ni:
        sizes = np.bincount(il.ravel())[1:]
        iris = ndi.binary_fill_holes(il == sizes.argmax() + 1) & ~lash
    lines = m & ~white & ~iris
    # complete what is hidden: the eye white is an ellipse, the iris a disc
    white_full = m | _fit_ellipse(m)
    iris_full = iris | (_fit_circle(iris, white) & ndi.binary_dilation(white_full, iterations=2)) if iris.any() else iris
    O[m] = reg.id(f"Eye White {side}", path, Z["Eye White"] + dz, sweep=False, fill="color", hint=white_full)
    O[iris] = reg.id(f"Iris {side}", path, Z["Iris"] + dz, sweep=False, hint=iris_full)
    O[lines | lash] = reg.id(f"Eye Line {side}", path, Z["Eye Line"] + dz, sweep=False)


def _fit_ellipse(m):
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros(m.shape, np.uint8)
    if cs:
        c = max(cs, key=len)
        if len(c) >= 5:
            cv2.ellipse(out, cv2.fitEllipse(c), 1, -1)
    return out.astype(bool)


def _fit_circle(iris, white):
    """Least-squares circle through the iris edge that borders the eye white
    (edges against lashes or hair are where it is cut off, not its real rim)."""
    edge = iris & ~ndi.binary_erosion(iris)
    rim = edge & ndi.binary_dilation(white, iterations=2)
    pts = np.argwhere(rim if rim.sum() >= 12 else edge).astype(np.float64)
    out = np.zeros(iris.shape, np.uint8)
    if len(pts) < 6:
        return iris
    y, x = pts[:, 0], pts[:, 1]
    A = np.stack([x, y, np.ones_like(x)], 1)
    b = x * x + y * y
    (c0, c1, c2), *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = c0 / 2, c1 / 2
    r = np.sqrt(max(c2 + cx * cx + cy * cy, 1.0))
    ys, xs = np.nonzero(iris)
    if r > 1.5 * max(np.ptp(xs), np.ptp(ys), 1):  # nonsense fit
        return iris
    cv2.circle(out, (int(round(cx)), int(round(cy))), int(round(r)), 1, -1)
    return out.astype(bool)


def _split_props(A, reg):
    """Turn the single "Prop" label into one part per connected object."""
    pid = next((i for i, d in enumerate(reg.parts, start=1) if d["name"] == "Prop"), None)
    if pid is None:
        return
    m = A == pid
    lb, n = ndi.label(ndi.binary_dilation(m, iterations=3))
    lb = np.where(m, lb, 0)
    sizes = np.bincount(lb.ravel())[1:] if n else []
    order = np.argsort(-np.asarray(sizes)) if n else []
    k = sum(d["name"].startswith("Prop ") for d in reg.parts)  # SAM found these already
    first = True
    for j in order:
        comp = lb == j + 1
        if sizes[j] < 0.002 * A.size:
            # crumbs far from the body: give them to their neighbours later
            A[comp] = 0
            continue
        k += 1
        if first:
            reg.parts[pid - 1]["name"] = f"Prop {k}"
            first = False
            continue
        A[comp] = reg.id(f"Prop {k}", ["Props"], Z["Prop"])
    if (A == 0).any():
        fgm = (A > 0)
        _, idx = ndi.distance_transform_edt(~fgm, return_indices=True)
        crumbs = m & (A == 0)
        A[crumbs] = A[idx[0][crumbs], idx[1][crumbs]]


def _ink_to_front(A, atoms, reg, fg):
    """Each ink stroke belongs to the frontmost part it borders."""
    ink = atoms.line & fg
    if not ink.any():
        return
    r = int(np.ceil(2 * atoms.hw + 1))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    owner = np.zeros(A.shape, np.int32)
    order = sorted(range(1, len(reg.parts) + 1), key=lambda i: -reg.parts[i - 1]["z"])
    for i in order:
        m = (A == i) & ~ink
        if not m.any():
            continue
        near = cv2.dilate(m.astype(np.uint8), k).astype(bool) & ink & (owner == 0)
        owner[near] = i
    # strokes already labelled by an exact override (eye lines) keep it
    keep = np.isin(A, [i for i, d in enumerate(reg.parts, start=1) if not d.get("sweep", True)]) & ink
    A[ink & ~keep & (owner > 0)] = owner[ink & ~keep & (owner > 0)]


def _convex(mask):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return mask
    out = np.zeros(mask.shape, np.uint8)
    cv2.fillPoly(out, [cv2.convexHull(np.vstack(cs))], 1)
    return out.astype(bool)


def _dominant(lab, mask, k=3):
    px = lab[mask]
    if len(px) < 20:
        return None
    px = px[:: max(1, len(px) // 20000)].astype(np.float32)
    k = min(k, len(px))
    _, lbl, cen = cv2.kmeans(px, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5), 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(lbl.ravel(), minlength=k)
    return cen[counts.argmax()]


def _spread_points(mask, n):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return []
    dt = ndi.distance_transform_edt(mask)
    order = np.argsort(-dt[ys, xs])
    pts = []
    for i in order:
        p = (float(xs[i]), float(ys[i]))
        if all(np.hypot(p[0] - q[0], p[1] - q[1]) > 0.15 * max(np.ptp(xs), np.ptp(ys), 1) for q in pts):
            pts.append(p)
        if len(pts) >= n:
            break
    return pts


def _same_object(lab, a, b):
    """Two masks of one long object: same colour and lying along the same line."""
    if np.linalg.norm(lab[a].mean(0) - lab[b].mean(0)) > 15:
        return False
    axes = []
    for m in (a, b):
        ys, xs = np.nonzero(m)
        pts = np.stack([xs, ys], 1).astype(np.float32)
        c = pts.mean(0)
        w, v = np.linalg.eigh(np.cov((pts - c).T))
        axes.append((c, v[:, 1], np.sqrt(max(w[0], 1.0))))
    (ca, va, wa), (cb, vb, wb) = axes
    if abs(float(va @ vb)) < 0.97:
        return False
    off = (cb - ca) - ((cb - ca) @ va) * va
    return float(np.linalg.norm(off)) < 3 * max(wa, wb)


def _elongated(m):
    ys, xs = np.nonzero(m)
    if len(ys) < 10:
        return False
    pts = np.stack([xs, ys], 1).astype(np.float32)
    ev = np.linalg.eigvalsh(np.cov(pts.T))
    return ev[1] > 25 * max(ev[0], 1e-3)


def _depth(sc, parts):
    """Let arms go behind the body when the outlines say so."""
    from .parts import occlusion_score  # T-junction test

    class _S:
        pass

    body = [q for q in parts if q.name in ("Top", "Bottom", "Body Skin", "Upper Body", "Lower Body")]
    if not body:
        return
    lbl = np.zeros(sc.fg.shape, np.int32)
    for i, q in enumerate(parts, start=1):
        lbl[q.mask] = i
    s = _S()
    s.labels = lbl
    s.line_halfwidth = max(1.5, sc.fg.shape[0] / 800)
    edge = cv2.Canny(cv2.cvtColor(sc.rgb, cv2.COLOR_RGB2GRAY), 60, 150) > 0
    s.line = ndi.binary_dilation(edge, iterations=1)
    body_ids = [parts.index(q) + 1 for q in body]
    for side in ("L", "R"):
        arm = [q for q in parts if q.path[-1:] == [f"Arm {side}"]]
        if not arm:
            continue
        sc_ = occlusion_score(s, [parts.index(q) + 1 for q in arm], body_ids)
        if sc_ < -0.3:
            for q in arm:
                q.z -= 50
