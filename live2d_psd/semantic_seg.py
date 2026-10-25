"""Character parsing driven by the learned anime body-part segmenter.

See-through's SAM-HQ model (partseg.py) labels every pixel with one of 19
body-part classes.  This module turns that into Live2D parts:

* several characters in one picture are separated (one region per face,
  grown along the drawing's edges) and each gets its own folder and rig;
* hair is split into front / side / back by where it sits on the face;
* eyes are split per side into eye white, iris and eye line, with the hidden
  parts completed; brows, nose, mouth, ears, earrings, eyewear, headwear;
* clothes and limbs: top / bottom, arms (upper arm, forearm, hand, using body
  keypoints when they are found), legs and feet left/right, tail, wings,
  held objects;
* labels are snapped to the line-art regions so part edges follow the lines,
  and each ink stroke goes to the part in front of it.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from . import models, partseg
from .semantic import (
    Scene, SemPart, _Reg, _convex, _depth, _eye_parts, _ink_to_front, _orient_pose, _seg_dist, _Sam,
)

C = {n: i for i, n in enumerate(partseg.CLASSES)}

ZT = {
    "Hair Back": 2, "Tail": 5, "Wing": 6, "Leg": 20, "Foot": 21, "Neck": 30, "Lower Body": 40, "Upper Body": 50,
    "Neckwear": 55, "Upper Arm": 60, "Forearm": 61, "Hand": 62, "Prop": 70, "Hair Side": 72, "Ear": 75,
    "Earring": 76, "Face": 80, "Nose": 83, "Mouth": 84, "Eyewear": 89, "Brow": 90, "Hair Front": 95, "Headwear": 96,
}


def parse_learned(sc: Scene, log=print) -> list[SemPart] | None:
    from .atoms import background_atoms, make_atoms, vote

    log("  anime part segmentation (See-through SAM-HQ)")
    logits = partseg.predict(sc.rgb, log)
    if logits is None:
        return None
    H, W = sc.fg.shape
    lab = sc.lab
    if _crowded(logits, H, W):
        log("  several characters or a wide picture: looking again at overlapping crops")
        logits = partseg.predict_tiled(sc.rgb, logits, log)
    conf = logits.max(0)
    cls_all = logits.argmax(0)
    labelled = conf > 0
    face_px = ((cls_all == C["face"]) & labelled & sc.fg).sum()
    if face_px < 0.002 * max(1, sc.fg.sum()):
        log("  no anime face found by the segmenter; trying the other parsers")
        return None

    # the segmenter is also a strong "this is a character" cue: background
    # removal keeps only the main subject, so other characters in the
    # picture come back here (any confident region holding a face or touching
    # the subject)
    if sc.had_background:
        sure = ndi.binary_opening(labelled & (conf > 0.5), iterations=1)
        lb, n = ndi.label(ndi.binary_closing(sure, iterations=2))
        if n:
            faces = (cls_all == C["face"]) & (conf > 1)
            near = ndi.binary_dilation(sc.fg, iterations=4)
            hit_face = np.zeros(n + 1, bool)
            hit_face[np.unique(lb[faces])] = True
            hit_fg = np.zeros(n + 1, bool)
            hit_fg[np.unique(lb[near])] = True
            keep = (hit_face | hit_fg)
            keep[0] = False
            add = keep[lb] & sure & ~sc.fg
            if add.any():
                log(f"  the part model found {int(add.sum())} more character pixels than background removal")
                sc.fg |= add
                sc.alpha = np.maximum(sc.alpha, add.astype(np.float32))
    atoms = make_atoms(sc.rgb, sc.fg, lab)
    if sc.had_background:
        drop = background_atoms(atoms, lab, sc.alpha, protect=ndi.binary_dilation(labelled, iterations=3))
        if drop.any():
            atoms.labels[drop] = 0
            keep = atoms.labels > 0
            sc.fg = keep | (atoms.line & ndi.binary_dilation(keep, iterations=int(2 * atoms.hw + 2)))
            sc.alpha = np.where(sc.fg, sc.alpha, 0).astype(np.float32)
            atoms.line &= sc.fg
    fg = sc.fg

    # every character pixel gets a class; where the model is silent, learn from
    # the pixels it is sure about in this very picture (colour + position)
    cls = np.where(labelled & fg, cls_all, -1)
    cls = _self_train(cls, lab, fg, conf)

    insts = _instances(sc, np.where(labelled & (conf > 0.5), cls_all, -1), log)
    log(f"  {len(insts)} character{'s' if len(insts) != 1 else ''}")
    reg = _Reg()
    P = np.zeros((H, W), np.int32)
    O = np.zeros((H, W), np.int32)
    sam = _Sam(sc, lambda *a: None)
    for rank, inst in enumerate(insts):
        prefix = [f"Character {rank + 1}"] if len(insts) > 1 else []
        dz = 100.0 * (len(insts) - 1 - rank)  # bigger face = closer
        _character(sc, cls, inst, prefix, dz, reg, P, O, sam, log)

    A = vote(atoms, P, len(reg.parts))
    A = np.where(O > 0, O, A)
    todo = fg & (A == 0)
    if todo.any() and (A > 0).any():
        _, idx = ndi.distance_transform_edt(A == 0, return_indices=True)
        A[todo] = A[idx[0][todo], idx[1][todo]]
    _ink_to_front(A, atoms, reg, fg)

    parts = []
    for i, d in enumerate(reg.parts, start=1):
        m = A == i
        if m.sum() < 20:
            continue
        hint = d.get("hint")
        parts.append(SemPart(d["name"], d["path"], d["z"], m, sweep=d.get("sweep", True), fill=d.get("fill", "inpaint"),
                             hidden_hint=hint))
    _depth(sc, parts)
    return parts


def _crowded(logits, H, W) -> bool:
    if max(H, W) < 1.3 * min(H, W):
        return False
    face = (logits.argmax(0) == C["face"]) & (logits.max(0) > 0)
    lb, n = ndi.label(ndi.binary_closing(face, iterations=3))
    if n == 0:
        return True  # nothing found at full size: maybe the characters are small
    sizes = np.bincount(lb.ravel())[1:]
    big = sizes[sizes > 0.1 * sizes.max()]
    small_face = np.sqrt(sizes.max()) < 0.12 * min(H, W)
    return len(big) >= 2 or small_face


_SMALL = {C[n] for n in ("face", "eyes", "eyewear", "ears", "earwear", "nose", "mouth")}


def _self_train(cls, lab, fg, conf):
    """Fill unlabelled character pixels from per-class colour/position clusters
    of the confidently labelled pixels of the same picture."""
    import cv2

    H, W = cls.shape
    todo = fg & (cls < 0)
    if not todo.any() or not (cls >= 0).any():
        return cls
    yy, xx = np.mgrid[0:H, 0:W]
    pos_w = 70.0  # crossing the whole picture costs as much as a large colour change
    feats = np.dstack([lab[..., 0], lab[..., 1] * 1.3, lab[..., 2] * 1.3, xx / W * pos_w, yy / H * pos_w]).astype(np.float32)
    centers, owners = [], []
    sure = (cls >= 0) & (conf > 1.0)
    for c in np.unique(cls[sure]):
        if c in _SMALL and c != C["face"]:  # eyes, mouths, ... are only where the model put them
            continue
        px = feats[sure & (cls == c)]
        if len(px) < 30:
            continue
        px = px[:: max(1, len(px) // 4000)]
        k = int(min(6, max(1, len(px) // 150)))
        _, _, cen = cv2.kmeans(px, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 0.5), 2, cv2.KMEANS_PP_CENTERS)
        centers.append(cen)
        owners += [c] * len(cen)
    if not centers:
        _, idx = ndi.distance_transform_edt(cls < 0, return_indices=True)
        return np.where(fg, cls[idx[0], idx[1]], -1)
    cen = np.vstack(centers)
    owners = np.array(owners)
    q = feats[todo]
    d = ((q[:, None, :] - cen[None]) ** 2).sum(-1)
    out = cls.copy()
    out[todo] = owners[d.argmin(1)]
    out[~fg] = -1
    return out


# --------------------------------------------------------------------------- instances


def _instances(sc, cls, log):
    """One mask per character: seeds are the faces, regions grow along the drawing."""
    fg = sc.fg
    face = (cls == C["face"]) | (cls == C["eyes"]) | (cls == C["mouth"]) | (cls == C["nose"])
    lb, n = ndi.label(ndi.binary_closing(face & fg, iterations=3))
    if n == 0:
        return [dict(mask=fg, face=None)]
    sizes = np.bincount(lb.ravel(), minlength=n + 1)[1:]
    eyes = np.bincount(lb[cls == C["eyes"]], minlength=n + 1)[1:]
    # a face seed is a sizeable face with eyes (or a big one: a face seen from
    # behind / with closed eyes); stray skin bits are not characters
    keep = [i + 1 for i in np.argsort(-sizes)
            if sizes[i] > max(0.12 * sizes.max(), 0.0008 * fg.sum()) and (eyes[i] > 0.01 * sizes[i] or sizes[i] > 0.5 * sizes.max())]
    faces = [lb == k for k in keep]
    if not faces:
        return [dict(mask=fg, face=None)]
    if len(faces) == 1:
        return [dict(mask=fg, face=faces[0])]
    from skimage.filters import sobel
    from skimage.graph import MCP_Geometric

    # crossing a drawn line or a colour edge is what separates characters, not
    # distance: a long sleeve still belongs to the body it is drawn onto
    edge = np.max([sobel(ndi.gaussian_filter(sc.lab[..., i], 1.0)) * w for i, w in ((0, 1.0), (1, 1.5), (2, 1.5))], 0)
    edge = np.clip(edge / max(np.percentile(edge[fg], 99), 1e-6), 0, 1)
    cost = 0.05 + 10.0 * edge ** 1.5
    cost[~fg] = 1e6  # don't travel through the background
    dists = []
    for f in faces:
        mcp = MCP_Geometric(cost)
        starts = list(zip(*np.nonzero(f)))[:: max(1, int(f.sum() // 400))]
        d, _ = mcp.find_costs(starts)
        # a bigger face means a bigger body: reach scales with the face size
        dists.append(d / f.sum() ** 0.25)
    owner = np.argmin(np.stack(dists), 0)
    out = []
    for k, f in enumerate(faces):
        m = fg & (owner == k)
        out.append(dict(mask=m, face=f))
    # front to back: the biggest face is usually the closest character
    out.sort(key=lambda d: -d["face"].sum())
    return out


# --------------------------------------------------------------------------- one character


def _character(sc, cls, inst, prefix, dz, reg, P, O, sam, log):
    H, W = cls.shape
    lab = sc.lab
    yy, xx = np.mgrid[0:H, 0:W]
    M = inst["mask"]

    def c(name):
        return (cls == C[name]) & M

    def pid(name, folder, zkey, **kw):
        return reg.id(name, list(prefix) + folder, ZT[zkey] + dz, **kw)

    seed = inst.get("face")
    if seed is None:
        seed = c("face") | c("eyes") | c("nose") | c("mouth")
    if not seed.any():
        P[M & (P == 0)] = pid("Body", ["Body"], "Upper Body")
        return
    # face features only count around this character's own face
    ys, xs = np.nonzero(seed)
    sz = max(xs.max() - xs.min(), ys.max() - ys.min()) + 1
    zone = np.zeros((H, W), bool)
    zone[max(0, ys.min() - sz // 3): ys.max() + sz // 3 + 1, max(0, xs.min() - sz // 3): xs.max() + sz // 3 + 1] = True
    M_head = M & zone

    def c(name, _c=c):
        m = _c(name)
        return m & M_head if C[name] in _SMALL else m

    face = c("face")
    feat = face | c("eyes") | c("nose") | c("mouth")
    if not feat.any():
        P[M & (P == 0)] = pid("Body", ["Body"], "Upper Body")
        return
    ys, xs = np.nonzero(feat)
    fx0, fx1, fy0, fy1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    fw, fh = max(1, fx1 - fx0), max(1, fy1 - fy0)
    fcx = (fx0 + fx1) / 2
    face_full = _convex(ndi.binary_fill_holes(feat))

    # ---- hair: front (over the face, incl. bangs above it), side, back
    hair = _real_hair(c("hair"), lab, fy0, fy1, fx0, fx1, yy)
    not_hair = c("hair") & ~hair  # goes to the clothes below
    bang_zone = face_full | ((yy < fy0 + 0.5 * fh) & (yy > fy0 - 0.45 * fh) & (xx > fx0 + 0.08 * fw) & (xx < fx1 - 0.08 * fw))
    front = hair & bang_zone & (yy < fy1 - 0.1 * fh)
    near_face = ndi.binary_dilation(face_full, iterations=max(2, int(0.3 * fw)))
    side = hair & ~front & near_face & (yy < fy1 + 0.4 * fh) & (yy > fy0)
    P[hair] = pid("Hair Back", ["Hair Back"], "Hair Back")
    P[side & (xx < fcx)] = pid("Hair Side R", ["Head"], "Hair Side")
    P[side & (xx >= fcx)] = pid("Hair Side L", ["Head"], "Hair Side")
    P[front] = pid("Hair Front", ["Head"], "Hair Front")
    P[c("headwear")] = pid("Headwear", ["Head"], "Headwear")

    # ---- face and features
    P[face] = pid("Face", ["Head"], "Face", fill="color")
    eyes = c("eyes")
    groups = {"R": eyes & (xx < fcx), "L": eyes & (xx >= fcx)}
    for side_, g in groups.items():
        if g.sum() < 10:
            continue
        lb, n = ndi.label(ndi.binary_closing(g, iterations=2))
        if n == 0:
            continue
        sizes = np.bincount(lb.ravel())[1:]
        main = lb == sizes.argmax() + 1
        my0 = np.nonzero(main)[0].min()
        brow = np.zeros_like(g)
        for i in range(1, n + 1):
            comp = lb == i
            if comp is main or (comp & main).any():
                continue
            cy = np.nonzero(comp)[0].mean()
            if cy < my0:
                brow |= comp
        eye = g & ~brow
        _eye_parts(reg, O, ndi.binary_fill_holes(eye | (main & ~brow)), side_, lab, prefix=prefix, dz=dz)
        if brow.any():
            O[brow & g] = pid(f"Brow {side_}", ["Head"], "Brow", sweep=False)
    O[c("nose") & (O == 0)] = pid("Nose", ["Head"], "Nose", sweep=False)
    O[c("mouth") & (O == 0)] = pid("Mouth", ["Head"], "Mouth", sweep=False)
    O[c("eyewear") & (O == 0)] = pid("Eyewear", ["Head"], "Eyewear", sweep=False)
    for name, key in (("ears", "Ear"), ("earwear", "Earring")):
        m = c(name)
        P[m & (xx < fcx)] = pid(f"{key} R", ["Head"], key)
        P[m & (xx >= fcx)] = pid(f"{key} L", ["Head"], key)

    # ---- neck, clothes, limbs
    P[c("neck")] = pid("Neck", ["Body"], "Neck")
    P[c("neckwear")] = pid("Neckwear", ["Body"], "Neckwear")
    pose = models.detect_pose(sc.rgb, M)
    if pose:
        _orient_pose(pose)
    body_cx = fcx
    top = c("topwear") | not_hair
    upper = top | c("handwear")
    if pose and _limbs(pose, upper, c("handwear"), P, pid, xx, yy, W, H):
        pass
    else:
        P[top] = pid("Upper Body", ["Body"], "Upper Body")
        hw = c("handwear")
        P[hw & (xx < body_cx)] = pid("Hand R", ["Body", "Arm R"], "Hand")
        P[hw & (xx >= body_cx)] = pid("Hand L", ["Body", "Arm L"], "Hand")
    P[c("bottomwear")] = pid("Lower Body", ["Body"], "Lower Body")
    legs = c("legwear") | c("footwear")
    if legs.any():
        lx = np.nonzero(legs)[1]
        if pose and min(pose["hip_l"][2], pose["hip_r"][2]) > 0.2:
            mid = (pose["hip_l"][0] + pose["hip_r"][0]) / 2
        else:
            mid = float(np.median(lx))
        for name, key in (("legwear", "Leg"), ("footwear", "Foot")):
            m = c(name)
            P[m & (xx < mid)] = pid(f"{key} R", ["Body", "Leg R"], key)
            P[m & (xx >= mid)] = pid(f"{key} L", ["Body", "Leg L"], key)
    P[c("tail")] = pid("Tail", ["Body"], "Tail")
    w = c("wings")
    P[w & (xx < body_cx)] = pid("Wing R", ["Body"], "Wing")
    P[w & (xx >= body_cx)] = pid("Wing L", ["Body"], "Wing")
    obj = c("objects")
    if obj.any():
        lb, n = ndi.label(ndi.binary_dilation(obj, iterations=2))
        sizes = np.bincount(lb.ravel())[1:]
        k = 0
        for i in np.argsort(-sizes):
            comp = (lb == i + 1) & obj
            if comp.sum() < 0.002 * M.sum():
                continue
            k += 1
            P[comp] = pid(f"Prop {k}", ["Props"], "Prop")


def _real_hair(hair, lab, fy0, fy1, fx0, fx1, yy):
    """Drop "hair" far below the face whose colour matches none of the hair on
    the head (dark suits, gold armour, ... the model sometimes calls hair)."""
    import cv2

    fh, fw = fy1 - fy0, fx1 - fx0
    head = hair & (yy < fy1)
    far = hair & (yy > fy1 + 0.5 * fh)
    if head.sum() < 50 or not far.any():
        return hair
    px = lab[head].astype(np.float32)[:: max(1, int(head.sum()) // 5000)]
    k = min(4, len(px))
    _, _, cen = cv2.kmeans(px, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 2, cv2.KMEANS_PP_CENTERS)
    q = lab[far].astype(np.float32)
    d = np.sqrt(((q[:, None, :] - cen[None]) ** 2).sum(-1)).min(1)
    odd = np.zeros_like(hair)
    odd[far] = d > 20
    # decide per connected piece, so a strand isn't cut into dots
    lb, n = ndi.label(far)
    if n:
        frac = ndi.mean(odd, lb, np.arange(1, n + 1))
        bad = np.zeros(n + 1, bool)
        bad[1:] = np.asarray(frac) > 0.5
        return hair & ~bad[lb]
    return hair


def _limbs(pose, upper, handwear, P, pid, xx, yy, W, H) -> bool:
    """Split top/handwear pixels between the torso and the arm bones."""
    def ok(k):
        q = pose[k]
        return q[2] > 0.05 and -0.5 * W <= q[0] < 1.5 * W and -0.5 * H <= q[1] < 1.5 * H

    if not (ok("shoulder_l") and ok("shoulder_r")):
        return False
    sl, sr = np.array(pose["shoulder_l"][:2]), np.array(pose["shoulder_r"][:2])
    sw = max(np.linalg.norm(sl - sr), 10)
    bones = {}
    for s in ("L", "R"):
        k = s.lower()
        for name, a, b in (("Upper Arm", "shoulder", "elbow"), ("Forearm", "elbow", "wrist"), ("Hand", "wrist", "index")):
            if ok(f"{a}_{k}") and ok(f"{b}_{k}"):
                bones[(name, s)] = (np.array(pose[f"{a}_{k}"][:2]), np.array(pose[f"{b}_{k}"][:2]))
    if not bones:
        return False
    hips = [np.array(pose[k][:2]) for k in ("hip_l", "hip_r") if ok(k)]
    torso = np.array([sr, sl] + (hips[::-1] if len(hips) == 2 else [sl + [0, sw], sr + [0, sw]]), np.float32)
    import cv2

    tm = np.zeros((H, W), np.uint8)
    cc = torso.mean(0)
    cv2.fillPoly(tm, [(cc + (torso - cc) * 1.05).astype(np.int32)], 1)
    d_torso = ndi.distance_transform_edt(~tm.astype(bool)) / (0.22 * sw)
    names, dists = [], []
    for (name, s), (A, B) in bones.items():
        r = sw * {"Upper Arm": 0.15, "Forearm": 0.13, "Hand": 0.16}[name]
        names.append((name, s))
        dists.append(_seg_dist(xx, yy, A, B) / r)
    D = np.stack(dists + [d_torso], 0)
    nearest = D.argmin(0)
    P[upper & (nearest == len(names))] = pid("Upper Body", ["Body"], "Upper Body")
    for i, (name, s) in enumerate(names):
        m = upper & (nearest == i)
        if name != "Hand":
            m &= ~handwear
        P[m] = pid(f"{name} {s}", ["Body", f"Arm {s}"], name)
    # gloves / hands always go to the closest hand
    hands = [(i, n) for i, n in enumerate(names) if n[0] == "Hand"]
    if hands:
        hd = np.stack([D[i] for i, _ in hands], 0).argmin(0)
        for j, (i, (name, s)) in enumerate(hands):
            P[handwear & (hd == j)] = pid(f"Hand {s}", ["Body", f"Arm {s}"], "Hand")
    return True
