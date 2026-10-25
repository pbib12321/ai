"""Group colour regions into named Live2D parts and decide their draw order.

The heuristics target the usual "standing character" layout (head on top,
body below, legs at the bottom) and the naming used by the official Live2D
sample models: L/R refer to the *character's* left/right, so "Eye L" is the
eye on the right-hand side of the picture.

Anything the heuristics get wrong can be fixed by editing the parts.json file
that the CLI writes next to the PSD and running again with --parts.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi
from skimage.color import rgb2lab

from .segment import Segmentation


@dataclass
class Part:
    name: str
    path: list[str]  # PSD group folders, outermost first
    regions: list[int]
    z: float  # larger = closer to the viewer
    kind: str = "fill"  # fill | eye_white | pupil | blob
    split_lines: bool = False  # put this part's interior strokes on their own layer
    sweep: bool = True  # continue this part underneath the parts covering it
    line_layer: str | None = None  # name for the outline layer (eyes)
    extra: dict = field(default_factory=dict)

    def to_json(self):
        d = {"name": self.name, "path": self.path, "regions": self.regions, "z": self.z, "kind": self.kind}
        if self.split_lines:
            d["split_lines"] = True
        if self.line_layer:
            d["line_layer"] = self.line_layer
        if not self.sweep:
            d["sweep"] = False
        return d

    @staticmethod
    def from_json(d):
        return Part(
            name=d["name"],
            path=list(d.get("path", [])),
            regions=[int(r) for r in d["regions"]],
            z=float(d["z"]),
            kind=d.get("kind", "fill"),
            split_lines=bool(d.get("split_lines", False)),
            line_layer=d.get("line_layer"),
            sweep=bool(d.get("sweep", True)),
        )


# Default depth of each part type, back (small) to front (large).
Z = {
    "Tail": 5,
    "Arm back": 10,
    "Leg": 20,
    "Neck": 30,
    "Body": 32,
    "Bottom": 40,
    "Top": 50,
    "Accessory": 55,
    "Arm front": 60,
    "Hair back": 65,
    "Ear": 70,
    "Face": 80,
    "Face details": 82,
    "Mouth": 84,
    "Eye white": 86,
    "Pupil": 87,
    "Eye line": 88,
    "Brow": 90,
    "Hair front": 95,
}


def _lab(rgb):
    return rgb2lab(np.array(rgb, np.float32).reshape(1, 1, 3) / 255.0)[0, 0]


def _de(a, b):
    return float(np.linalg.norm(_lab(a) - _lab(b)))


class _UF:
    def __init__(self, ids):
        self.p = {i: i for i in ids}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)

    def groups(self):
        out: dict[int, list[int]] = {}
        for i in self.p:
            out.setdefault(self.find(i), []).append(i)
        return list(out.values())


def _side(cx, center_x):
    # Character's right is the viewer's left.
    return "R" if cx < center_x else "L"


def auto_parts(seg: Segmentation, merge_de: float = 10.0) -> list[Part]:
    R = seg.regions
    ys, xs = np.nonzero(seg.fg)
    fy0, fy1 = ys.min(), ys.max()
    fh = fy1 - fy0

    def nbrs(r):
        return R[r].neighbors_direct | R[r].neighbors_line

    # ---- head: biggest region in the upper part of the figure
    upper = [r for r in R.values() if not r.is_blob and r.centroid[1] < fy0 + 0.6 * fh]
    face = max(upper or R.values(), key=lambda r: r.area)
    skin = face.color
    fx0, fy0f, fx1, fy1f = face.bbox
    face_cx = face.centroid[0]

    def is_skin(r):
        return _de(R[r].color, skin) < 12

    # ---- features enclosed by the face
    cand = {r for r in R if r != face.id and _inside(R[r].bbox, face.bbox)}
    changed = True
    while changed:
        changed = False
        for r in list(cand):
            if not nbrs(r) <= cand | {face.id}:
                cand.discard(r)
                changed = True
    enclosed = cand

    parts: list[Part] = []
    used: set[int] = {face.id}

    def light(r):
        L = _lab(R[r].color)
        return L[0] > 80 and np.hypot(L[1], L[2]) < 20

    eyes = [r for r in enclosed if not R[r].is_blob and light(r) and R[r].area > 0.01 * face.area]
    eyes.sort(key=lambda r: R[r].centroid[0])
    eyes = eyes[:2] if len(eyes) >= 2 else eyes
    eye_bottom = max((R[e].bbox[3] for e in eyes), default=face.centroid[1])
    eye_top = min((R[e].bbox[1] for e in eyes), default=face.centroid[1])
    for e in eyes:
        side = _side(R[e].centroid[0], face_cx) if len(eyes) == 1 else ("R" if e == eyes[0] else "L")
        pupils = [
            p
            for p in enclosed
            if p not in used and p != e and (R[p].is_blob or _lab(R[p].color)[0] < 50) and _inside(R[p].bbox, R[e].bbox, 2)
        ]
        # also colour rings (iris) touching the pupil, inside the eye white
        iris = [
            p
            for p in enclosed
            if p not in used and p not in pupils and p != e and p not in eyes and _inside(R[p].bbox, R[e].bbox, 2)
        ]
        path = ["Head", f"Eye {side}"]
        parts.append(Part(f"Eye White {side}", path, [e], Z["Eye white"], kind="eye_white", line_layer=f"Eye Line {side}", sweep=False))
        if pupils or iris:
            parts.append(Part(f"Pupil {side}", path, sorted(pupils + iris), Z["Pupil"], kind="pupil", sweep=False))
        used |= {e, *pupils, *iris}

    brows = [b for b in enclosed if b not in used and R[b].is_blob and R[b].centroid[1] < eye_top + 0.25 * (eye_bottom - eye_top)]
    for b in sorted(brows, key=lambda r: R[r].centroid[0]):
        side = _side(R[b].centroid[0], face_cx)
        name = f"Brow {side}"
        if any(p.name == name for p in parts):
            next(p for p in parts if p.name == name).regions.append(b)
        else:
            parts.append(Part(name, ["Head"], [b], Z["Brow"], kind="blob", sweep=False))
        used.add(b)

    rest = [r for r in enclosed if r not in used]
    if rest:
        # cluster the remaining features (mouth, nose, cheeks, ...)
        uf = _UF(rest)
        for r in rest:
            for n in nbrs(r):
                if n in uf.p:
                    uf.union(r, n)
        clusters = sorted(uf.groups(), key=lambda g: -sum(R[r].area for r in g))
        for i, g in enumerate(clusters):
            cy = np.mean([R[r].centroid[1] for r in g])
            name = "Mouth" if i == 0 and cy > eye_top else f"Face Feature {i + 1}"
            parts.append(Part(name, ["Head"], sorted(g), Z["Mouth"], sweep=False))
            used |= set(g)

    parts.append(Part("Face", ["Head"], [face.id], Z["Face"], split_lines=True))

    # ---- things attached to the head: ears / hair / neck
    attached = sorted(face.neighbors_line | face.neighbors_direct)
    ears = sorted(
        (r for r in attached if r not in used and R[r].centroid[1] < face.centroid[1] and is_skin(r)),
        key=lambda r: R[r].centroid[0],
    )
    for i, r in enumerate(ears):
        if len(ears) == 2:
            side = "R" if i == 0 else "L"
        else:
            side = _side(R[r].centroid[0], face_cx)
        name = f"Ear {side}"
        if any(p.name == name for p in parts):
            name += f" {r}"
        parts.append(Part(name, ["Head"], [r], Z["Ear"]))
        used.add(r)
    for r in attached:
        if r in used:
            continue
        reg = R[r]
        if reg.centroid[1] < face.centroid[1]:
            parts.append(Part(f"Hair {r}", ["Head"], [r], Z["Hair back"]))
            used.add(r)
        elif is_skin(r) and reg.bbox[1] >= face.centroid[1] and reg.area < 0.2 * face.area:
            parts.append(Part("Neck", ["Body"], [r], Z["Neck"]))
            used.add(r)

    # ---- body
    body = [r for r in R if r not in used]
    # shading regions touch their base colour directly (no ink between) -> merge
    uf = _UF(body)
    for r in body:
        for n in R[r].neighbors_direct:
            if n not in uf.p or R[r].is_blob or R[n].is_blob:
                continue
            # different colours touching = shading; same colour touching = a
            # piece cut off at a narrow neck, which stays with skin (fingers,
            # toes) but is kept apart for clothing (sleeves)
            if _de(R[r].color, R[n].color) > merge_de or (is_skin(r) and is_skin(n)):
                uf.union(r, n)
    clusters = uf.groups()
    # the representative of a cluster is its biggest region
    rep = {}
    for g in clusters:
        big = max(g, key=lambda r: R[r].area)
        rep[big] = g

    skin_c = [b for b in rep if is_skin(b)]
    cloth_c = [b for b in rep if not is_skin(b)]
    neck_ids = [p for p in parts if p.name == "Neck"]
    body_top = R[neck_ids[0].regions[0]].bbox[3] if neck_ids else face.bbox[3]
    cloth_px = [R[r] for b in cloth_c for r in rep[b]]
    if cloth_px:
        cx0 = min(r.bbox[0] for r in cloth_px)
        cx1 = max(r.bbox[2] for r in cloth_px)
    else:
        cx0, cx1 = fx0, fx1
    center_x = R[neck_ids[0].regions[0]].centroid[0] if neck_ids else face_cx

    # legs: skin at the bottom of the figure
    bottom_line = fy0 + 0.72 * fh
    leg_c = [b for b in skin_c if R[b].centroid[1] > bottom_line]
    leg_c.sort(key=lambda b: -sum(R[r].area for r in rep[b]))
    seeds = [b for b in leg_c[:2] if sum(R[r].area for r in rep[b]) > 0.3 * sum(R[r].area for r in rep[leg_c[0]])] if leg_c else []
    legs: dict[int, list[int]] = {s: list(rep[s]) for s in seeds}
    for b in leg_c:
        if b in seeds:
            continue
        touching = [s for s in seeds if any(n in legs[s] for r in rep[b] for n in nbrs(r))]
        if touching:
            legs[min(touching, key=lambda s: np.hypot(R[s].centroid[0] - R[b].centroid[0], R[s].centroid[1] - R[b].centroid[1]))] += rep[b]
    leg_c = [b for b in leg_c if any(set(rep[b]) <= set(g) for g in legs.values())]
    for s, g in legs.items():
        side = _side(R[s].centroid[0], center_x) if len(legs) == 2 else _side(R[s].centroid[0], center_x)
        parts.append(Part(f"Leg {side}", ["Body", f"Leg {side}"], sorted(g), Z["Leg"]))
        used |= set(g)

    # arms: skin hanging from a piece of clothing above them; tail: the rest
    other_skin = [b for b in skin_c if b not in leg_c]
    arms: list[int] = []
    for b in sorted(other_skin, key=lambda b: -sum(R[r].area for r in rep[b])):
        reg = R[b]
        top = reg.bbox[1] + 0.25 * (reg.bbox[3] - reg.bbox[1])
        hang = any(
            not is_skin(n) and R[n].centroid[1] < top for r in rep[b] for n in nbrs(r)
        )
        inside_body = cx0 <= reg.centroid[0] <= cx1
        gx0 = min(R[r].bbox[0] for r in rep[b])
        gy0 = min(R[r].bbox[1] for r in rep[b])
        gx1 = max(R[r].bbox[2] for r in rep[b])
        gy1 = max(R[r].bbox[3] for r in rep[b])
        limb = (gy1 - gy0) > 1.8 * (gx1 - gx0) and gy0 < body_top + 0.35 * (fy1 - body_top)
        if (hang or limb) and len(arms) < 2:
            arms.append(b)
        elif not inside_body and sum(R[r].area for r in rep[b]) > 0.002 * seg.fg.sum():
            parts.append(Part("Tail" if not any(p.name == "Tail" for p in parts) else f"Tail {b}", ["Body"], sorted(rep[b]), Z["Tail"]))
            used |= set(rep[b])
    arm_parts: dict[int, Part] = {}
    for b in arms:
        side = _side(R[b].centroid[0], center_x)
        p = Part(f"Arm {side}", ["Body", f"Arm {side}"], sorted(rep[b]), Z["Arm front"])
        parts.append(p)
        arm_parts[b] = p
        used |= set(rep[b])
    # small skin bits (fingers) go to the nearest arm/leg, others become "Body"
    for b in other_skin:
        if set(rep[b]) & used:
            continue
        owners = [p for p in parts if p.name.startswith(("Arm", "Leg"))]
        near = [p for p in owners if any(n in p.regions for r in rep[b] for n in nbrs(r))]
        if near and sum(R[r].area for r in rep[b]) < 0.1 * sum(R[r].area for r in near[0].regions):
            near[0].regions = sorted(near[0].regions + rep[b])
        else:
            existing = next((p for p in parts if p.name == "Body"), None)
            if existing:
                existing.regions = sorted(existing.regions + rep[b])
            else:
                parts.append(Part("Body", ["Body"], sorted(rep[b]), Z["Body"]))
        used |= set(rep[b])

    # clothing: garments are same-coloured clusters joined by ink
    cloth_c = [b for b in cloth_c if b not in used]
    uf = _UF(cloth_c)
    gap = 0.1 * fh
    for a in cloth_c:
        for b in cloth_c:
            if a < b and _de(R[a].color, R[b].color) < merge_de and (
                any(n in rep[b] for r in rep[a] for n in nbrs(r)) or _bbox_gap(R, rep[a], rep[b]) < gap
            ):
                uf.union(a, b)
    garments = [sorted(r for b in g for r in rep[b]) for g in uf.groups()]
    garments.sort(key=lambda g: -sum(R[r].area for r in g))
    body_area = sum(R[r].area for g in garments for r in g) or 1
    accessories, real = [], []
    for g in garments:
        area = sum(R[r].area for r in g)
        outside = {n for r in g for n in nbrs(r)} - set(g)
        others = [x for x in garments if x is not g and set(x) & outside]
        if area < 0.06 * body_area and len(others) == 1 and outside <= set(others[0]) | used:
            accessories.append(g)
        else:
            real.append(g)

    leg_tops = [min(R[r].bbox[1] for r in p.regions) for p in parts if p.name.startswith("Leg")]
    waist = min(leg_tops) if leg_tops else fy0 + 0.6 * fh
    tops, bottoms = [], []
    for g in real:
        cy = sum(R[r].centroid[1] * R[r].area for r in g) / sum(R[r].area for r in g)
        y0 = min(R[r].bbox[1] for r in g)
        (bottoms if cy > body_top + 0.55 * (waist - body_top) and y0 > body_top + 0.2 * (waist - body_top) else tops).append(g)

    # sleeves: garment pieces the arms come out of
    for b, ap in arm_parts.items():
        arm_regs = set(ap.regions)
        top_y = min(R[r].bbox[1] for r in arm_regs)
        for g in tops:
            cuff = [
                r
                for r in g
                if (R[r].neighbors_line | R[r].neighbors_direct) & arm_regs
                and R[r].centroid[1] < top_y + 0.15 * (max(R[a].bbox[3] for a in arm_regs) - top_y)
                and R[r].area < 0.5 * max(R[x].area for x in g)
            ]
            if not cuff:
                continue
            sleeve = set(cuff)
            dist = lambda r: abs(R[r].centroid[0] - center_x)
            frontier = list(cuff)
            biggest = max(g, key=lambda r: R[r].area)
            while frontier:
                c = frontier.pop()
                for n in nbrs(c):
                    if n in g and n not in sleeve and n != biggest and dist(n) >= dist(c) - 0.25 * (R[c].bbox[2] - R[c].bbox[0]):
                        sleeve.add(n)
                        frontier.append(n)
            g[:] = [r for r in g if r not in sleeve]
            parts.append(Part(f"Sleeve {ap.name[-1]}", ap.path, sorted(sleeve), ap.z + 1))
            used |= sleeve

    for i, g in enumerate(tops):
        if g:
            parts.append(Part("Top" if i == 0 else f"Top {i + 1}", ["Body"], g, Z["Top"] - i * 0.1))
            used |= set(g)
    for i, g in enumerate(bottoms):
        parts.append(Part("Bottom" if i == 0 else f"Bottom {i + 1}", ["Body"], g, Z["Bottom"] - i * 0.1))
        used |= set(g)
    for i, g in enumerate(accessories):
        parts.append(Part(f"Accessory {i + 1}", ["Body"], g, Z["Accessory"] + i * 0.1))
        used |= set(g)

    left = [r for r in R if r not in used]
    for r in left:
        parts.append(Part(f"Piece {r}", ["Body"], [r], Z["Body"]))

    seen: dict[str, int] = {}
    for p in sorted(parts, key=lambda p: -sum(seg.regions[r].area for r in p.regions)):
        if p.name in seen:
            seen[p.name] += 1
            p.name = f"{p.name} {seen[p.name]}"
        else:
            seen[p.name] = 1

    _decide_arm_depth(seg, parts)
    _order_pairs(seg, parts, ("Leg R", "Leg L"))
    return parts


def _bbox_gap(R, ga, gb):
    a = [min(R[r].bbox[0] for r in ga), min(R[r].bbox[1] for r in ga), max(R[r].bbox[2] for r in ga), max(R[r].bbox[3] for r in ga)]
    b = [min(R[r].bbox[0] for r in gb), min(R[r].bbox[1] for r in gb), max(R[r].bbox[2] for r in gb), max(R[r].bbox[3] for r in gb)]
    dx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    return float(np.hypot(dx, dy))


def _inside(inner, outer, pad=0):
    return inner[0] >= outer[0] - pad and inner[1] >= outer[1] - pad and inner[2] <= outer[2] + pad and inner[3] <= outer[3] + pad


# --------------------------------------------------------------------------- depth cues


def occlusion_score(seg: Segmentation, a: list[int], b: list[int]) -> float:
    """> 0 when region set ``a`` looks like it is in front of ``b``.

    Uses T-junctions: where two outlines meet, the outline of the front object
    keeps going while the back object's outline stops at a corner, so the
    front object fills about half of a small disc around the junction and the
    back object only about a quarter.
    """
    ma = np.isin(seg.labels, a)
    mb = np.isin(seg.labels, b)
    hw = seg.line_halfwidth
    r = int(np.ceil(2 * hw + 2))
    da = ndi.binary_dilation(ma, iterations=r)
    db = ndi.binary_dilation(mb, iterations=r)
    shared = seg.line & da & db
    lab, n = ndi.label(shared, structure=np.ones((3, 3)))
    if n == 0:
        return 0.0
    R = int(6 * hw + 4)
    yy, xx = np.mgrid[-R : R + 1, -R : R + 1]
    disk = xx**2 + yy**2 <= R * R
    score = 0.0
    h, w = seg.labels.shape
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(ys) < 3 * hw:
            continue
        pts = np.stack([xs, ys], 1).astype(np.float32)
        c = pts.mean(0)
        u, s, vt = np.linalg.svd(pts - c, full_matrices=False)
        proj = (pts - c) @ vt[0]
        for end in (pts[proj.argmin()], pts[proj.argmax()]):
            x, y = int(end[0]), int(end[1])
            x0, x1, y0, y1 = max(0, x - R), min(w, x + R + 1), max(0, y - R), min(h, y + R + 1)
            dk = disk[y0 - (y - R) : y1 - (y - R), x0 - (x - R) : x1 - (x - R)]
            fa = (ma[y0:y1, x0:x1] & dk).sum()
            fb = (mb[y0:y1, x0:x1] & dk).sum()
            tot = dk.sum()
            score += (fa - fb) / tot
    return score


def _decide_arm_depth(seg, parts):
    clothes = [p for p in parts if p.name.startswith(("Top", "Bottom"))]
    if not clothes:
        return
    cloth_regs = [r for p in clothes for r in p.regions]
    for arm in [p for p in parts if p.name.startswith("Arm ")]:
        s = occlusion_score(seg, arm.regions, cloth_regs)
        if s < 0:
            for p in parts:
                if p.path[:2] == arm.path[:2]:
                    p.z = p.z - Z["Arm front"] + Z["Arm back"]


def _order_pairs(seg, parts, names):
    ps = [next((p for p in parts if p.name == n), None) for n in names]
    if None in ps:
        return
    a, b = ps
    s = occlusion_score(seg, a.regions, b.regions)
    if s > 0:
        a.z, b.z = max(a.z, b.z) + 0.5, min(a.z, b.z)
    elif s < 0:
        b.z, a.z = max(a.z, b.z) + 0.5, min(a.z, b.z)


def save_parts(parts: list[Part], path: str):
    with open(path, "w") as f:
        json.dump({"parts": [p.to_json() for p in sorted(parts, key=lambda p: -p.z)]}, f, indent=1)


def load_parts(path: str) -> list[Part]:
    with open(path) as f:
        return [Part.from_json(d) for d in json.load(f)["parts"]]
