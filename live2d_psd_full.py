#!/usr/bin/env python3
"""live2d_psd - single-file version.

Turns a flat, outlined 2D character picture into a layered PSD ready for
Live2D Cubism Editor (also opens in Photopea / Photoshop).

    pip install numpy scipy opencv-python-headless scikit-image Pillow
    python live2d_psd_full.py character.png              # -> character_cubism.psd
    python live2d_psd_full.py character.png --scale 2
    python live2d_psd_full.py character.png --parts character_cubism.parts.json
"""
from __future__ import annotations

import argparse
import json
import os
import struct
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage.color import rgb2lab
from skimage.morphology import skeletonize

# ======================================================================
# segment
# ======================================================================
@dataclass
class Region:
    id: int
    color: tuple[int, int, int]
    area: int
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 (exclusive)
    centroid: tuple[float, float]
    is_blob: bool = False  # solid dark shape (pupil, eyebrow) rather than a fill
    neighbors_direct: set[int] = field(default_factory=set)  # touching, no line between
    neighbors_line: set[int] = field(default_factory=set)  # separated by a line


@dataclass
class Segmentation:
    rgba: np.ndarray  # H x W x 4 uint8 (background removed)
    fg: np.ndarray  # bool
    line: np.ndarray  # bool, thin outline strokes
    line_color: tuple[int, int, int]
    line_halfwidth: float
    labels: np.ndarray  # int32, 0 = background or line, >0 region id
    regions: dict[int, Region]


# --------------------------------------------------------------------------- input


def load_rgba(path: str, bg_tolerance: float = 40.0) -> np.ndarray:
    """Load an image and make its background transparent if it has none."""
    im = Image.open(path)
    im = im.convert("RGBA")
    rgba = np.array(im)
    a = rgba[..., 3]
    if (a < 128).mean() > 0.01:
        return rgba
    # Opaque image: flood fill the background from the border.
    rgb = rgba[..., :3].astype(np.int16)
    h, w = a.shape
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    vals, counts = np.unique(border, axis=0, return_counts=True)
    bg = vals[counts.argmax()]
    close = np.abs(rgb - bg).max(-1) <= bg_tolerance
    lab, _ = ndi.label(close)
    edge_labels = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    edge_labels = edge_labels[edge_labels > 0]
    bgmask = np.isin(lab, edge_labels)
    # The anti-aliased rim between the background and the outline is a mix of
    # the two; pixels that are more background than ink belong to the background.
    lum = rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    bg_lum = float(np.dot(bg, [0.299, 0.587, 0.114]))
    ink = ndi.minimum_filter(lum, size=9)
    rim = ndi.binary_dilation(bgmask, iterations=3) & ~bgmask
    bgmask |= rim & (np.abs(lum - bg_lum) < 0.5 * np.abs(ink - bg_lum))
    rgba = rgba.copy()
    rgba[..., 3] = np.where(bgmask, 0, 255).astype(np.uint8)
    return rgba


# --------------------------------------------------------------------------- lines


def _line_masks(rgba: np.ndarray, fg: np.ndarray):
    rgb = rgba[..., :3].astype(np.float32)
    lum = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    # A pixel is "ink" when it is much darker than the brightest colour around it.
    local_max = ndi.maximum_filter(np.where(fg, lum, 0), size=15)
    dark = fg & (((lum < 0.5 * local_max) & (lum < 110)) | (lum < 45))
    dark = ndi.binary_opening(dark, iterations=1) | (dark & (lum < 30))

    dt = cv2.distanceTransform(dark.astype(np.uint8), cv2.DIST_L2, 5)
    ske = skeletonize(dark)
    hw = float(np.median(dt[ske])) if ske.any() else 2.0
    hw = max(hw, 1.0)

    # Solid dark shapes are much thicker than a stroke: pupils, brows, hair shadows.
    r = int(np.ceil(hw * 1.9))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    core = cv2.erode(dark.astype(np.uint8), k)
    n, lab, st, _ = cv2.connectedComponentsWithStats(core)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= max(12, (hw * 1.5) ** 2)
    core = keep[lab].astype(np.uint8)
    blob = (cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 3, 2 * r + 3))) > 0) & dark
    line = dark & ~blob

    ink = rgba[..., :3][line & (lum < 60)]
    line_color = tuple(int(v) for v in np.median(ink, 0)) if len(ink) else (0, 0, 0)
    return line, blob, hw, line_color


# --------------------------------------------------------------------------- colours


def _palette(lab_px: np.ndarray, min_count: int, merge_de: float) -> np.ndarray:
    q = np.round(lab_px / 4.0).astype(np.int32)
    keys, inv, counts = np.unique(q, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(-counts)
    pal: list[np.ndarray] = []
    for i in order:
        if counts[i] < min_count:
            break
        c = lab_px[inv.ravel() == i].mean(0)
        if pal and np.min(np.linalg.norm(np.array(pal) - c, axis=1)) < merge_de:
            continue
        pal.append(c)
    return np.array(pal) if pal else lab_px.mean(0, keepdims=True)


def _fill_nearest(labels: np.ndarray, unknown: np.ndarray) -> np.ndarray:
    if not unknown.any():
        return labels
    _, idx = ndi.distance_transform_edt(unknown | (labels == 0), return_indices=True)
    filled = labels[idx[0], idx[1]]
    return np.where(unknown, filled, labels)


# --------------------------------------------------------------------------- splitting


def _split_constrictions(mask: np.ndarray, hw: float, min_frac: float = 0.01) -> list[tuple]:
    """Find straight cuts across narrow necks of a region.

    Returns a list of ((x0, y0), (x1, y1)) segments.  A cut is accepted when it
    joins two sharp concave corners of the outline, is short compared with the
    contour distance between those corners, and leaves two substantial pieces.
    """
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return []
    c = max(cs, key=cv2.contourArea)[:, 0, :].astype(np.float32)
    n = len(c)
    if n < 60:
        return []
    area = mask.sum()
    step = max(3, int(hw * 3))
    prev = np.roll(c, step, 0)
    nxt = np.roll(c, -step, 0)
    v1 = prev - c
    v2 = nxt - c
    ang = np.degrees(
        np.arccos(np.clip((v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-6), -1, 1))
    )
    # concave = the bisector points outside the shape
    bis = (v1 / (np.linalg.norm(v1, axis=1, keepdims=True) + 1e-6) + v2 / (np.linalg.norm(v2, axis=1, keepdims=True) + 1e-6))
    probe = np.round(c + bis * 3).astype(int)
    h, w = mask.shape
    probe[:, 0] = probe[:, 0].clip(0, w - 1)
    probe[:, 1] = probe[:, 1].clip(0, h - 1)
    concave = ~mask[probe[:, 1], probe[:, 0]] & (ang < 135)
    idx = np.where(concave)[0]
    if len(idx) < 2:
        return []
    # non-maximum suppression: keep sharpest point per run
    corners = []
    run = [idx[0]]
    for i in idx[1:]:
        if i - run[-1] <= step:
            run.append(i)
        else:
            corners.append(min(run, key=lambda j: ang[j]))
            run = [i]
    corners.append(min(run, key=lambda j: ang[j]))
    cuts = []
    for a_i in range(len(corners)):
        for b_i in range(a_i + 1, len(corners)):
            i, j = corners[a_i], corners[b_i]
            d = float(np.linalg.norm(c[i] - c[j]))
            cd = min(abs(i - j), n - abs(i - j))
            if d < 4 * hw or cd < 1.5 * d:
                continue
            # the cut must stay inside the shape
            t = np.linspace(0, 1, max(8, int(d)))
            pts = np.round(c[i][None] * (1 - t[:, None]) + c[j][None] * t[:, None]).astype(int)
            inner = mask[pts[1:-1, 1], pts[1:-1, 0]]
            if inner.mean() < 0.97:
                continue
            test = mask.copy().astype(np.uint8)
            cv2.line(test, tuple(int(v) for v in c[i]), tuple(int(v) for v in c[j]), 0, 3)
            nl, lab = cv2.connectedComponents(test, connectivity=4)
            if nl < 3:
                continue
            sizes = np.bincount(lab.ravel())[1:]
            sizes = np.sort(sizes)[::-1]
            if sizes[1] < min_frac * area:
                continue
            # narrowness: the cut must be short compared with the main piece
            big = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
            ys, xs = np.nonzero(big)
            if d > 0.25 * max(np.ptp(xs), np.ptp(ys)):
                continue
            cuts.append((d, (tuple(c[i]), tuple(c[j]))))
    cuts.sort()
    chosen = []
    for d, seg in cuts:
        if all(np.linalg.norm(np.array(seg[0]) - np.array(s[0])) > 6 * hw for s in chosen):
            chosen.append(seg)
    return chosen


# --------------------------------------------------------------------------- main


def segment(
    rgba: np.ndarray,
    *,
    merge_de: float = 10.0,
    min_region: int | None = None,
    split_necks: bool = True,
) -> Segmentation:
    h, w = rgba.shape[:2]
    fg = rgba[..., 3] >= 128
    line, blob, hw, line_color = _line_masks(rgba, fg)

    if min_region is None:
        min_region = max(25, int(fg.sum() * 2e-5))

    lab_img = rgb2lab(rgba[..., :3].astype(np.float32) / 255.0)
    fill = fg & ~line & ~blob
    # Pixels hugging the ink or the silhouette are anti-aliasing mixtures.
    near_ink = ndi.binary_dilation(line | blob, iterations=max(1, int(round(hw))))
    soft_edge = (rgba[..., 3] < 250) & fg
    core = fill & ~near_ink & ~soft_edge

    pal = _palette(lab_img[core], min_count=min_region, merge_de=merge_de)
    d = np.linalg.norm(lab_img[core][:, None, :] - pal[None], axis=2)
    cls = np.zeros((h, w), np.int32)
    cls[core] = d.argmin(1) + 1
    cls = _fill_nearest(cls, fill & (cls == 0))
    cls[~fill] = 0

    # Connected components per colour, blobs as their own components.
    labels = np.zeros((h, w), np.int32)
    nid = 0
    for k in range(1, len(pal) + 1):
        m = cls == k
        if not m.any():
            continue
        lab, n = ndi.label(m)
        labels[m] = lab[m] + nid
        nid += n
    blab, nb = ndi.label(blob)
    labels[blob] = blab[blob] + nid
    blob_ids = set(range(nid + 1, nid + nb + 1))
    nid += nb

    # Absorb specks into whichever neighbour surrounds them.
    sizes = np.bincount(labels.ravel(), minlength=nid + 1)
    tiny = (sizes < min_region)
    tiny[0] = False
    tmask = tiny[labels]
    if tmask.any():
        labels[tmask] = 0
        labels = _fill_nearest(labels, tmask)
        labels[~(fill | blob)] = 0

    # Cut regions across narrow necks (e.g. a head flowing into a neck with no line).
    if split_necks:
        for rid in np.unique(labels):
            if rid == 0 or rid in blob_ids:
                continue
            m = labels == rid
            if m.sum() < 50 * min_region:
                continue
            for p0, p1 in _split_constrictions(m, hw):
                cut = np.zeros((h, w), np.uint8)
                cv2.line(cut, tuple(int(round(v)) for v in p0), tuple(int(round(v)) for v in p1), 1, 2)
                m2 = m & ~cut.astype(bool)
                lab, n = ndi.label(m2)
                if n < 2:
                    continue
                biggest = np.bincount(lab.ravel())[1:].argmax() + 1
                for comp in range(1, n + 1):
                    if comp == biggest:
                        continue
                    sel = lab == comp
                    if sel.sum() < min_region:
                        continue
                    nid += 1
                    labels[sel] = nid
                # cut pixels go to the nearest piece
                cutpx = m & cut.astype(bool)
                labels[cutpx] = 0
                labels = _fill_nearest(labels, cutpx)
                m = labels == rid

    labels = _merge_pockets(labels, lab_img, hw, min_region, merge_de, blob_ids)

    # Re-number compactly.
    ids = np.unique(labels)
    ids = ids[ids > 0]
    remap = np.zeros(labels.max() + 1, np.int32)
    remap[ids] = np.arange(1, len(ids) + 1)
    new_blob = {int(remap[b]) for b in blob_ids if b < len(remap) and remap[b] > 0}
    labels = remap[labels]

    regions: dict[int, Region] = {}
    objs = ndi.find_objects(labels)
    for rid, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        m = labels[sl] == rid
        ys, xs = np.nonzero(m)
        px = rgba[sl][..., :3][m]
        col = tuple(int(v) for v in np.median(px, 0))
        regions[rid] = Region(
            id=rid,
            color=col,
            area=int(m.sum()),
            bbox=(sl[1].start, sl[0].start, sl[1].stop, sl[0].stop),
            centroid=(float(xs.mean() + sl[1].start), float(ys.mean() + sl[0].start)),
            is_blob=rid in new_blob,
        )
    for rid in new_blob:
        if rid in regions:
            regions[rid].color = line_color

    _adjacency(labels, line, hw, regions)
    return Segmentation(rgba, fg, line, line_color, hw, labels, regions)


def _merge_pockets(labels, lab_img, hw, min_region, merge_de, blob_ids):
    """Give small pockets the label of a bigger same-coloured region across the ink.

    E.g. bits of face colour trapped between eyelashes, or toes cut off by
    claw lines: they are drawn separately but are the same surface.
    """
    limit = max(8 * min_region, int((5 * hw) ** 2))
    sizes = np.bincount(labels.ravel())
    r = int(np.ceil(2 * hw + 2))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    objs = ndi.find_objects(labels)
    color = {}
    for rid in np.argsort(sizes):
        if rid == 0 or sizes[rid] == 0 or sizes[rid] >= limit or rid > len(objs) or objs[rid - 1] is None:
            continue
        sl = objs[rid - 1]
        y0, y1 = max(0, sl[0].start - r), min(labels.shape[0], sl[0].stop + r)
        x0, x1 = max(0, sl[1].start - r), min(labels.shape[1], sl[1].stop + r)
        win = labels[y0:y1, x0:x1]
        m = win == rid
        if not m.any():
            continue
        ring = cv2.dilate(m.astype(np.uint8), k).astype(bool) & ~m
        cands = [c for c in np.unique(win[ring]) if c not in (0, rid) and sizes[c] > sizes[rid]]
        if not cands:
            continue
        mine = lab_img[y0:y1, x0:x1][m].mean(0)
        best = None
        for c in sorted(cands, key=lambda c: -sizes[c]):
            if c not in color:
                color[c] = lab_img[labels == c].mean(0)
            if np.linalg.norm(color[c] - mine) < merge_de * 1.5 and (rid in blob_ids) == (c in blob_ids):
                best = c
                break
        if best is None:
            continue
        win[m] = best
        sizes[best] += sizes[rid]
        sizes[rid] = 0
    return labels


def _adjacency(labels: np.ndarray, line: np.ndarray, hw: float, regions: dict[int, Region]) -> None:
    # Direct contact: horizontally / vertically neighbouring pixels with different labels.
    for a, b in ((labels[:, :-1], labels[:, 1:]), (labels[:-1], labels[1:])):
        m = (a != b) & (a > 0) & (b > 0)
        pairs = np.unique(np.stack([a[m], b[m]], 1), axis=0)
        for x, y in pairs:
            regions[int(x)].neighbors_direct.add(int(y))
            regions[int(y)].neighbors_direct.add(int(x))
    # Across a line: both regions reach the same ink pixel within ~one stroke width.
    r = int(np.ceil(hw * 2 + 2))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    touch: dict[int, np.ndarray] = {}
    ys, xs = np.nonzero(line)
    for rid, reg in regions.items():
        x0, y0, x1, y1 = reg.bbox
        X0, Y0 = max(0, x0 - r), max(0, y0 - r)
        X1, Y1 = min(labels.shape[1], x1 + r), min(labels.shape[0], y1 + r)
        m = (labels[Y0:Y1, X0:X1] == rid).astype(np.uint8)
        dm = cv2.dilate(m, k).astype(bool)
        sel = (ys >= Y0) & (ys < Y1) & (xs >= X0) & (xs < X1)
        hit = np.zeros(len(ys), bool)
        hit[sel] = dm[ys[sel] - Y0, xs[sel] - X0]
        touch[rid] = hit
    ids = list(regions)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if (touch[a] & touch[b]).sum() >= max(3, hw * 2):
                if b not in regions[a].neighbors_direct:
                    regions[a].neighbors_line.add(b)
                    regions[b].neighbors_line.add(a)


# ======================================================================
# parts
# ======================================================================
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


# ======================================================================
# complete
# ======================================================================
LINE = -1  # index-image value for outline pixels


@dataclass
class PieceLayer:
    name: str
    path: list[str]
    z: float
    # index image over the whole canvas: 0 = empty, >0 colour id, LINE = outline
    index: np.ndarray
    colors: dict[int, tuple[int, int, int]]


def _kind(name: str) -> str:
    # "Leg L" and "Leg R" are the same kind of part: they sit side by side
    # rather than one emerging from under the other.
    for suffix in (" L", " R"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def build_layers(seg: Segmentation, parts: list[Part], sweep: float = 1.0) -> list[PieceLayer]:
    h, w = seg.labels.shape
    hw = seg.line_halfwidth
    parts = sorted(parts, key=lambda p: -p.z)  # front first
    n = len(parts)
    fills = [np.isin(seg.labels, p.regions) for p in parts]

    # ---------------------------------------------------------------- outlines
    r = int(np.ceil(2 * hw + 1))
    k = _disk(r)
    near = [cv2.dilate(f.astype(np.uint8), k).astype(bool) & seg.line for f in fills]
    owner = np.full((h, w), -1, np.int32)
    ncand = np.zeros((h, w), np.int32)
    for i in range(n):
        ncand += near[i]
        owner[(owner < 0) & near[i]] = i
    lost = seg.line & (owner < 0)
    if lost.any():
        any_fill = np.zeros((h, w), np.int32) - 1
        for i in range(n):
            any_fill[fills[i]] = i
        _, idx = ndi.distance_transform_edt(any_fill < 0, return_indices=True)
        owner[lost] = any_fill[idx[0][lost], idx[1][lost]]

    # Interior strokes (only one part nearby, not on the silhouette).
    near_bg = cv2.dilate((~seg.fg).astype(np.uint8), k).astype(bool)
    interior = seg.line & (ncand == 1) & ~near_bg
    boundary = seg.line & ~interior
    detail = np.zeros((h, w), bool)  # floating strokes -> "details" layer
    lab, nl = ndi.label(interior, structure=np.ones((3, 3)))
    if nl:
        ring = cv2.dilate(boundary.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        for i, sl in enumerate(ndi.find_objects(lab), start=1):
            comp = lab[sl] == i
            o = owner[sl][comp][0]
            touch = set(int(t) for t in np.unique(owner[sl][ring[sl] & boundary[sl] & ndi.binary_dilation(comp, iterations=2)]))
            touch.discard(-1)
            if not touch:
                if parts[o].split_lines:
                    detail[sl][comp] = True
                continue
            # lashes: strokes on the face growing out of an eye's outline
            eyes = [t for t in touch if parts[t].line_layer and parts[t].z > parts[o].z]
            if eyes and parts[o].split_lines:
                owner[sl][comp] = eyes[0]

    # ---------------------------------------------------------------- per part
    color_ids: dict[tuple[int, int, int], int] = {}

    def cid(c):
        if c not in color_ids:
            color_ids[c] = len(color_ids) + 1
        return color_ids[c]

    region_cid = np.zeros(seg.labels.max() + 1, np.int32)
    for rid, reg in seg.regions.items():
        region_cid[rid] = cid(reg.color)
    line_cid = cid(seg.line_color)

    own_line = [(owner == i) & ~detail for i in range(n)]
    # Side-by-side pieces of the same kind (Leg L / Leg R) don't hide each
    # other, so both keep the stroke between them.
    for i in range(n):
        for j in range(i + 1, n):
            if _kind(parts[i].name) == _kind(parts[j].name) and parts[i].name != parts[j].name:
                own_line[j] |= near[i] & near[j] & (owner == i)
    visible = [fills[i] | own_line[i] | (detail & (owner == i)) for i in range(n)]

    layers: list[PieceLayer] = []
    front = np.zeros((h, w), bool)
    for i, p in enumerate(parts):
        fill = fills[i]
        line = own_line[i]
        own = fill | line | (detail & (owner == i))
        idx = np.zeros((h, w), np.int32)
        idx[fill] = region_cid[seg.labels[fill]]
        idx[line] = LINE

        hidden_ok = front & seg.fg & ~own

        # holes enclosed by this part
        holes = ndi.binary_fill_holes(own) & ~own & hidden_ok
        complete = own | holes
        # sweep the cross-section under the parts in front of it
        if p.sweep and hidden_ok.any():
            covers = [visible[j] for j in range(i) if _kind(parts[j].name) != _kind(p.name)]
            ext_idx = _sweep(idx, own, covers, hidden_ok, hw, sweep)
            if ext_idx is not None:
                take = (ext_idx != 0) & ~complete
                idx[take] = ext_idx[take]
                complete |= take

        # fill colour everywhere (also under outlines) from the nearest fill pixel
        fill_now = idx > 0
        if fill_now.any():
            _, nidx = ndi.distance_transform_edt(~fill_now, return_indices=True)
            base = idx[nidx[0], nidx[1]]
        else:
            base = np.full((h, w), line_cid, np.int32)
        lines = complete & (idx == LINE)

        if p.line_layer:
            # the white of the eye stops at its ring; lashes live on the line layer only
            ring = lines & cv2.dilate(fill.astype(np.uint8), _disk(int(np.ceil(2 * hw + 1)))).astype(bool)
            body = ndi.binary_fill_holes(fill | ring | holes) & complete
            layers.append(PieceLayer(p.name, p.path, p.z, np.where(body, base, 0), {}))
            # above the pupil, so the iris slides under the eye outline
            layers.append(PieceLayer(p.line_layer, p.path, p.z + 1.5, np.where(lines, LINE, 0), {}))
        else:
            out = np.where(complete, base, 0)
            out[lines] = LINE
            dmask = detail & (owner == i)
            if p.split_lines and dmask.any():
                out[dmask] = base[dmask]
                layers.append(PieceLayer(f"{p.name} Details", p.path, p.z + 0.5, np.where(dmask, LINE, 0), {}))
            layers.append(PieceLayer(p.name, p.path, p.z, out, {}))

        front |= visible[i]

    palette = {v: k for k, v in color_ids.items()}
    palette[LINE] = seg.line_color
    for L in layers:
        L.colors = palette
    return layers


def _sweep(idx, own, fronts, allowed, hw, factor):
    """Extend a part underneath the parts covering it, along the direction it is heading.

    For every front part it touches, the strip of this part along the shared
    edge is pushed forward step by step underneath it (and anything else in
    front).  The push length follows the width of the strip, capped at half
    the part's own length in that direction, and each ray stops as soon as it
    would come out from under the covering parts.
    """
    h, w = own.shape
    ys, xs = np.nonzero(own)
    if len(ys) == 0:
        return None
    pts_own = np.stack([xs, ys], 1).astype(np.float32)
    bw = max(2.0, hw * 0.75)
    pad = int(max(np.ptp(xs), np.ptp(ys)) * 0.6 + 4 * hw)
    Y0, Y1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)
    X0, X1 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
    own_w = own[Y0:Y1, X0:X1]
    allowed_w = allowed[Y0:Y1, X0:X1]
    idx_w = idx[Y0:Y1, X0:X1]
    hh, ww = own_w.shape
    out = np.zeros((hh, ww), np.int32)
    near_own = ndi.binary_dilation(own_w, iterations=int(bw + 2))
    for fm in fronts:
        f = fm[Y0:Y1, X0:X1]
        if not (f & near_own).any():
            continue
        dist_f = ndi.distance_transform_edt(~f)
        band = own_w & (dist_f <= bw + 1)
        if band.sum() < 4 * hw:
            continue
        gy, gx = np.gradient(ndi.gaussian_filter(dist_f, 2.0))
        lab, nb = ndi.label(band, structure=np.ones((3, 3)))
        for i in range(1, nb + 1):
            comp = lab == i
            if comp.sum() < 3 * hw:
                continue
            cy, cx = np.nonzero(comp)
            v = -np.array([gx[comp].mean(), gy[comp].mean()])
            nv = np.linalg.norm(v)
            if nv < 1e-3:
                continue
            v /= nv
            perp = np.array([-v[1], v[0]])
            cross = float(np.ptp(np.stack([cx, cy], 1) @ perp))
            if cross < 2 * hw:
                continue
            length = float(np.ptp(pts_own @ v))
            steps = int(min(max(3 * hw, factor * cross), 0.5 * length))
            if steps < 2:
                continue
            src = own_w & (dist_f <= bw + 3) & ndi.binary_dilation(comp, iterations=3)
            sy, sx = np.nonzero(src)
            vals = idx_w[sy, sx].copy()
            # Only the two side outlines continue underneath; any other ink in the
            # strip (corners, fold lines, the far edge) would smear into bars.
            side = np.stack([sx, sy], 1).astype(np.float32) @ perp
            lateral = (side <= side.min() + 2 * hw + 1) | (side >= side.max() - 2 * hw - 1)
            inner_ink = (vals == LINE) & ~lateral
            if inner_ink.any():
                fills_here = vals > 0
                if fills_here.any():
                    fy, fx = sy[fills_here], sx[fills_here]
                    for k in np.nonzero(inner_ink)[0]:
                        j = np.argmin((fy - sy[k]) ** 2 + (fx - sx[k]) ** 2)
                        vals[k] = vals[fills_here][j]
            alive = np.ones(len(sy), bool)
            for t in range(1, steps + 1):
                ny = np.round(sy + v[1] * t).astype(np.int64)
                nx = np.round(sx + v[0] * t).astype(np.int64)
                inb = (ny >= 0) & (ny < hh) & (nx >= 0) & (nx < ww)
                ok = alive & inb
                ok[ok] = allowed_w[ny[ok], nx[ok]]
                # once a ray has crossed the shared outline it dies when it would re-emerge
                if t > 2 * bw + 3:
                    alive &= ok
                if not alive.any():
                    break
                sel = ok.copy()
                sel[ok] = out[ny[ok], nx[ok]] == 0
                out[ny[sel], nx[sel]] = vals[sel]
    filled = out != 0
    if not filled.any():
        return None
    # close pinholes left by rounding
    closed = ndi.binary_closing(filled, iterations=1) & allowed_w
    if (closed & ~filled).any():
        _, nidx = ndi.distance_transform_edt(~filled, return_indices=True)
        out = np.where(closed & ~filled, out[nidx[0], nidx[1]], out)
    full = np.zeros((h, w), np.int32)
    full[Y0:Y1, X0:X1] = out
    return full


# ======================================================================
# render
# ======================================================================
@dataclass
class RenderedLayer:
    name: str
    path: list[str]
    z: float
    rgba: np.ndarray  # h x w x 4 uint8, cropped
    left: int
    top: int


def _smooth(mask: np.ndarray, scale: float, ss: int, sigma: float) -> np.ndarray:
    h, w = mask.shape
    S = scale * ss
    up = cv2.resize(mask.astype(np.float32), (int(round(w * S)), int(round(h * S))), interpolation=cv2.INTER_LINEAR)
    s = sigma * S
    k = int(np.ceil(s * 3)) * 2 + 1
    up = cv2.GaussianBlur(up, (k, k), s)
    hi = (up > 0.5).astype(np.float32)
    return cv2.resize(hi, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)


def render_layer(L: PieceLayer, scale: float = 1.0, sigma: float = 0.9, ss: int = 4) -> RenderedLayer | None:
    m = L.index != 0
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return None
    pad = 3
    H, W = m.shape
    y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad + 1)
    idx = L.index[y0:y1, x0:x1]
    alpha = _smooth(idx != 0, scale, ss, sigma)

    ids, counts = np.unique(idx[idx > 0], return_counts=True)
    order = [int(i) for i in ids[np.argsort(-counts)]]
    oh, ow = alpha.shape
    rgb = np.zeros((oh, ow, 3), np.float32)
    if order:
        rgb[:] = L.colors[order[0]]
        for c in order[1:]:
            a = _smooth(idx == c, scale, ss, sigma)[..., None]
            rgb = rgb * (1 - a) + np.array(L.colors[c], np.float32) * a
    else:
        rgb[:] = L.colors[LINE]
    if (idx == LINE).any():
        a = _smooth(idx == LINE, scale, ss, sigma)[..., None]
        if order:
            rgb = rgb * (1 - a) + np.array(L.colors[LINE], np.float32) * a
    out = np.dstack([np.clip(rgb + 0.5, 0, 255), np.clip(alpha * 255 + 0.5, 0, 255)]).astype(np.uint8)
    out[out[..., 3] == 0, :3] = 0
    return RenderedLayer(L.name, L.path, L.z, out, int(round(x0 * scale)), int(round(y0 * scale)))


def composite(layers: list[RenderedLayer], width: int, height: int) -> np.ndarray:
    """Alpha-composite rendered layers back to front."""
    acc = np.zeros((height, width, 4), np.float32)
    for L in sorted(layers, key=lambda l: l.z):
        h, w = L.rgba.shape[:2]
        y0, x0 = L.top, L.left
        y1, x1 = min(height, y0 + h), min(width, x0 + w)
        src = L.rgba[: y1 - y0, : x1 - x0].astype(np.float32) / 255
        dst = acc[y0:y1, x0:x1]
        sa = src[..., 3:4]
        da = dst[..., 3:4]
        oa = sa + da * (1 - sa)
        rgb = (src[..., :3] * sa + dst[..., :3] * da * (1 - sa)) / np.maximum(oa, 1e-6)
        acc[y0:y1, x0:x1] = np.dstack([rgb, oa])
    return np.clip(acc * 255 + 0.5, 0, 255).astype(np.uint8)


# ======================================================================
# psd
# ======================================================================
def _pascal(name: str) -> bytes:
    b = name.encode("ascii", "replace")[:255]
    data = bytes([len(b)]) + b
    data += b"\0" * ((4 - len(data) % 4) % 4)
    return data


def _block(key: bytes, payload: bytes) -> bytes:
    if len(payload) % 2:
        payload += b"\0"
    return b"8BIM" + key + struct.pack(">I", len(payload)) + payload


def _luni(name: str) -> bytes:
    u = name.encode("utf-16-be")
    return _block(b"luni", struct.pack(">I", len(u) // 2) + u)


def _record(name, top, left, bottom, right, chan_lens, blend=b"norm", flags=0x08, extra=b""):
    rec = struct.pack(">iiiiH", top, left, bottom, right, len(chan_lens))
    for cid, ln in chan_lens:
        rec += struct.pack(">hI", cid, ln)
    rec += b"8BIM" + blend + struct.pack(">BBBB", 255, 0, flags, 0)
    ex = struct.pack(">I", 0) + struct.pack(">I", 0) + _pascal(name) + _luni(name) + extra
    rec += struct.pack(">I", len(ex)) + ex
    return rec


class _Pixel:
    def __init__(self, name, rgba, left, top, visible=True):
        self.name, self.rgba, self.left, self.top, self.visible = name, rgba, left, top, visible


class _Group:
    def __init__(self, name):
        self.name = name
        self.children: list = []  # top-most first


def build_tree(layers) -> _Group:
    """layers: objects with name, path, z, rgba, left, top (any order)."""
    root = _Group("")
    for L in sorted(layers, key=lambda l: -l.z):
        g = root
        for folder in L.path:
            sub = next((c for c in g.children if isinstance(c, _Group) and c.name == folder), None)
            if sub is None:
                sub = _Group(folder)
                g.children.append(sub)
            g = sub
        g.children.append(_Pixel(L.name, L.rgba, L.left, L.top))
    return root


def write_psd(path: str, width: int, height: int, layers, merged: np.ndarray) -> None:
    root = build_tree(layers)
    records: list[bytes] = []
    chandata: list[bytes] = []

    def emit_pixel(p: _Pixel):
        h, w = p.rgba.shape[:2]
        chans = [(-1, p.rgba[..., 3]), (0, p.rgba[..., 0]), (1, p.rgba[..., 1]), (2, p.rgba[..., 2])]
        lens = []
        data = b""
        for cid, c in chans:
            raw = struct.pack(">H", 0) + np.ascontiguousarray(c).tobytes()
            lens.append((cid, len(raw)))
            data += raw
        flags = 0x08 | (0 if p.visible else 0x02)
        records.append(_record(p.name, p.top, p.left, p.top + h, p.left + w, lens, flags=flags))
        chandata.append(data)

    def emit_empty(name, kind):
        lens = [(-1, 2), (0, 2), (1, 2), (2, 2)]
        if kind == 3:
            extra = _block(b"lsct", struct.pack(">I", 3))
            records.append(_record(name, 0, 0, 0, 0, lens, flags=0x18, extra=extra))
        else:
            extra = _block(b"lsct", struct.pack(">I", 1) + b"8BIM" + b"pass")
            records.append(_record(name, 0, 0, 0, 0, lens, blend=b"pass", flags=0x18, extra=extra))
        chandata.append(struct.pack(">H", 0) * 4)

    # PSD lists layers bottom-up; a folder is "end marker, children, folder header".
    def emit(node):
        for child in reversed(node.children):
            if isinstance(child, _Group):
                emit_empty("</Layer group>", 3)
                emit(child)
                emit_empty(child.name, 1)
            else:
                emit_pixel(child)

    emit(root)

    layer_info = struct.pack(">h", -len(records)) + b"".join(records) + b"".join(chandata)
    if len(layer_info) % 2:
        layer_info += b"\0"
    layer_info = struct.pack(">I", len(layer_info)) + layer_info
    lmi = layer_info + struct.pack(">I", 0)  # empty global layer mask

    with open(path, "wb") as f:
        f.write(b"8BPS" + struct.pack(">H", 1) + b"\0" * 6)
        f.write(struct.pack(">HIIHH", 4, height, width, 8, 3))
        f.write(struct.pack(">I", 0))  # colour mode data
        f.write(struct.pack(">I", 0))  # image resources
        f.write(struct.pack(">I", len(lmi)) + lmi)
        f.write(struct.pack(">H", 0))  # raw merged image
        for c in (0, 1, 2, 3):
            f.write(np.ascontiguousarray(merged[..., c]).tobytes())


# ======================================================================
# preview
# ======================================================================
def save_preview(path, seg, parts, rendered, merged, thumb=260):
    h, w = seg.labels.shape
    owner = {r: p for p in parts for r in p.regions}
    rng = np.random.default_rng(7)
    pcol = {p.name: rng.integers(70, 255, 3) for p in parts}
    regmap = np.full((h, w, 3), 255, np.uint8)
    for rid in seg.regions:
        p = owner.get(rid)
        regmap[seg.labels == rid] = pcol[p.name] if p else (255, 0, 0)
    regmap[seg.line] = 0
    for rid, reg in seg.regions.items():
        org = (int(reg.centroid[0]) - 8, int(reg.centroid[1]) + 5)
        cv2.putText(regmap, str(rid), org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3)
        cv2.putText(regmap, str(rid), org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    big_h = thumb * 3
    left = []
    for img in (regmap, merged):
        im = Image.fromarray(img)
        if im.mode == "RGBA":
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            bg.alpha_composite(im)
            im = bg
        im = im.convert("RGB")
        im.thumbnail((10_000, big_h))
        left.append(im)

    layers = sorted(rendered, key=lambda l: -l.z)
    cols = 6
    rows = (len(layers) + cols - 1) // cols
    lw = sum(i.width for i in left) + 20
    sheet = Image.new("RGB", (lw + cols * thumb, max(big_h + 30, rows * (thumb + 24))), "white")
    d = ImageDraw.Draw(sheet)
    x = 0
    for i, im in enumerate(left):
        sheet.paste(im, (x, 24))
        d.text((x + 4, 4), ["regions (ids for parts.json)", "PSD composite"][i], fill="black")
        x += im.width + 10
    checker = (np.indices((thumb, thumb)).sum(0) // 8 % 2 * 40 + 200).astype(np.uint8)
    for i, L in enumerate(layers):
        cx = lw + (i % cols) * thumb
        cy = (i // cols) * (thumb + 24)
        im = Image.fromarray(L.rgba)
        im.thumbnail((thumb - 8, thumb - 8))
        bg = Image.fromarray(np.dstack([checker] * 3 + [np.full_like(checker, 255)])).crop((0, 0, im.width, im.height))
        bg.alpha_composite(im)
        sheet.paste(bg.convert("RGB"), (cx + 4, cy + 20))
        d.text((cx + 4, cy + 4), "/".join(L.path + [L.name])[:40], fill="black")
    sheet.save(path)


# ======================================================================
# __init__
# ======================================================================
def convert(
    src: str,
    out: str,
    *,
    scale: float = 1.0,
    parts_file: str | None = None,
    save_parts_file: str | None = None,
    preview: str | None = None,
    sweep: float = 1.0,
    smooth: float = 0.9,
    merge_de: float = 10.0,
    log=print,
):
    rgba = load_rgba(src)
    h, w = rgba.shape[:2]
    log(f"segmenting {w}x{h} image")
    seg = segment(rgba, merge_de=merge_de)
    log(f"  {len(seg.regions)} regions, outline width ~{seg.line_halfwidth * 2:.0f}px")
    parts = load_parts(parts_file) if parts_file else auto_parts(seg, merge_de=merge_de)
    missing = set(seg.regions) - {r for p in parts for r in p.regions}
    if missing:
        log(f"  warning: regions not assigned to any part: {sorted(missing)}")
    if save_parts_file:
        save_parts(parts, save_parts_file)
    log(f"  {len(parts)} parts: " + ", ".join(p.name for p in sorted(parts, key=lambda p: -p.z)))
    log("completing hidden areas and outlines")
    pieces = build_layers(seg, parts, sweep=sweep)
    log(f"rendering {len(pieces)} layers at {scale}x")
    rendered = [r for r in (render_layer(p, scale=scale, sigma=smooth) for p in pieces) if r is not None]
    W, H = int(round(w * scale)), int(round(h * scale))
    merged = composite(rendered, W, H)
    write_psd(out, W, H, rendered, merged)
    log(f"wrote {out}")
    if preview:
        save_preview(preview, seg, parts, rendered, merged)
        log(f"wrote {preview}")
    return seg, parts, rendered, merged


# ======================================================================
# __main__
# ======================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="live2d_psd_full.py",
        description="Turn a flat, outlined 2D character image into a layered PSD ready for Live2D Cubism rigging.",
    )
    ap.add_argument("image", help="input image (PNG/WebP/JPG; transparent background preferred)")
    ap.add_argument("-o", "--out", help="output .psd (default: <image>_cubism.psd)")
    ap.add_argument("--scale", type=float, default=1.0, help="render the PSD this many times larger (default 1)")
    ap.add_argument("--parts", help="use this parts.json instead of automatic part detection")
    ap.add_argument("--no-preview", action="store_true", help="don't write the preview PNG")
    ap.add_argument("--sweep", type=float, default=1.0, help="how far hidden parts continue under others (0 = off)")
    ap.add_argument("--smooth", type=float, default=0.9, help="edge smoothing radius in input pixels")
    ap.add_argument("--merge-colors", type=float, default=10.0, help="colours closer than this (CIE dE) are one colour")
    a = ap.parse_args(argv)

    base = os.path.splitext(a.out or a.image)[0]
    if not a.out:
        base += "_cubism"
    out = a.out or base + ".psd"
    convert(
        a.image,
        out,
        scale=a.scale,
        parts_file=a.parts,
        save_parts_file=base + ".parts.json",
        preview=None if a.no_preview else base + ".preview.png",
        sweep=a.sweep,
        smooth=a.smooth,
        merge_de=a.merge_colors,
    )


if __name__ == "__main__":
    main()
