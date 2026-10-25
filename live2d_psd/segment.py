"""Split a flat-coloured, outlined 2D illustration into line art and colour regions.

The output of this module is purely geometric: a label image where every
non-line foreground pixel belongs to exactly one colour region, a mask of the
line art, and per-region statistics.  Semantic grouping happens in parts.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.color import rgb2lab
from skimage.morphology import skeletonize


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
