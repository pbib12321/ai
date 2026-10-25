"""Atoms: small regions that never straddle a drawn line or a sharp colour edge.

Parts are built from whole atoms, so their edges follow the artist's lines.
Flat / cel-shaded art gives line-bounded colour regions; painted art with
gradients falls back to superpixels.  Ink lines are kept separate and handed
to whichever part is in front later.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import slic

from .segment import _fill_nearest, _line_masks, _palette


@dataclass
class Atoms:
    labels: np.ndarray  # int32, 0 = background or ink
    line: np.ndarray  # bool ink strokes
    blob: np.ndarray  # bool solid dark shapes (also atoms)
    hw: float  # stroke half width
    n: int
    area: np.ndarray  # per atom
    mean_lab: np.ndarray  # n+1 x 3
    cy: np.ndarray
    cx: np.ndarray


def make_atoms(rgb: np.ndarray, fg: np.ndarray, lab: np.ndarray, max_atoms: int = 2500) -> Atoms:
    h, w = fg.shape
    rgba = np.dstack([rgb, np.where(fg, 255, 0).astype(np.uint8)])
    line, blob, hw, _ = _line_masks(rgba, fg)
    fill = fg & ~line
    n_fg = int(fg.sum())
    min_region = max(20, int(n_fg * 3e-5))

    # colour regions inside the line art
    near_ink = ndi.binary_dilation(line, iterations=max(1, int(round(hw))))
    core = fill & ~near_ink
    pal = _palette(lab[core], min_count=min_region, merge_de=7.0)
    d = np.linalg.norm(lab[core][:, None, :] - pal[None], axis=2)
    cls = np.zeros((h, w), np.int32)
    cls[core] = d.argmin(1) + 1
    cls = _fill_nearest(cls, fill & (cls == 0))
    cls[~fill] = 0
    labels = np.zeros((h, w), np.int32)
    nid = 0
    for k in range(1, len(pal) + 1):
        m = cls == k
        if m.any():
            lb, n = ndi.label(m)
            labels[m] = lb[m] + nid
            nid += n
    sizes = np.bincount(labels.ravel(), minlength=nid + 1)
    tiny = sizes < min_region
    tiny[0] = False
    t = tiny[labels]
    labels[t] = 0
    labels = _fill_nearest(labels, t)
    labels[~fill] = 0
    nreal = len(np.unique(labels)) - 1

    if nreal > max_atoms:
        # gradients everywhere: colour regions shatter, use superpixels instead
        n_seg = int(np.clip(n_fg / 400, 300, max_atoms))
        sp = slic(rgb, n_segments=n_seg, compactness=12, mask=fill, start_label=1, channel_axis=-1)
        labels = sp.astype(np.int32)
        labels[~fill] = 0

    ids = np.unique(labels)
    ids = ids[ids > 0]
    remap = np.zeros(labels.max() + 1, np.int32)
    remap[ids] = np.arange(1, len(ids) + 1)
    labels = remap[labels]
    n = len(ids)
    area = np.bincount(labels.ravel(), minlength=n + 1).astype(np.float64)
    mean_lab = np.zeros((n + 1, 3))
    for c in range(3):
        mean_lab[:, c] = np.bincount(labels.ravel(), weights=lab[..., c].ravel(), minlength=n + 1) / np.maximum(area, 1)
    yy, xx = np.mgrid[0:h, 0:w]
    cy = np.bincount(labels.ravel(), weights=yy.ravel(), minlength=n + 1) / np.maximum(area, 1)
    cx = np.bincount(labels.ravel(), weights=xx.ravel(), minlength=n + 1) / np.maximum(area, 1)
    return Atoms(labels, line, blob, hw, n, area, mean_lab, cy, cx)


def vote(atoms: Atoms, proposal: np.ndarray, nlab: int, split_share: float = 0.6, split_min: int = 400):
    """Give each atom the label most of its pixels propose.

    proposal: int map, 0 = no opinion.  Large atoms whose pixels disagree a
    lot keep the per-pixel proposal instead (a sleeve and torso drawn as one
    white area).  Returns an int map over atom pixels; 0 where nobody voted.
    """
    L = atoms.labels
    key = L.ravel().astype(np.int64) * (nlab + 1) + proposal.ravel()
    counts = np.bincount(key, minlength=(atoms.n + 1) * (nlab + 1)).reshape(atoms.n + 1, nlab + 1)
    counts[:, 0] = 0
    total = counts.sum(1)
    best = counts.argmax(1)
    share = counts.max(1) / np.maximum(total, 1)
    best[total == 0] = 0
    out = best[L]
    mixed = (share < split_share) & (atoms.area >= split_min) & (total > 0)
    mixed[0] = False
    mm = mixed[L]
    out[mm] = proposal[mm]
    out[L == 0] = 0
    return out


def background_atoms(atoms: Atoms, lab: np.ndarray, matte: np.ndarray, protect: np.ndarray | None = None) -> np.ndarray:
    """Atoms that are really background the matte let through (railings, sky
    seen between an arm and the body).  Returns a bool mask to remove."""
    bgpx = lab[matte < 0.05]
    if len(bgpx) < 100:
        return np.zeros(matte.shape, bool)
    bgpx = bgpx[:: max(1, len(bgpx) // 30000)].astype(np.float32)
    import cv2

    k = min(10, len(bgpx))
    _, _, cen = cv2.kmeans(bgpx, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0), 2, cv2.KMEANS_PP_CENTERS)
    d_bg = np.linalg.norm(atoms.mean_lab[:, None, :] - cen[None], axis=2).min(1)
    L = atoms.labels
    mean_m = np.bincount(L.ravel(), weights=matte.ravel(), minlength=atoms.n + 1) / np.maximum(atoms.area, 1)
    fg_area = atoms.area[1:].sum()
    # unsure matte + background colour, unless SAM sees it as part of the character
    prot = np.zeros(atoms.n + 1, bool)
    if protect is not None:
        share = np.bincount(L.ravel(), weights=protect.ravel().astype(np.float64), minlength=atoms.n + 1) / np.maximum(atoms.area, 1)
        prot = share > 0.5
    drop = (d_bg < 7) & (mean_m < 0.95) & ~prot
    # small enclosed patch with exactly the background's colour (sky between arm and body)
    border = np.zeros(atoms.n + 1, bool)
    border[np.unique(np.concatenate([L[0], L[-1], L[:, 0], L[:, -1]]))] = True
    # (only when the character has little else of that colour, so a white
    # shirt in front of a white wall survives)
    cand = np.nonzero((d_bg < 4) & (atoms.area < 0.05 * fg_area) & ~border & _isolated_colour(atoms, d_tol=20))[0]
    for c in cand:
        same = np.linalg.norm(atoms.mean_lab - atoms.mean_lab[c], axis=1) < 10
        same[c] = same[0] = False
        if atoms.area[same].sum() < 3 * atoms.area[c] and not prot[c]:
            drop[c] = True
    drop[0] = False
    return drop[L]


def _isolated_colour(atoms: Atoms, d_tol: float) -> np.ndarray:
    """Atoms whose colour no touching atom shares (a hole, not part of a surface)."""
    L = atoms.labels
    out = np.ones(atoms.n + 1, bool)
    for a, b in ((L[:, :-1], L[:, 1:]), (L[:-1], L[1:])):
        m = (a != b) & (a > 0) & (b > 0)
        pa, pb = a[m], b[m]
        close = np.linalg.norm(atoms.mean_lab[pa] - atoms.mean_lab[pb], axis=1) < d_tol
        out[pa[close]] = False
        out[pb[close]] = False
    return out
