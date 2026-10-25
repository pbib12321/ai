"""Turn labelled parts into complete, self-contained art pieces.

For every part this module decides which outline strokes belong to it and
paints the areas hidden behind the parts in front of it, the way a Live2D
artist prepares a PSD:

* each piece keeps its own outline (the stroke between two parts belongs to
  the front one, the back piece continues underneath it);
* holes left by features on top (eyes on a face, a badge on a shirt) are
  filled with the surrounding colour;
* where a piece disappears under another (legs under a skirt, neck under the
  head, ears behind the face) its cross-section is swept onward underneath,
  carrying its side outlines with it.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi

from .parts import Part
from .segment import Segmentation

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
