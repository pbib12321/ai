"""Layer building for shaded / textured art: keep the original pixels.

Each part keeps the exact pixels of the input (shading, texture, soft hair
edges from the matte).  Areas hidden behind parts in front are completed the
same way as for flat art (holes filled, cross-sections swept underneath), and
then painted by carrying the real pixels along the sweep and inpainting the
rest from the part's own colours.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

from .complete import _kind, _sweep
from .render import RenderedLayer
from .semantic import Scene, SemPart


def build_texture_layers(sc: Scene, parts: list[SemPart], sweep: float = 1.0, background: bool = False, log=print,
                         carry: bool = False):
    """carry: hidden areas reached by the sweep copy the real pixels along it
    (outlines continue underneath; right for flat-colour art), instead of
    being inpainted (right for shaded/painted art, where copies streak)."""
    H, W = sc.fg.shape
    parts = sorted(parts, key=lambda p: -p.z)
    flat = (np.arange(H * W, dtype=np.int64) + 1).reshape(H, W)
    hw = max(1.5, max(H, W) / 1000)
    out: list[RenderedLayer] = []
    front = np.zeros((H, W), bool)
    for i, p in enumerate(parts):
        own = p.mask
        # only paint hidden areas under parts that are fully opaque there; under
        # a see-through edge (a gap between the legs) they would show
        hidden_ok = front & sc.fg & ~own & (sc.alpha > 0.97)
        complete = own.copy()
        rgb = np.zeros((H, W, 3), np.uint8)
        rgb[own] = sc.rgb[own]
        carried = np.zeros((H, W), bool)
        if hidden_ok.any():
            holes = ndi.binary_fill_holes(own) & hidden_ok
            if p.hidden_hint is not None:
                holes |= p.hidden_hint & hidden_ok
            complete |= holes
            if p.sweep:
                covers = [q.mask for q in parts[:i] if _kind(q.name) != _kind(p.name)]
                # the sweep decides the hidden shape; its colours come from
                # inpainting, since copied pixels smear texture into streaks
                ext = _sweep(flat * own, own, covers, hidden_ok, hw, sweep)
                if ext is not None:
                    take = (ext > 0) & ~complete
                    if carry:
                        rgb[take] = sc.rgb.reshape(-1, 3)[ext[take] - 1]
                        carried |= take
                    complete |= take
            todo = complete & ~own & ~carried
            if todo.any():
                rgb = _paint(rgb, own | carried, todo, p.fill)
        alpha = np.where(own, sc.alpha, 0).astype(np.float32)
        alpha[complete & ~own] = 1.0
        layer = _crop(p.name, p.path, p.z, rgb, alpha)
        if layer:
            out.append(layer)
        front |= own
    if background and sc.had_background:
        log("  painting the background behind the character")
        out.append(_background(sc))
    return out


def _paint(rgb, known, todo, mode):
    """Paint ``todo`` pixels from the ``known`` pixels of the same part.

    inpaint: Telea inpainting (run at reduced size for speed);
    color:   the part's median colour, blended into the edge (face skin and eye
             whites under hair or eyes are close to flat).
    """
    ys, xs = np.nonzero(known | todo)
    pad = 4
    y0, x0 = max(0, ys.min() - pad), max(0, xs.min() - pad)
    y1, x1 = min(rgb.shape[0], ys.max() + 1 + pad), min(rgb.shape[1], xs.max() + 1 + pad)
    crop = rgb[y0:y1, x0:x1].copy()
    k = known[y0:y1, x0:x1]
    t = todo[y0:y1, x0:x1]
    if not k.any():
        return rgb
    h, w = k.shape
    s = min(1.0, 400 / max(h, w))
    sw, sh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    src = crop.copy()
    src[~k] = 0
    if s < 1:
        # average only known pixels when shrinking
        wsum = cv2.resize(k.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
        small = cv2.resize(src.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA) / np.maximum(wsum[..., None], 1e-3)
        small = np.clip(small, 0, 255).astype(np.uint8)
        ks = wsum > 0.5
    else:
        small, ks = src, k
    painted = None
    if mode != "color" and t.sum() > 400:
        from .models import ai_inpaint

        # AI fill at full resolution for big hidden areas of textured parts
        ai = ai_inpaint(src, ~k)
        if ai is not None:
            painted = ai
    if painted is None:
        painted = cv2.inpaint(small, (~ks).astype(np.uint8), 5, cv2.INPAINT_TELEA) if (~ks).any() else small
        if s < 1:
            painted = cv2.resize(painted, (w, h), interpolation=cv2.INTER_LINEAR)
    if mode == "color":
        med = np.median(crop[k], 0).astype(np.float32)
        d = ndi.distance_transform_edt(~k)
        wgt = np.clip(d / max(4.0, 0.03 * max(h, w)), 0, 1)[..., None]
        painted = (painted * (1 - wgt) + med * wgt).astype(np.uint8)
    crop[t] = painted[t]
    rgb = rgb.copy()
    rgb[y0:y1, x0:x1] = crop
    return rgb


def _crop(name, path, z, rgb, alpha):
    m = alpha > 0.004
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    a = np.clip(alpha[y0:y1, x0:x1] * 255 + 0.5, 0, 255).astype(np.uint8)
    c = rgb[y0:y1, x0:x1].copy()
    c[a == 0] = 0
    return RenderedLayer(name, path, z, np.dstack([c, a]), int(x0), int(y0))


def _background(sc: Scene) -> RenderedLayer:
    H, W = sc.fg.shape
    hole = ndi.binary_dilation(sc.alpha > 0.02, iterations=4)
    s = min(1.0, 700 / max(H, W))
    small = cv2.resize(sc.rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    sh = cv2.resize(hole.astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    sh = cv2.dilate(sh, np.ones((3, 3), np.uint8))
    painted = cv2.inpaint(small, sh, 7, cv2.INPAINT_TELEA)
    big = cv2.resize(painted, (W, H), interpolation=cv2.INTER_LINEAR)
    bg = np.where(hole[..., None], big, sc.rgb)
    return RenderedLayer("Background", ["Background"], -100, np.dstack([bg, np.full((H, W), 255, np.uint8)]), 0, 0)
