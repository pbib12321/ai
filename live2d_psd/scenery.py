"""Animated background: split the scene behind the character into moving layers.

* Sky: the smooth region connected to the top edge (any colour) becomes
  a scrolling, seamlessly looping texture seen through a fixed mask, so
  clouds drift slowly.
* Plants: green foliage becomes separate layers that sway in the wind,
  hinged where they enter the picture (palm fronds from the side, bushes
  from the bottom).
* Grass: wide strips of green along the bottom wave blade-by-blade.
* Base: everything else, with the character and the moving pieces painted
  out underneath so nothing tears when they move.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.color import rgb2lab

from .render import RenderedLayer


def _inpaint(img, hole, max_side=700, radius=7):
    if not hole.any():
        return img
    from .models import ai_inpaint

    ai = ai_inpaint(img, hole)
    if ai is not None:
        return ai
    H, W = hole.shape
    s = min(1.0, max_side / max(H, W))
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img.copy()
    sh = cv2.resize(hole.astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    sh = cv2.dilate(sh, np.ones((3, 3), np.uint8))
    p = cv2.inpaint(small, sh, radius, cv2.INPAINT_TELEA)
    big = cv2.resize(p, (W, H), interpolation=cv2.INTER_LINEAR) if s < 1 else p
    return np.where(hole[..., None], big, img)


def build_scenery(rgb: np.ndarray, alpha: np.ndarray, animate: bool = True, log=print) -> list[RenderedLayer]:
    H, W = alpha.shape
    char = ndi.binary_dilation(alpha > 0.02, iterations=4)
    bg = _inpaint(rgb, char)
    layers = []
    if not animate:
        return [RenderedLayer("Background", ["Background"], -100, np.dstack([bg, np.full((H, W), 255, np.uint8)]), 0, 0)]
    lab = rgb2lab(bg.astype(np.float32) / 255)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    yy = np.mgrid[0:H, 0:W][0]

    # ---- sky: the smooth region connected to the top edge, any colour
    # (blue day, orange sunset, purple dusk); clouds are soft, not detailed
    gray = cv2.cvtColor(bg, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx, gy = np.gradient(cv2.GaussianBlur(gray, (0, 0), 1.5))
    detail = cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), 3)
    green_like = (a < -6) & (b > 2)
    cand = (detail < 4.0) & (yy < 0.75 * H) & ~green_like & (L > 15)
    cand = ndi.binary_opening(cand, iterations=2)
    lb, n = ndi.label(cand)
    top_ids = np.unique(lb[0][lb[0] > 0])
    sky = np.isin(lb, top_ids) if len(top_ids) else np.zeros((H, W), bool)
    sky = ndi.binary_closing(sky, iterations=4)
    sky = ndi.binary_fill_holes(sky) & (yy < 0.8 * H)
    # a flat backdrop gains nothing from scrolling: it needs clouds or a gradient
    has_sky = sky.mean() > 0.04 and float(rgb[sky].std()) > 6

    # ---- vegetation: green, not sky
    green = green_like & (L > 12) & (L < 92) & ~sky  # green, not teal or blue
    green = ndi.binary_opening(green, iterations=1)
    green = ndi.binary_closing(green, iterations=3)
    lb, n = ndi.label(green)
    plants, grass = [], np.zeros((H, W), bool)
    for i, sl in enumerate(ndi.find_objects(lb), start=1):
        if sl is None:
            continue
        m = lb == i
        area = m.sum()
        if area < 0.004 * H * W:
            continue
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        if y0 > 0.6 * H and (x1 - x0) > 2.0 * (y1 - y0):
            grass |= m
        else:
            plants.append(m)
    moving = grass.copy()
    for m in plants:
        moving |= m
    base = _inpaint(bg, ndi.binary_dilation(moving, iterations=3)) if moving.any() else bg
    layers.append(RenderedLayer("Background", ["Background"], -100, np.dstack([base, np.full((H, W), 255, np.uint8)]), 0, 0))

    if has_sky:
        # sky colours everywhere (non-sky painted from the sky), then made to wrap
        tex = _inpaint(bg, ~sky)
        x = np.linspace(0, 1, W, endpoint=False)
        w = (1 - np.abs(2 * x - 1))[None, :, None]  # 0 at the edges, 1 in the middle
        tex = (tex * w + np.roll(tex, W // 2, axis=1) * (1 - w)).astype(np.uint8)
        mask = (cv2.GaussianBlur(sky.astype(np.float32), (0, 0), 2) * 255).astype(np.uint8)
        layer = RenderedLayer("Sky", ["Background"], -99, np.dstack([tex, np.full((H, W), 255, np.uint8)]), 0, 0)
        layer.extra = {"mask": mask}
        layers.append(layer)
        log("  sky: clouds will drift and loop")

    for k, m in enumerate(sorted(plants, key=lambda m: -m.sum()), start=1):
        ys, xs = np.nonzero(m)
        # hinge where the plant comes into the picture
        dists = {"bottom": H - 1 - ys.max(), "left": xs.min(), "right": W - 1 - xs.max(), "top": ys.min()}
        edge = min(dists, key=dists.get)
        if dists[edge] > 0.05 * max(H, W):
            edge = "bottom"
        sel = {"bottom": ys >= ys.max() - 3, "top": ys <= ys.min() + 3, "left": xs <= xs.min() + 3, "right": xs >= xs.max() - 3}[edge]
        anchor = (float(xs[sel].mean()), float(ys[sel].mean()))
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        a_ = (cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.0)[y0:y1, x0:x1] * 255).astype(np.uint8)
        layer = RenderedLayer(f"Plant {k}", ["Background"], -98 + 0.01 * k, np.dstack([bg[y0:y1, x0:x1], a_]), int(x0), int(y0))
        layer.extra = {"anchor": anchor}
        layers.append(layer)
    if plants:
        log(f"  {len(plants)} plant(s) will sway in the wind")
    if grass.any():
        ys, xs = np.nonzero(grass)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        a_ = (cv2.GaussianBlur(grass.astype(np.float32), (0, 0), 1.0)[y0:y1, x0:x1] * 255).astype(np.uint8)
        layers.append(RenderedLayer("Grass", ["Background"], -97, np.dstack([bg[y0:y1, x0:x1], a_]), int(x0), int(y0)))
        log("  grass will wave")
    return layers
