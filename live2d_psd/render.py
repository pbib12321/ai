"""Render piece layers as clean, anti-aliased RGBA images.

Every colour area and every outline is treated as a shape: its mask is
upsampled, smoothed and re-thresholded at high resolution, then averaged
back down.  This removes JPEG noise, pixel stair-steps and colour fringes,
much like tracing the art to vectors and rasterising it again, and lets the
output be rendered larger than the input (--scale).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .complete import LINE, PieceLayer


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
