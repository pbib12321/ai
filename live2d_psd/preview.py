"""Preview sheet: numbered region map (for editing parts.json) plus every layer."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw


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
