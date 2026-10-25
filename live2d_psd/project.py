"""Editable project: the source image, a per-pixel part map and the part list.

Every conversion saves one next to the PSD (``<name>_cubism.l2d/``) so the
result can be fixed in the editor and rebuilt without re-running detection:

    labels.png   16-bit image, pixel value = part id (0 = not part of the character)
    project.json source image, mode, settings and the parts (id, name, folder, depth, options)
"""
from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image


def save_project(folder, src, mode, parts_meta, labels, alpha, settings):
    os.makedirs(folder, exist_ok=True)
    Image.fromarray(labels.astype(np.uint16)).save(os.path.join(folder, "labels.png"))
    Image.fromarray(np.clip(alpha * 255 + 0.5, 0, 255).astype(np.uint8)).save(os.path.join(folder, "alpha.png"))
    with open(os.path.join(folder, "project.json"), "w") as f:
        json.dump({"source": os.path.abspath(src), "mode": mode, "settings": settings, "parts": parts_meta}, f, indent=1)


def load_project(folder):
    with open(os.path.join(folder, "project.json")) as f:
        proj = json.load(f)
    labels = np.array(Image.open(os.path.join(folder, "labels.png"))).astype(np.int32)
    alpha = np.array(Image.open(os.path.join(folder, "alpha.png"))).astype(np.float32) / 255
    return proj, labels, alpha


def meta_of(part, pid):
    d = {"id": pid, "name": part.name, "path": list(part.path), "z": float(part.z), "sweep": bool(part.sweep)}
    for k in ("kind", "line_layer", "split_lines", "fill"):
        v = getattr(part, k, None)
        if v not in (None, False):
            d[k] = v
    return d
