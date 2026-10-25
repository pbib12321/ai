"""Anime body-part segmentation with See-through's fine-tuned SAM-HQ (ViT-H).

https://huggingface.co/24yearsold/l2d_sam_iter2 (Apache-2.0).  19 classes:
hair, headwear, face, eyes, eyewear, ears, earwear, nose, mouth, neck,
neckwear, topwear, handwear, bottomwear, legwear, footwear, tail, wings,
objects.  Needs PyTorch; uses the Apple GPU (MPS) on Apple Silicon.
"""
from __future__ import annotations

import os

import numpy as np

from . import models

CLASSES = [
    "hair", "headwear", "face", "eyes", "eyewear", "ears", "earwear", "nose", "mouth", "neck",
    "neckwear", "topwear", "handwear", "bottomwear", "legwear", "footwear", "tail", "wings", "objects",
]
URL = "https://huggingface.co/24yearsold/l2d_sam_iter2/resolve/main/checkpoint-18000.pt"
NAME = "l2d_sam_checkpoint-18000.pt"
models._URLS[NAME] = URL

_model = None
_failed = False


def available() -> bool:
    if _failed or os.environ.get("LIVE2D_PSD_PARTSEG", "1") == "0":
        return False
    import importlib.util

    return importlib.util.find_spec("torch") is not None


def _load(log=print):
    global _model, _failed
    if _model is not None or _failed:
        return _model
    try:
        # run ops the Apple GPU lacks on the CPU instead of failing
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        import torch

        from .vendor.seethrough.semanticsam import SemanticSam

        path = models.fetch(NAME)
        if path is None:
            raise RuntimeError("could not download the part segmentation model")
        log("  loading the anime part segmentation model (SAM-HQ ViT-H, first load is slow)")
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if "state_dict" in sd:
            sd = sd["state_dict"]
        with torch.device("meta"):
            m = SemanticSam(class_num=19)
        m.load_state_dict(sd, strict=False, assign=True)
        dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        m = m.to(dev).eval()
        if dev == "cpu":
            torch.set_num_threads(max(1, os.cpu_count() or 1))
        _model = (m, dev)
    except Exception as e:
        log(f"  part segmentation model unavailable ({e}); using the rule-based parser")
        _failed = True
    return _model


def predict(rgb: np.ndarray, log=print) -> np.ndarray | None:
    """Per-class logits (19, H, W) for the picture, or None."""
    if not available():
        return None
    mm = _load(log)
    if mm is None:
        return None
    import torch

    m, dev = mm
    with torch.inference_mode():
        out = m.inference(np.ascontiguousarray(rgb))[0]
    return out.float().cpu().numpy()


def predict_tiled(rgb: np.ndarray, base: np.ndarray | None = None, log=print) -> np.ndarray | None:
    """Logits from overlapping square crops along the long side, blended with
    the whole-picture prediction.  In group pictures each character is then
    seen bigger and with fewer neighbours, so smaller faces are found."""
    H, W = rgb.shape[:2]
    if base is None:
        base = predict(rgb, log)
        if base is None:
            return None
    S = min(H, W)
    L = max(H, W)
    n = int(np.ceil((L - S) / (S * 2 / 3))) + 1
    if n < 2:
        return base
    starts = np.linspace(0, L - S, n).round().astype(int)
    # per class, the most confident view wins (a face seen only in one crop
    # must survive); views are trusted less towards their cut edges
    out = base - 1.0
    ramp = np.minimum(np.arange(S), np.arange(S)[::-1]).astype(np.float32)
    pen = 10.0 * (1.0 - np.clip(ramp / (0.25 * S), 0.0, 1.0)) ** 2
    for st in starts:
        crop = rgb[:, st:st + S] if W >= H else rgb[st:st + S]
        lg = predict(np.ascontiguousarray(crop), log)
        if lg is None:
            return base
        if W >= H:
            edge = pen[None, :].copy()
            if st == 0:
                edge[:, : S // 2] = 0  # the picture's own border is not a cut
            if st + S >= W:
                edge[:, S // 2:] = 0
            sl = out[:, :, st:st + S]
            np.maximum(sl, lg - edge[None], out=sl)
        else:
            edge = pen[:, None].copy()
            if st == 0:
                edge[: S // 2] = 0
            if st + S >= H:
                edge[S // 2:] = 0
            sl = out[:, st:st + S]
            np.maximum(sl, lg - edge[None], out=sl)
    return out
