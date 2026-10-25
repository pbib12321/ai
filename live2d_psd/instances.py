"""Anime character instance segmentation: one mask per character.

CartoonSegmentation's RTMDet-Ins-L ("Instance-guided Cartoon Editing with a
Large-scale Dataset", weights MIT, huggingface.co/dreMaz/AnimeInstanceSegmentation),
exported to ONNX by tools/export_animeins.py.  The post-processing of
mmdet's RTMDetInsHead (box decoding, NMS, per-instance dynamic convolutions
for the masks) is reimplemented here in NumPy, so only onnxruntime is needed.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from . import models

# The graph (no weights) ships in models/; the weights are the original MIT
# checkpoint, fetched from Hugging Face on first use and loaded with torch.
NAME = "animeins_rtmdet_graph.onnx"
CKPT = "animeins_rtmdetl_e60.ckpt"
models._URLS[CKPT] = "https://huggingface.co/dreMaz/AnimeInstanceSegmentation/resolve/main/rtmdetl_e60.ckpt"
_sess = None
_failed = False

SIZE = 640
STRIDES = (8, 16, 32)
MEAN = np.array([103.53, 116.28, 123.675], np.float32)  # BGR
STD = np.array([57.375, 57.12, 58.395], np.float32)
WEIGHT_NUMS, BIAS_NUMS, DYC = (80, 64, 8), (8, 8, 1), 8


def available() -> bool:
    if _failed or os.environ.get("LIVE2D_PSD_INSTANCES", "1") == "0" or models._path(NAME) is None:
        return False
    import importlib.util

    return importlib.util.find_spec("torch") is not None


def _session_from(graph: str, ckpt: str):
    import onnxruntime as ort
    import torch

    try:
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    except Exception:  # older torch / extra metadata in the checkpoint
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    sess = ort.InferenceSession(graph, providers=["CPUExecutionProvider"])
    feed = {}
    for i in sess.get_inputs():
        if i.name == "image":
            continue
        key = i.name[2:] if i.name.startswith("m.") else i.name
        if key not in sd:
            raise KeyError(f"weight {i.name} missing from the checkpoint")
        feed[i.name] = np.ascontiguousarray(sd[key].float().numpy())
    return _Net(sess, feed)


class _Net:
    """The ONNX graph plus its weights, fed as inputs on every run."""

    def __init__(self, sess, weights):
        self.sess, self.weights = sess, weights

    def run(self, names, feed):
        return self.sess.run(names, {**self.weights, **feed})


def _session(log=print):
    global _sess, _failed
    if _sess is None and not _failed:
        try:
            graph = models._path(NAME)
            ckpt = models.fetch(CKPT)
            if graph is None or ckpt is None:
                raise RuntimeError("model files missing")
            _sess = _session_from(graph, ckpt)
        except Exception as e:
            log(f"  character detector unavailable ({e})")
            _failed = True
    return _sess


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _nms(boxes, scores, thr):
    order = np.argsort(-scores)
    keep = []
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while len(order):
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (area[i] + area[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= thr]
    return np.array(keep, int)


def _masks(feat, kernels, points, strides):
    """Dynamic 1x1 convs: (8+2 relative coords) -> 8 -> 8 -> 1 per instance."""
    C, h, w = feat.shape
    ys, xs = np.mgrid[0:h, 0:w]
    coord = np.stack([xs.ravel() * STRIDES[0], ys.ravel() * STRIDES[0]], 1).astype(np.float32)  # (hw, 2)
    out = []
    for k, p, s in zip(kernels, points, strides):
        rel = ((p[None] - coord) / (s * 8)).T  # (2, hw)
        x = np.concatenate([rel, feat.reshape(C, -1)], 0)  # (10, hw)
        parts = np.split(k, np.cumsum(WEIGHT_NUMS + BIAS_NUMS)[:-1])
        ws, bs = parts[:3], parts[3:]
        for i in range(3):
            W = ws[i].reshape(DYC if i < 2 else 1, -1)
            x = W @ x + bs[i][:, None]
            if i < 2:
                x = np.maximum(x, 0)
        out.append(x.reshape(h, w))
    return np.array(out, np.float32)


def detect(rgb: np.ndarray, score_thr: float = 0.3, log=print) -> list[dict]:
    """Characters in the picture, best first: dicts with ``mask`` (bool HxW),
    ``prob`` (float HxW), ``score`` and ``box`` (x0, y0, x1, y1)."""
    if not available():
        return []
    s = _session(log)
    if s is None:
        return []
    H, W = rgb.shape[:2]
    k = SIZE / max(H, W)
    nh, nw = max(1, round(H * k)), max(1, round(W * k))
    img = np.full((SIZE, SIZE, 3), 114, np.float32)
    img[:nh, :nw] = cv2.resize(rgb[..., ::-1], (nw, nh), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
    x = ((img - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    outs = s.run(None, {"image": x})
    cls, box, ker, feat = outs[0:3], outs[3:6], outs[6:9], outs[9][0]

    boxes, scores, kernels, points, strides = [], [], [], [], []
    for lv, st in enumerate(STRIDES):
        c = _sigmoid(cls[lv][0, 0]).ravel()
        idx = np.nonzero(c > 0.05)[0]
        if len(idx) > 1000:
            idx = idx[np.argsort(-c[idx])[:1000]]
        if not len(idx):
            continue
        h, w = cls[lv].shape[-2:]
        px = (idx % w).astype(np.float32) * st
        py = (idx // w).astype(np.float32) * st
        d = box[lv][0].reshape(4, -1)[:, idx]
        b = np.stack([px - d[0], py - d[1], px + d[2], py + d[3]], 1)
        b[:, 0::2] = b[:, 0::2].clip(0, SIZE)
        b[:, 1::2] = b[:, 1::2].clip(0, SIZE)
        boxes.append(b)
        scores.append(c[idx])
        kernels.append(ker[lv][0].reshape(ker[lv].shape[1], -1)[:, idx].T)
        points.append(np.stack([px, py], 1))
        strides.append(np.full(len(idx), st, np.float32))
    if not boxes:
        return []
    boxes, scores, kernels = np.concatenate(boxes), np.concatenate(scores), np.concatenate(kernels)
    points, strides = np.concatenate(points), np.concatenate(strides)
    keep = _nms(boxes, scores, 0.6)[:100]
    keep = keep[scores[keep] > score_thr]
    if not len(keep):
        return []
    logits = _masks(feat, kernels[keep], points[keep], strides[keep])  # (n, 80, 80) at stride 8
    res = []
    for i, j in enumerate(keep):
        lg = cv2.resize(logits[i], (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)[:nh, :nw]
        prob = _sigmoid(cv2.resize(lg, (W, H), interpolation=cv2.INTER_LINEAR))
        res.append(dict(mask=prob > 0.5, prob=prob.astype(np.float32), score=float(scores[j]),
                        box=tuple(float(v) / k for v in boxes[j])))
    res = [r for r in res if r["mask"].sum() > 0.002 * H * W]
    log(f"  character detector: {len(res)} character{'s' if len(res) != 1 else ''}")
    return res
