"""Optional machine-learning helpers.

Everything here is optional: when a model or library is missing, the caller
gets ``None`` and falls back to the classic image-processing path.

Models are fetched on first use into ``~/.cache/live2d_psd`` (override with
the ``LIVE2D_PSD_MODELS`` environment variable):

* ISNet (DIS) background removal, taken from the ``@imgly/background-removal-node``
  npm package (AGPL-3.0, so for personal use, not for redistribution in
  closed products).
* MediaPipe pose landmarker (Apache-2.0) from Google's model storage.
* lbpcascade_animeface (MIT) anime face detector from GitHub.
* anime-segmentation's ISNet-IS (skytnt, Apache-2.0), trained on anime
  characters; much better than the general ISNet on crowded anime pictures.
* MobileSAM (Apache-2.0) exported to ONNX; shipped in the repository's
  ``models/`` folder, see tools/export_mobilesam.py.
"""
from __future__ import annotations

import io
import json
import os
import tarfile
import urllib.request

import cv2
import numpy as np

CACHE = os.environ.get("LIVE2D_PSD_MODELS") or os.path.join(os.path.expanduser("~"), ".cache", "live2d_psd")
REPO_MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

_URLS = {
    "pose_landmarker_heavy.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    "lbpcascade_animeface.xml": "https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml",
    # MI-GAN (Picsart AI Research, ICCV 2023): a small inpainting GAN, ONNX pipeline
    "migan_pipeline_v2.onnx": "https://huggingface.co/andraniksargsyan/migan/resolve/main/migan_pipeline_v2.onnx",
    # skytnt/anime-segmentation: ISNet trained on anime characters (several per picture)
    "isnetis.onnx": "https://huggingface.co/skytnt/anime-seg/resolve/main/isnetis.onnx",
}
AI_INPAINT = os.environ.get("LIVE2D_PSD_AI_INPAINT", "1") != "0"
_IMGLY = "https://registry.npmjs.org/@imgly/background-removal-node/-/background-removal-node-1.4.5.tgz"

_log = print


def set_logger(fn):
    global _log
    _log = fn


def _path(name: str) -> str | None:
    for d in (REPO_MODELS, CACHE):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def _download(url: str) -> bytes:
    _log(f"  downloading {url.split('/')[-1]} (first run only)")
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def fetch(name: str) -> str | None:
    p = _path(name)
    if p:
        return p
    os.makedirs(CACHE, exist_ok=True)
    dst = os.path.join(CACHE, name)
    try:
        if name in _URLS:
            data = _download(_URLS[name])
        elif name in ("isnet_medium.onnx", "isnet_small.onnx"):
            data = _isnet_from_npm(name)
        else:
            return None
    except Exception as e:  # offline, blocked, ...
        _log(f"  could not get {name}: {e}")
        return None
    with open(dst + ".part", "wb") as f:
        f.write(data)
    os.replace(dst + ".part", dst)
    return dst


def _isnet_from_npm(name: str) -> bytes:
    # The npm package stores each ONNX model as 4 MB chunks named by hash.
    tgz = tarfile.open(fileobj=io.BytesIO(_download(_IMGLY)), mode="r:gz")
    files = {m.name.split("/")[-1]: m for m in tgz.getmembers() if m.isfile()}
    res = json.load(tgz.extractfile(files["resources.json"]))
    key = "/models/medium" if "medium" in name else "/models/small"
    return b"".join(tgz.extractfile(files[c["hash"]]).read() for c in res[key]["chunks"])


# --------------------------------------------------------------------------- ISNet


_sessions: dict = {}


def _ort(name):
    if name in _sessions:
        return _sessions[name]
    try:
        import onnxruntime as ort
    except ImportError:
        _sessions[name] = None
        return None
    p = fetch(name)
    s = ort.InferenceSession(p, providers=["CPUExecutionProvider"]) if p else None
    _sessions[name] = s
    return s


def _isnet_general(rgb):
    s = _ort("isnet_medium.onnx") or _ort("isnet_small.onnx")
    if s is None:
        return None
    h, w = rgb.shape[:2]
    x = cv2.resize(rgb, (1024, 1024), interpolation=cv2.INTER_AREA).astype(np.float32) / 255 - 0.5
    o = s.run(["output"], {"input": x.transpose(2, 0, 1)[None]})[0][0, 0]
    o = (o - o.min()) / (o.max() - o.min() + 1e-6)
    return cv2.resize(o, (w, h), interpolation=cv2.INTER_LINEAR)


def _isnet_anime(rgb):
    s = _ort("isnetis.onnx")
    if s is None:
        return None
    S = 1024
    h, w = rgb.shape[:2]
    k = S / max(h, w)
    nh, nw = max(1, int(h * k)), max(1, int(w * k))
    ph, pw = (S - nh) // 2, (S - nw) // 2
    x = np.zeros((S, S, 3), np.float32)
    x[ph:ph + nh, pw:pw + nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    o = s.run(None, {"img": x.transpose(2, 0, 1)[None]})[0][0, 0][ph:ph + nh, pw:pw + nw]
    return np.clip(cv2.resize(o, (w, h), interpolation=cv2.INTER_LINEAR), 0, 1)


def _unsure(m):
    return float(np.mean((m > 0.15) & (m < 0.85)))


def remove_background(rgb: np.ndarray) -> np.ndarray | None:
    """Soft foreground matte in [0, 1] for the character(s), or None.

    Runs the anime-trained and the general ISNet and keeps the one that is
    more certain about this picture (fewer half-transparent pixels): the
    anime model keeps every character in group pictures, the general one is
    better on western comics and photo-like art."""
    mattes = [m for m in (_isnet_anime(rgb), _isnet_general(rgb)) if m is not None]
    if not mattes:
        return None
    return min(mattes, key=_unsure)


# --------------------------------------------------------------------------- inpainting

_ai_failed = False


def ai_inpaint(rgb: np.ndarray, hole: np.ndarray) -> np.ndarray | None:
    """Fill ``hole`` with MI-GAN, or return None (then the caller uses OpenCV).

    The ONNX pipeline takes a uint8 RGB image and a uint8 mask where 255 marks
    known pixels and 0 the area to fill; it does its own resizing.
    """
    global _ai_failed
    if _ai_failed or not AI_INPAINT or not hole.any():
        return None
    s = _ort("migan_pipeline_v2.onnx")
    if s is None:
        _ai_failed = True
        return None
    try:
        ins = s.get_inputs()
        img_in, mask_in = (ins[0], ins[1]) if "mask" not in ins[0].name.lower() else (ins[1], ins[0])
        mask = np.where(hole, 0, 255).astype(np.uint8)
        nchw = len(img_in.shape) == 4 and img_in.shape[1] == 3
        if nchw:
            feed = {img_in.name: rgb.transpose(2, 0, 1)[None].copy(), mask_in.name: mask[None, None]}
        else:
            feed = {img_in.name: rgb[None].copy(), mask_in.name: mask[None, ..., None]}
        out = np.asarray(s.run(None, feed)[0])[0]
        if out.shape[0] == 3 and out.shape[-1] != 3:
            out = out.transpose(1, 2, 0)
        if out.shape[:2] != rgb.shape[:2]:
            out = cv2.resize(out.astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        if out.dtype != np.uint8:
            out = np.clip(out if out.max() > 1.5 else out * 255, 0, 255).astype(np.uint8)
        return np.where(hole[..., None], out, rgb)
    except Exception as e:
        _log(f"  AI inpainting unavailable ({e}); using OpenCV inpainting")
        _ai_failed = True
        return None


# --------------------------------------------------------------------------- pose


POSE_NAMES = [
    "nose", "eye_in_l", "eye_l", "eye_out_l", "eye_in_r", "eye_r", "eye_out_r", "ear_l", "ear_r",
    "mouth_l", "mouth_r", "shoulder_l", "shoulder_r", "elbow_l", "elbow_r", "wrist_l", "wrist_r",
    "pinky_l", "pinky_r", "index_l", "index_r", "thumb_l", "thumb_r", "hip_l", "hip_r",
    "knee_l", "knee_r", "ankle_l", "ankle_r", "heel_l", "heel_r", "foot_l", "foot_r",
]


def detect_pose(rgb: np.ndarray, mask: np.ndarray) -> dict | None:
    """Body keypoints {name: (x, y, visibility)} in image pixels, or None.

    The character is cut out and centred on a grey square first; MediaPipe
    finds drawn characters far more reliably that way.  "_l" is the
    character's left, matching Live2D naming.
    """
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
    except Exception:
        return None
    p = fetch("pose_landmarker_heavy.task")
    if not p:
        return None
    try:
        det = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=p), min_pose_detection_confidence=0.2)
        )
    except Exception as e:
        _log(f"  pose model unavailable: {e}")
        return None
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    S = int(max(y1 - y0, x1 - x0) * 1.1)
    oy, ox = (S - (y1 - y0)) // 2, (S - (x1 - x0)) // 2
    sq = np.full((S, S, 3), 128, np.uint8)
    crop = rgb[y0:y1, x0:x1]
    m = mask[y0:y1, x0:x1, None]
    sq[oy : oy + y1 - y0, ox : ox + x1 - x0] = np.where(m, crop, 128)
    try:
        res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(sq)))
    finally:
        det.close()
    if not res.pose_landmarks:
        return None
    out = {}
    for name, lm in zip(POSE_NAMES, res.pose_landmarks[0]):
        out[name] = (lm.x * S - ox + x0, lm.y * S - oy + y0, float(lm.visibility))
    return out


# --------------------------------------------------------------------------- anime faces


def detect_faces(rgb: np.ndarray, mask: np.ndarray | None = None) -> list[tuple[float, float, float, float, float]]:
    """Anime face boxes (x, y, w, h, angle_degrees), biggest first."""
    p = fetch("lbpcascade_animeface.xml")
    if not p:
        return []
    cas = cv2.CascadeClassifier(p)
    img = rgb.copy()
    if mask is not None:
        img[~mask] = 255
    g = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    h, w = g.shape
    scale = 1.0
    if max(h, w) > 1600:
        scale = 1600 / max(h, w)
        g = cv2.resize(g, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    found = []
    for ang in (0, -15, 15, -30, 30):
        M = cv2.getRotationMatrix2D((g.shape[1] / 2, g.shape[0] / 2), ang, 1.0)
        rg = cv2.warpAffine(g, M, (g.shape[1], g.shape[0]), borderValue=255) if ang else g
        faces = cas.detectMultiScale(rg, scaleFactor=1.05, minNeighbors=4, minSize=(24, 24))
        Mi = cv2.invertAffineTransform(M)
        for x, y, fw, fh in faces:
            cx, cy = x + fw / 2, y + fh / 2
            ox, oy = Mi @ np.array([cx, cy, 1.0])
            found.append((ox / scale - fw / scale / 2, oy / scale - fh / scale / 2, fw / scale, fh / scale, float(ang)))
        if found:
            break
    found.sort(key=lambda f: -f[2] * f[3])
    return found


# --------------------------------------------------------------------------- SAM


class MobileSAM:
    """Segment-anything with the MobileSAM ONNX export (encoder + batched decoder)."""

    def __init__(self, enc, dec):
        self.enc, self.dec = enc, dec

    @staticmethod
    def load():
        try:
            import onnxruntime as ort
        except ImportError:
            return None
        e, d = _path("mobilesam_encoder.onnx"), _path("mobilesam_decoder.onnx")
        if not (e and d):
            return None
        opt = ort.SessionOptions()
        return MobileSAM(
            ort.InferenceSession(e, opt, providers=["CPUExecutionProvider"]),
            ort.InferenceSession(d, opt, providers=["CPUExecutionProvider"]),
        )

    def embed(self, rgb: np.ndarray):
        h, w = rgb.shape[:2]
        s = 1024 / max(h, w)
        nh, nw = int(round(h * s)), int(round(w * s))
        x = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32)
        x = (x - np.array([123.675, 116.28, 103.53], np.float32)) / np.array([58.395, 57.12, 57.375], np.float32)
        pad = np.zeros((1024, 1024, 3), np.float32)
        pad[:nh, :nw] = x
        emb = self.enc.run(None, {"image": pad.transpose(2, 0, 1)[None]})[0]
        return emb, s, (h, w)

    def predict(self, emb_info, points: np.ndarray, labels: np.ndarray | None = None):
        """points: (B, N, 2) image pixels -> masks (B, 3, h, w) logits at 256 res, scores (B, 3)."""
        emb, s, _ = emb_info
        pts = (points * s).astype(np.float32)
        if labels is None:
            labels = np.ones(pts.shape[:2], np.float32)
        low, iou = self.dec.run(None, {"embeddings": emb, "points": pts, "labels": labels.astype(np.float32)})
        return low, iou

    def upscale(self, emb_info, low: np.ndarray) -> np.ndarray:
        """Low-res logits (256x256 over the padded 1024 square) -> logits at image size."""
        _, s, (h, w) = emb_info
        up = cv2.resize(low, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        up = up[: int(round(h * s)), : int(round(w * s))]
        return cv2.resize(up, (w, h), interpolation=cv2.INTER_LINEAR)

    def everything(self, rgb: np.ndarray, mask: np.ndarray, grid: int = 32, batch: int = 64, log=None):
        """Automatic mask generation restricted to ``mask``. Returns list of (bool mask, score)."""
        emb = self.embed(rgb)
        h, w = rgb.shape[:2]
        ys, xs = np.nonzero(mask)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        gx = np.linspace(x0, x1, grid + 2)[1:-1]
        gy = np.linspace(y0, y1, int(grid * (y1 - y0) / max(1, x1 - x0)) + 2)[1:-1]
        P = np.array([(x, y) for y in gy for x in gx if mask[int(y), int(x)]], np.float32)
        cands = []
        for i in range(0, len(P), batch):
            pts = P[i : i + batch, None, :]
            low, iou = self.predict(emb, pts)
            for b in range(len(pts)):
                for k in range(low.shape[1]):
                    if iou[b, k] < 0.75:
                        continue
                    lg = low[b, k]
                    # stability: mask barely changes when the threshold moves
                    hi, lo = (lg > 1.0).sum(), (lg > -1.0).sum()
                    if lo == 0 or hi / lo < 0.85:
                        continue
                    cands.append((float(iou[b, k]), b + i, k, lg))
        # non-maximum suppression at low resolution, then upscale survivors
        cands.sort(key=lambda c: -c[0])
        kept, kept_low = [], []
        for sc, _, _, lg in cands:
            m = lg > 0
            area = m.sum()
            if area < 20:
                continue
            dup = False
            for km in kept_low:
                inter = (m & km).sum()
                if inter / (area + km.sum() - inter) > 0.7:
                    dup = True
                    break
            if dup:
                continue
            kept_low.append(m)
            kept.append((sc, lg))
        out = []
        for sc, lg in kept:
            full = self.upscale(emb, lg) > 0
            full &= mask
            if full.sum() > 30:
                out.append((full, sc))
        return out
