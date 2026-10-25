"""Convert a 2D character illustration into a layered, Live2D-ready PSD."""
from __future__ import annotations

import numpy as np

from .complete import build_layers
from .parts import auto_parts, load_parts, save_parts
from .psd import write_psd
from .render import composite, render_layer
from .segment import load_rgba, segment

__all__ = ["convert", "build_project"]


def convert(
    src: str,
    out: str,
    *,
    mode: str = "auto",
    scale: float = 1.0,
    parts_file: str | None = None,
    save_parts_file: str | None = None,
    preview: str | None = None,
    sweep: float = 1.0,
    smooth: float = 0.9,
    merge_de: float = 10.0,
    background: str = "auto",
    project: str | None = None,
    log=print,
):
    """mode: auto | cartoon (flat colours + outlines, redrawn clean) | anime (shaded/textured, original pixels)."""
    from . import models

    models.set_logger(log)
    if mode not in ("auto", "cartoon", "anime"):
        raise ValueError(f"unknown mode {mode}")

    sc = None
    if mode in ("auto", "anime") and not parts_file:
        from .semantic import detect, load_scene

        sc = load_scene(src, log)
        if mode == "auto":
            detect(sc, log)
            if (sc.face is not None or sc.pose is not None) and not _is_flat_cartoon(sc):
                mode = "anime"
            elif _is_flat_cartoon(sc) and _cartoon_rules_fit(sc):
                mode = "cartoon"
            else:
                mode = "anime"  # the region/SAM parser, with creature mode for animals and mascots
            log(f"  using the {mode} pipeline")
        else:
            detect(sc, log)

    if mode == "anime":
        return _convert_anime(sc, src, out, scale, preview, sweep, background, log, project)
    return _convert_cartoon(src, sc, out, scale, parts_file, save_parts_file, preview, sweep, smooth, merge_de, log, project)


def _is_flat_cartoon(sc) -> bool:
    """Few flat colours separated by dark outlines (Gumball, Rick and Morty...)."""
    lab = sc.lab[sc.fg & (sc.alpha > 0.98)]
    if len(lab) < 100:
        return False
    q = np.round(lab[:: max(1, len(lab) // 200000)] / 6).astype(np.int32)
    _, counts = np.unique(q, axis=0, return_counts=True)
    counts = np.sort(counts)[::-1]
    coverage = counts[:24].sum() / counts.sum()
    return coverage > 0.82


def _cartoon_rules_fit(sc) -> bool:
    """Do the outline rules recognise this character (a face with pupils in its eyes)?"""
    rgba = np.dstack([sc.rgb, np.where(sc.alpha >= 0.5, 255, 0).astype(np.uint8)])
    try:
        parts = auto_parts(segment(rgba))
    except Exception:
        return False
    names = {p.name for p in parts}
    eyes_ok = all(f"{k} {s}" in names for k in ("Pupil", "Eye White") for s in ("L", "R"))
    unexplained = sum(n.startswith(("Face Feature", "Piece")) for n in names)
    return "Face" in names and eyes_ok and unexplained <= 3


def _convert_cartoon(src, sc, out, scale, parts_file, save_parts_file, preview, sweep, smooth, merge_de, log, project=None):
    if sc is not None:
        rgba = np.dstack([sc.rgb, (sc.alpha * 255).astype(np.uint8)])
        rgba[sc.alpha < 0.5, 3] = 0
        rgba[sc.alpha >= 0.5, 3] = 255
    else:
        rgba = load_rgba(src)
    h, w = rgba.shape[:2]
    log(f"segmenting {w}x{h} image")
    seg = segment(rgba, merge_de=merge_de)
    log(f"  {len(seg.regions)} regions, outline width ~{seg.line_halfwidth * 2:.0f}px")
    parts = load_parts(parts_file) if parts_file else auto_parts(seg, merge_de=merge_de)
    missing = set(seg.regions) - {r for p in parts for r in p.regions}
    if missing:
        log(f"  warning: regions not assigned to any part: {sorted(missing)}")
    if save_parts_file:
        save_parts(parts, save_parts_file)
    log(f"  {len(parts)} parts: " + ", ".join(p.name for p in sorted(parts, key=lambda p: -p.z)))
    log("completing hidden areas and outlines")
    pieces = build_layers(seg, parts, sweep=sweep)
    log(f"rendering {len(pieces)} layers at {scale}x")
    rendered = [r for r in (render_layer(p, scale=scale, sigma=smooth) for p in pieces) if r is not None]
    W, H = int(round(w * scale)), int(round(h * scale))
    merged = composite(rendered, W, H)
    write_psd(out, W, H, rendered, merged)
    log(f"wrote {out}")
    if project:
        from .project import meta_of, save_project

        labels = np.zeros((h, w), np.uint16)
        metas = []
        for i, p in enumerate(sorted(parts, key=lambda p: p.z), start=1):
            labels[p.mask if p.mask is not None else np.isin(seg.labels, p.regions)] = i
            metas.append(meta_of(p, i))
        save_project(project, src, "cartoon", metas, labels, rgba[..., 3] / 255.0, {"scale": scale, "sweep": sweep, "smooth": smooth, "merge_de": merge_de})
        log(f"wrote {project}  (fix parts with: python -m live2d_psd edit {project})")
    if preview:
        from .preview import save_preview

        save_preview(preview, seg, parts, rendered, merged)
        log(f"wrote {preview}")
    return seg, parts, rendered, merged


def _convert_anime(sc, src, out, scale, preview, sweep, background, log, project=None):
    from .semantic import parse
    from .texture import build_texture_layers

    H, W = sc.fg.shape
    log(f"parsing the character ({W}x{H})")
    parts = parse(sc, log)
    log(f"  {len(parts)} parts: " + ", ".join(p.name for p in sorted(parts, key=lambda p: -p.z)))
    log("completing hidden areas")
    keep_bg = background == "keep" or (background == "auto" and sc.had_background)
    rendered = build_texture_layers(sc, parts, sweep=sweep, background=keep_bg, log=log, carry=_is_flat_cartoon(sc))
    if scale != 1.0:
        rendered = [_rescale(r, scale) for r in rendered]
    W2, H2 = int(round(W * scale)), int(round(H * scale))
    merged = composite(rendered, W2, H2)
    write_psd(out, W2, H2, rendered, merged)
    log(f"wrote {out}")
    if project:
        from .project import meta_of, save_project

        labels = np.zeros((H, W), np.uint16)
        metas = []
        for i, p in enumerate(sorted(parts, key=lambda p: p.z), start=1):
            labels[p.mask] = i
            metas.append(meta_of(p, i))
        save_project(project, src, "anime", metas, labels, sc.alpha, {"scale": scale, "sweep": sweep, "background": background, "had_background": bool(sc.had_background)})
        log(f"wrote {project}  (fix parts with: python -m live2d_psd edit {project})")
    if preview:
        from .preview import save_semantic_preview

        save_semantic_preview(preview, sc, parts, rendered, merged)
        log(f"wrote {preview}")
    return sc, parts, rendered, merged


def _rescale(r, s):
    import cv2

    from .render import RenderedLayer

    h, w = r.rgba.shape[:2]
    img = cv2.resize(r.rgba, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=cv2.INTER_LANCZOS4 if s > 1 else cv2.INTER_AREA)
    return RenderedLayer(r.name, r.path, r.z, img, int(round(r.left * s)), int(round(r.top * s)))


def build_project(folder: str, out: str, log=print):
    """Rebuild the PSD from a (possibly hand-edited) project folder."""
    from PIL import Image

    from .project import load_project

    proj, labels, alpha = load_project(folder)
    st = proj.get("settings", {})
    scale = st.get("scale", 1.0)
    sweep = st.get("sweep", 1.0)
    rgb = np.array(Image.open(proj["source"]).convert("RGB"))
    H, W = labels.shape
    metas = proj["parts"]
    if proj["mode"] == "anime":
        from skimage.color import rgb2lab

        from .semantic import Scene, SemPart
        from .texture import build_texture_layers

        sc = Scene(rgb=rgb, alpha=alpha, fg=(alpha > 0.5) | (labels > 0), had_background=bool(st.get("had_background", False)))
        sc.lab = rgb2lab(rgb.astype(np.float32) / 255)
        parts = []
        for m in metas:
            mask = labels == m["id"]
            if mask.any():
                parts.append(SemPart(m["name"], m["path"], m["z"], mask, sweep=m.get("sweep", True), fill=m.get("fill", "inpaint"),
                                     hidden_hint=mask if m["name"].startswith("Eye White") else None))
        keep_bg = st.get("background", "auto") != "drop" and sc.had_background
        rendered = build_texture_layers(sc, parts, sweep=sweep, background=keep_bg, log=log, carry=_is_flat_cartoon(sc))
        if scale != 1.0:
            rendered = [_rescale(r, scale) for r in rendered]
    else:
        from .parts import Part

        rgba = np.dstack([rgb, np.where(alpha >= 0.5, 255, 0).astype(np.uint8)])
        seg = segment(rgba, merge_de=st.get("merge_de", 10.0))
        parts = []
        for m in metas:
            mask = labels == m["id"]
            if mask.any():
                parts.append(Part(m["name"], m["path"], [], m["z"], kind=m.get("kind", "fill"), split_lines=m.get("split_lines", False),
                                  sweep=m.get("sweep", True), line_layer=m.get("line_layer"), mask=mask))
        pieces = build_layers(seg, parts, sweep=sweep)
        rendered = [r for r in (render_layer(p, scale=scale, sigma=st.get("smooth", 0.9)) for p in pieces) if r is not None]
    W2, H2 = int(round(W * scale)), int(round(H * scale))
    merged = composite(rendered, W2, H2)
    write_psd(out, W2, H2, rendered, merged)
    log(f"wrote {out} ({len(rendered)} layers)")
    return rendered
