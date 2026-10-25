"""Convert a flat 2D character illustration into a layered, Live2D-ready PSD."""
from __future__ import annotations

from .complete import build_layers
from .parts import auto_parts, load_parts, save_parts
from .psd import write_psd
from .render import composite, render_layer
from .segment import load_rgba, segment

__all__ = ["convert"]


def convert(
    src: str,
    out: str,
    *,
    scale: float = 1.0,
    parts_file: str | None = None,
    save_parts_file: str | None = None,
    preview: str | None = None,
    sweep: float = 1.0,
    smooth: float = 0.9,
    merge_de: float = 10.0,
    log=print,
):
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
    if preview:
        from .preview import save_preview

        save_preview(preview, seg, parts, rendered, merged)
        log(f"wrote {preview}")
    return seg, parts, rendered, merged
