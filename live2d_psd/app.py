"""One-stop: picture -> layers -> rig -> animated preview window (or video)."""
from __future__ import annotations

import os

import numpy as np
from PIL import Image


def animate(src: str, video: str | None = None, seconds: float = 8.0, mode: str = "idle", redo: bool = False,
            wind: float = 0.8, scenery: bool = True, convert_mode: str = "auto", log=print):
    from . import build_project, convert
    from .project import load_project
    from .puppet import build_rig
    from .scenery import build_scenery

    base = os.path.splitext(src)[0]
    if base.endswith("_cubism"):
        base = base[: -len("_cubism")]
    project = base + "_cubism.l2d"
    psd = base + "_cubism.psd"
    if redo or not os.path.isdir(project):
        convert(src, psd, mode=convert_mode, background="drop", project=project, preview=base + "_cubism.preview.png",
                save_parts_file=base + "_cubism.parts.json", log=log)
    else:
        log(f"using the saved project {project} (pass --redo to re-detect parts)")
    rendered = build_project(project, psd, log=log)
    proj, labels, alpha = load_project(project)
    rgba = np.array(Image.open(proj["source"]).convert("RGBA"))
    had_bg = (rgba[..., 3] < 128).mean() < 0.01
    layers = [r for r in rendered if r.name != "Background"]
    if had_bg:
        log("building the animated background")
        layers += build_scenery(rgba[..., :3], alpha, animate=scenery, log=log)
    H, W = labels.shape
    creature = not any(r.name.startswith(("Upper Body", "Top", "Hair")) for r in layers)
    rig = build_rig(layers, W, H, creature=creature)
    n_phys = sum(m.chain is not None for m in rig.meshes)
    log(f"rig: {len(rig.meshes)} meshes, {n_phys} with physics, eyes: {', '.join(sorted(rig.eyes)) or 'none'}")
    if video:
        from .viewer import record

        log(f"recording {seconds:.0f}s ({mode}) to {video}")
        record(rig, video, seconds=seconds, mode=mode, wind=wind)
        log(f"wrote {video}")
    else:
        from .viewer import run_window

        run_window(rig, title=os.path.basename(base))
    return rig
