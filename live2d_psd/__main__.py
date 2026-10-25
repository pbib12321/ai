import argparse
import os
import sys

from . import convert


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("edit", "build"):
        return _sub(argv)
    ap = argparse.ArgumentParser(
        prog="python -m live2d_psd",
        description="Turn a flat, outlined 2D character image into a layered PSD ready for Live2D Cubism rigging.",
    )
    ap.add_argument("image", help="input image (PNG/WebP/JPG; transparent background preferred)")
    ap.add_argument("-o", "--out", help="output .psd (default: <image>_cubism.psd)")
    ap.add_argument("--mode", choices=["auto", "cartoon", "anime"], default="auto",
                    help="cartoon: flat colours + outlines, redrawn clean; anime: shaded/painted art, keeps the original pixels (default: auto)")
    ap.add_argument("--background", choices=["auto", "keep", "drop"], default="auto",
                    help="add a filled-in background layer when the input had a background (anime mode, default auto = yes)")
    ap.add_argument("--scale", type=float, default=1.0, help="render the PSD this many times larger (default 1)")
    ap.add_argument("--parts", help="use this parts.json instead of automatic part detection")
    ap.add_argument("--no-preview", action="store_true", help="don't write the preview PNG")
    ap.add_argument("--sweep", type=float, default=1.0, help="how far hidden parts continue under others (0 = off)")
    ap.add_argument("--smooth", type=float, default=0.9, help="edge smoothing radius in input pixels")
    ap.add_argument("--merge-colors", type=float, default=10.0, help="colours closer than this (CIE dE) are one colour")
    a = ap.parse_args(argv)

    base = os.path.splitext(a.out or a.image)[0]
    if not a.out:
        base += "_cubism"
    out = a.out or base + ".psd"
    convert(
        a.image,
        out,
        scale=a.scale,
        parts_file=a.parts,
        save_parts_file=base + ".parts.json",
        preview=None if a.no_preview else base + ".preview.png",
        sweep=a.sweep,
        smooth=a.smooth,
        merge_de=a.merge_colors,
        mode="cartoon" if a.parts else a.mode,
        background=a.background,
        project=base + ".l2d",
    )


def _sub(argv):
    from . import build_project

    ap = argparse.ArgumentParser(prog=f"python -m live2d_psd {argv[0]}")
    ap.add_argument("target", help="project folder (*.l2d), or an image to convert first")
    ap.add_argument("-o", "--out", help="PSD to write (default: next to the project)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv[1:])
    target = a.target
    if not os.path.isdir(target):
        base = os.path.splitext(target)[0]
        if base.endswith("_cubism"):
            target = base + ".l2d"
        elif os.path.isdir(base + "_cubism.l2d"):
            target = base + "_cubism.l2d"
        else:
            main([a.target])
            target = base + "_cubism.l2d"
    out = a.out or target[: -len(".l2d")] + ".psd"
    if argv[0] == "build":
        build_project(target, out)
    else:
        from .editor import serve

        serve(target, out, port=a.port, open_browser=not a.no_browser)


if __name__ == "__main__":
    main()
