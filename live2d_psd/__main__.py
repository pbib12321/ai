import argparse
import os

from . import convert


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m live2d_psd",
        description="Turn a flat, outlined 2D character image into a layered PSD ready for Live2D Cubism rigging.",
    )
    ap.add_argument("image", help="input image (PNG/WebP/JPG; transparent background preferred)")
    ap.add_argument("-o", "--out", help="output .psd (default: <image>_cubism.psd)")
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
    )


if __name__ == "__main__":
    main()
