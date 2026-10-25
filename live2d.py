#!/usr/bin/env python3
"""Picture -> animated Live2D-style puppet, shown in a preview window.

    python live2d.py picture.png              # convert (first time) and open the preview
    python live2d.py picture.png --follow     # start in mouse-follow mode
    python live2d.py picture.png --redo       # re-detect the parts
    python live2d.py picture.png --record out.mp4 --seconds 10
    python live2d.py edit picture_cubism.l2d  # fix parts in the browser editor
    python live2d.py convert picture.png      # PSD only (for Live2D Cubism Editor)

Preview keys: M idle/mouse-follow · Q/W wind -/+ · P physics · B background ·
Space pause · C screenshot · R record · +/- zoom · Esc quit.
"""
import sys

from live2d_psd.__main__ import main

if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "convert":
        main(args[1:])
    elif args and args[0] in ("edit", "build", "animate"):
        main(args)
    else:
        main(["animate"] + args)
