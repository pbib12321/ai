"""Run the converter on every picture in a folder and make one comparison sheet.

    python tools/benchmark.py my_test_pictures/ results/

Keep your tricky pictures in a folder and run this after every change: each
picture's preview sheet is written to the results folder, plus
``overview.png`` with all part maps side by side and ``report.txt`` with the
parts found and the time taken.
"""
import glob
import os
import sys
import time

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from live2d_psd import convert  # noqa: E402


def main(src, out):
    os.makedirs(out, exist_ok=True)
    files = sorted(f for f in glob.glob(os.path.join(src, "*")) if os.path.splitext(f)[1].lower() in
                   (".png", ".jpg", ".jpeg", ".webp", ".avif", ".bmp"))
    lines, thumbs = [], []
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        t = time.time()
        log = []
        try:
            _, parts, _, _ = convert(f, os.path.join(out, name + ".psd"), preview=os.path.join(out, name + ".preview.png"),
                                     log=log.append)
            names = sorted({p.name for p in parts})
            lines.append(f"{name}: {len(names)} parts in {time.time() - t:.0f}s: {', '.join(names)}")
            im = Image.open(os.path.join(out, name + ".preview.png"))
            im = im.crop((0, 0, im.width * 2 // 5, im.height))
            im.thumbnail((700, 700))
            thumbs.append(im)
        except Exception as e:  # keep going, report it
            lines.append(f"{name}: FAILED ({e})")
        print(lines[-1], flush=True)
    with open(os.path.join(out, "report.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    if thumbs:
        cols = 3
        w = max(t.width for t in thumbs)
        h = max(t.height for t in thumbs)
        rows = (len(thumbs) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * w, rows * h), "white")
        for i, t in enumerate(thumbs):
            sheet.paste(t.convert("RGB"), ((i % cols) * w, (i // cols) * h))
        sheet.save(os.path.join(out, "overview.png"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
