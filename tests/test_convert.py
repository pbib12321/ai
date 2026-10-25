"""End-to-end check on a procedurally drawn cartoon character."""
import numpy as np
import pytest
from PIL import Image, ImageDraw

from live2d_psd import convert

psd_tools = pytest.importorskip("psd_tools")

SKIN = (240, 200, 160)
SHIRT = (200, 60, 60)
PANTS = (60, 80, 160)
INK = (0, 0, 0)


def draw_character(path):
    im = Image.new("RGBA", (500, 900), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    W = 6
    # legs, body, arms, neck, head, face features (back to front)
    d.rectangle((185, 600, 235, 860), fill=SKIN, outline=INK, width=W)
    d.rectangle((265, 600, 315, 860), fill=SKIN, outline=INK, width=W)
    d.rectangle((170, 520, 330, 640), fill=PANTS, outline=INK, width=W)
    d.rectangle((225, 300, 275, 360), fill=SKIN, outline=INK, width=W)
    d.rectangle((170, 340, 330, 540), fill=SHIRT, outline=INK, width=W)
    d.rectangle((120, 350, 165, 560), fill=SKIN, outline=INK, width=W)
    d.rectangle((335, 350, 380, 560), fill=SKIN, outline=INK, width=W)
    d.ellipse((110, 40, 390, 320), fill=SKIN, outline=INK, width=W)
    for cx in (195, 305):
        d.ellipse((cx - 40, 120, cx + 40, 200), fill=(255, 255, 255), outline=INK, width=W)
        d.ellipse((cx - 14, 146, cx + 14, 174), fill=INK)
        d.rounded_rectangle((cx - 35, 90, cx + 35, 104), radius=7, fill=INK)
    d.ellipse((210, 240, 290, 275), fill=(220, 90, 110), outline=INK, width=W)
    im.save(path)


def test_convert(tmp_path):
    src = tmp_path / "char.png"
    out = tmp_path / "char.psd"
    draw_character(src)
    seg, parts, rendered, merged = convert(str(src), str(out), save_parts_file=str(tmp_path / "p.json"), log=lambda *a: None)

    names = {p.name for p in parts}
    for expected in ("Face", "Eye White L", "Eye White R", "Pupil L", "Pupil R", "Brow L", "Brow R", "Mouth",
                     "Arm L", "Arm R", "Leg L", "Leg R", "Top", "Bottom"):
        assert expected in names, names

    psd = psd_tools.PSDImage.open(out)
    assert psd.size == (500, 900)
    groups = {l.name for l in psd if l.is_group()}
    assert {"Head", "Body"} <= groups
    layer_names = {l.name for l in psd.descendants() if not l.is_group()}
    assert {"Eye Line L", "Eye Line R", "Face"} <= layer_names

    # the layered result must look like the input
    comp = np.asarray(psd.composite(force=True).convert("RGBA"), np.float32)
    ref = np.asarray(Image.open(src).convert("RGBA"), np.float32)

    def flat(a):
        return a[..., :3] * a[..., 3:] / 255 + 255 * (1 - a[..., 3:] / 255)

    diff = np.abs(flat(comp) - flat(ref)).mean(-1)
    assert diff.mean() < 4, diff.mean()

    # the face is complete underneath the eyes
    face = next(l for l in psd.descendants() if l.name == "Face")
    fa = np.asarray(face.topil())[..., 3]
    x0, y0 = face.left, face.top
    assert fa[160 - y0, 195 - x0] == 255 and fa[160 - y0, 305 - x0] == 255


def test_project_rebuild_and_folders(tmp_path):
    """The saved project rebuilds the same PSD; folders keep the draw order."""
    from live2d_psd import build_project
    from live2d_psd.psd import build_tree

    src = tmp_path / "char.png"
    draw_character(src)
    convert(str(src), str(tmp_path / "a.psd"), mode="cartoon", project=str(tmp_path / "a.l2d"), log=lambda *a: None)
    build_project(str(tmp_path / "a.l2d"), str(tmp_path / "b.psd"), log=lambda *a: None)
    a = np.asarray(psd_tools.PSDImage.open(tmp_path / "a.psd").composite(force=True).convert("RGBA"), np.float32)
    b = np.asarray(psd_tools.PSDImage.open(tmp_path / "b.psd").composite(force=True).convert("RGBA"), np.float32)
    assert np.abs(a - b).mean() < 0.5

    class L:
        def __init__(self, name, path, z):
            self.name, self.path, self.z, self.rgba, self.left, self.top = name, path, z, np.zeros((1, 1, 4), np.uint8), 0, 0

    # an arm between two head layers splits the Head folder instead of reordering
    tree = build_tree([L("Hair", ["Head"], 90), L("Hand", ["Body", "Arm"], 85), L("Face", ["Head"], 80)])
    assert [c.name for c in tree.children] == ["Head", "Body", "Head"]
