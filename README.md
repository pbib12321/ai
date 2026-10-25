# live2d_psd

Turns a flat, outlined 2D character picture (cartoon style) into a layered PSD
that is ready to import into **Live2D Cubism Editor**, and that opens cleanly in
Photopea or Photoshop.

```
pip install -r requirements.txt
python -m live2d_psd character.png                 # -> character_cubism.psd
python -m live2d_psd character.png --scale 2 -o big.psd
```

Each run writes three files:

| file | what it is |
| --- | --- |
| `*_cubism.psd` | the layered PSD |
| `*_cubism.parts.json` | which colour regions make up each part, plus names, folders and draw order |
| `*_cubism.preview.png` | numbered region map, the re-assembled image and every layer on its own |

## What the PSD contains

The layout follows the official Live2D sample models (mark, natori,
tororo/hijiki, wanko). Every piece is cut out separately and carries **its
own outline**. Areas that are hidden behind other pieces are **painted in**,
so nothing tears open when the rig moves:

```
Head/
  Brow R, Brow L
  Eye R/  Eye Line R (ring + lashes), Pupil R, Eye White R
  Eye L/  Eye Line L, Pupil L, Eye White L
  Mouth
  Face Details   (whiskers, cheek marks: strokes floating on the face)
  Face           (complete under the eyes, brows and mouth)
  Ear R, Ear L   (continue underneath the face)
Body/
  Arm L/  Sleeve L, Arm L           (front or back of the body, detected)
  Accessory 1                       (badges and pendants sitting on clothes)
  Top                               (continues under arms and accessories)
  Bottom                            (continues up under the top)
  Neck                              (continues up under the head, down into the collar)
  Leg R/, Leg L/                    (continue up under the bottom; both keep the stroke between them)
  Arm R/  Sleeve R, Arm R
  Tail
```

As in the Live2D samples, **L and R mean the character's left and right**, so
"Eye L" is the eye on the right-hand side of the picture.

The layers are plain RGB pixel layers in nested group folders. There are no
clipping masks, layer masks or effects, because Cubism can't import them.
Set up clipping, such as the pupil inside the eye white, in Cubism after
import.

## How it works

1. **Segment** (`segment.py`): removes the background if the image has none,
   separates the ink from the colour areas and measures the stroke width. It
   tells solid dark shapes (pupils, brows) apart from outlines, splits the
   colour areas into flat-colour regions, and cuts regions at narrow necks
   where there's no line (for example where the head flows into the neck).
   Small pockets that are the same surface, like face colour trapped between
   eyelashes, are merged back.
2. **Name the parts** (`parts.py`): finds the head, and the eyes, pupils,
   brows and mouth enclosed by it. Then ears, neck, legs, arms (hanging from
   sleeves, or long limbs at the sides), tail, garments, sleeves and
   accessories. Shading regions join their base colour. Draw order comes
   from Live2D conventions plus a T-junction test that decides whether each
   arm is in front of or behind the clothes.
3. **Complete the pieces** (`complete.py`): each outline stroke goes to the
   part in front, while the part behind continues underneath. Holes left by
   features are filled. Where a piece goes under another (legs under a skirt,
   neck under the head, ears behind the face), its cross-section and side
   outlines are swept onward underneath. The sweep is capped and clipped, so
   it can never show in the assembled picture.
4. **Render clean** (`render.py`): every colour area and stroke is redrawn
   from a smoothed high-resolution mask, which is the same as tracing to
   vectors and rasterising again. This removes JPEG noise and stair-steps,
   and `--scale` renders larger than the input.
5. **Write the PSD** (`psd.py`): a small layered-PSD writer with groups and
   Unicode names.

## Fixing the automatic result

The heuristics are built for a standing character drawn with dark outlines
and flat colours. When a part is wrong, open the preview, find the region
numbers, edit `*.parts.json` and run again with it:

```
python -m live2d_psd character.png --parts character_cubism.parts.json
```

In `parts.json` you can move region ids between parts, rename parts, change
`path` (folders), change `z` (bigger = closer to the viewer), split
`regions` into new parts, or set `"sweep": false` to stop a part continuing
underneath others. Region numbers stay the same as long as the input image
and the `--merge-colors` setting don't change.

## Limits

- Made for cel-shaded, outlined art. Painterly or soft-shaded images with no
  ink lines won't split into parts well.
- Shapes that aren't drawn at all in the input, such as the back of a
  sleeve, can't be invented. Hidden areas are extended from what's visible.
- Arms that are drawn fused into the torso with no line between them stay
  one region. Split them with `parts.json` or draw the missing line first.

## Tests

```
python -m pytest tests
```

The test draws a simple character, converts it, re-opens the PSD with
`psd-tools`, and checks the part names, the folders, that the composite
matches the input, and that the face is complete underneath the eyes.
