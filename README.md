# live2d_psd

Turns a 2D character picture into a layered PSD that is ready to rig in
**Live2D Cubism Editor**, and that opens cleanly in Photopea or Photoshop.

It handles two kinds of art, picked automatically:

| | **cartoon** mode | **anime** mode |
| --- | --- | --- |
| for | flat colours + dark outlines (Gumball, Rick and Morty, …) | anime, cel-shaded, painted or textured art |
| pixels | redrawn clean: flat colours, smooth outlines, any `--scale` | the original pixels: shading, texture and soft hair edges kept |
| parts from | colour regions + outline rules | face detection, body keypoints, MobileSAM, colour models |
| background | expects a plain or transparent one | removed with ISNet; optionally kept as a filled-in layer |

```
pip install -r requirements.txt
python -m live2d_psd character.png                 # -> character_cubism.psd (+ preview, + editable project)
python -m live2d_psd character.png --mode anime --background drop
python -m live2d_psd edit character_cubism.l2d     # fix parts in your browser, then "Export PSD"
python -m live2d_psd build character_cubism.l2d    # rebuild the PSD from a saved project
```

Each conversion writes:

| file | what it is |
| --- | --- |
| `*_cubism.psd` | the layered PSD |
| `*_cubism.preview.png` | the part map, the re-assembled image and every layer |
| `*_cubism.l2d/` | the editable project (part map + part list) for `edit` / `build` |
| `*_cubism.parts.json` | cartoon mode only: colour-region ids per part (older `--parts` workflow) |

## What the PSD contains

Every piece is cut out separately, and the areas hidden behind other pieces
are painted in, so nothing tears open when the rig moves:

- **The face** continues under the bangs, eyes and mouth.
- **The eye white** continues under the iris, which slides under the eye line.
- **Hair** continues behind the face.
- **Limbs** continue under sleeves and skirts, the **neck** goes under the
  head, and the **body** goes under the arms.

Anime-mode layout, following the official Live2D sample models (natori, mark):

```
Head/        Hair Front, Brow L/R, Eye L/ (Eye Line, Iris, Eye White), Eye R/ (…), Mouth, Nose, Face, Hair Side L/R
Props/       Prop 1, …                       staffs, swords, papers held in a hand
Body/        Arm L/ (Hand, Forearm, Upper Arm), Upper Body, Lower Body, Neck, Leg L/ …, Arm R/ …
Hair Back/   Hair Back
Background/  Background                      only when the input had one (--background drop to skip)
```

The cartoon-mode layout is the same idea, with Eye White / Pupil / Eye Line,
Sleeves, Top / Bottom, Tail and Ears.

**L and R mean the character's left and right**, as in Live2D, so "Eye L" is
the eye on the right-hand side of the picture. Layers are plain RGB pixel
layers in nested folders. There are no clipping masks, layer masks or
effects, because Cubism can't import them. Set up clipping (iris inside eye
white) in Cubism after import.

## The editor

`python -m live2d_psd edit <project>` opens a page in your browser, served
from your own computer:

- **AI select (A)**: click on a part of the picture and MobileSAM selects that
  object. Right-click removes an area, Enter gives the selection to the part
  highlighted in the list.
- **Wand (W)**: selects the connected colour area under the click.
- **Brush (B)** and **Eraser (E)**: paint by hand. **Pick (I)**: click a part
  in the picture to select it in the list.
- **Undo**: Ctrl+Z. **Zoom**: mouse wheel. **Pan**: hold Space or the middle
  button and drag. Press `0` to fit the picture to the window.
- **Part list**: rename parts, change their folder (`Body/Arm L`), move them
  up or down in draw order, add new parts, or delete parts.
- **Export PSD**: saves the project and rebuilds the PSD.

## How it works

1. **Background**: if the image has transparency it is used as-is. Otherwise
   ISNet cuts the character out. Leftover background (railings, sky between
   an arm and the body) is removed when its colour matches the background and
   MobileSAM doesn't see it as part of the character.
2. **Regions**: the picture is split into small regions that never cross a
   drawn line. Painted art without lines uses superpixels instead. Every part
   is built from whole regions, so part edges follow the artist's lines.
3. **Detection**:
   - an anime face detector (also tried at ±15°/30° for tilted heads);
   - MediaPipe body keypoints, run on the cut-out character;
   - skin and hair colour models sampled around the face.
4. **Parts**:
   - **Face and features**: the face is the skin inside the face oval.
     Eyes, brows, mouth and nose are what is enclosed by it. Each eye splits
     into eye line, iris and white.
   - **Hair**: hair is split into front, side and back by where it sits
     relative to the face.
   - **Body**: the body is split along the skeleton (upper/fore arm, hand,
     thigh/shin or whole leg, upper/lower body). Cloth just below the hips is
     the skirt.
   - **Props**: MobileSAM objects that are long and thin, or held in a hand,
     and lie outside the body become props.
   - **Outlines**: each line goes to the part in front of it.
5. **Hidden areas**: holes are filled, and pieces are swept onward under the
   parts that cover them. Anime mode paints those areas by inpainting from the
   part's own pixels. Cartoon mode redraws them in flat colour with
   continuing outlines.
6. **PSD**: a small writer for grouped RGB layers. Folders follow the draw
   order exactly.

## Models

These are downloaded on first use into `~/.cache/live2d_psd`; set the
`LIVE2D_PSD_MODELS` environment variable to use another folder. Everything
is optional: without a model the tool falls back to colour and position
rules.

| model | used for | licence | from |
| --- | --- | --- | --- |
| ISNet (DIS) | background removal | AGPL-3.0 | npm `@imgly/background-removal-node` |
| MediaPipe pose landmarker | body keypoints | Apache-2.0 | storage.googleapis.com |
| lbpcascade_animeface | anime faces | MIT | github.com/nagadomi |
| MobileSAM | AI select, object masks | Apache-2.0 | `models/` in this repo, made by `tools/export_mobilesam.py` |

## Limits

- Heavily stylised or non-human characters with no face the detector knows
  fall back to the cartoon rules (cartoon art) or a single "Body" layer
  (shaded art). Fix those in the editor in a minute or two.
- Shapes that aren't drawn at all, such as the back of a sleeve or the far
  side of a skirt, can't be invented. Hidden areas are extended and inpainted
  from what is visible.
- Painted eyes with no outlines are found less reliably than cel-shaded ones.
- Keypoints for poses the body detector doesn't understand (lying down, heavy
  foreshortening) can put arms and legs in the wrong place. AI select in the
  editor is the quick fix.

## Tests

```
python -m pytest tests
```
