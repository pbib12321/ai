# live2d_psd

Turns a 2D picture (anime, cartoon, painted art, animals, mascots) into:

1. **An animated Live2D-style puppet** in a preview window, with idle
   animation, mouse following, hair/cloth/tail physics, wind, and an animated
   background (drifting clouds, swaying plants, waving grass).
2. **A layered PSD** that is ready to rig in **Live2D Cubism Editor** (or to
   open in Photopea or Photoshop).

```
./setup_mac.sh                                   # once (macOS Apple Silicon; Linux/Windows: pip install -r requirements.txt)
source .venv/bin/activate
python live2d.py picture.png                     # convert + open the animated preview
python live2d.py picture.png --follow            # start in mouse-follow mode
python live2d.py picture.png --record out.mp4    # save a video instead
python live2d.py edit picture_cubism.l2d         # fix parts in the browser editor, then run again
python live2d.py convert picture.png             # just the PSD
```

### The preview

| key | |
| --- | --- |
| **M** | idle ↔ mouse follow: head, eyes and body follow the mouse |
| **Q / W** | less / more wind |
| **P** | physics on / off |
| **B** | background on / off |
| **Space** | pause |
| **C** | screenshot |
| **R** | start / stop recording an .mp4 |
| **+ / −** | zoom |
| **Esc** | quit |

The puppet is driven by the standard Live2D parameters:

- **Head**: angle X/Y/Z, which slides each layer by its depth plus a dome
  warp, so the face turns instead of sliding flat.
- **Body**: angle X/Z.
- **Breathing**.
- **Eyes**: per-eye blinking (random, with occasional double blinks) and
  eyeball X/Y. Irises and pupils are clipped to the eye white, so they never
  leave the eye.
- **Mouth**: opening.
- **Brows**: move with the blink.
- **Arms**: sway.
- **Ears**: twitch (animals).
- **Physics**: front, side and back hair, skirts, sleeves, tails and ears
  hang from an anchor on two-link pendulums. They swing from the motion of
  what they hang from, and from gusty wind.

The idle mode looks around with small saccades and breathes.

**Note on `.moc3`:** this is our own runtime. Live2D's `.moc3` format and
Cubism Core are proprietary, and the open projects that write `.moc3`
(e.g. image2live2d) say the remaining barrier is Live2D's licence. For a real
Cubism model, import the PSD into Cubism Editor. Its layers, names, folders
and filled-in hidden areas are prepared the way the official sample models
are.

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
| MI-GAN | AI inpainting of hidden areas and of the background behind the character | see MI-GAN `LICENSE-WEIGHTS` | huggingface.co/andraniksargsyan/migan (`LIVE2D_PSD_AI_INPAINT=0` to turn off) |

## Animals, mascots and backgrounds

- **Creature mode**: when no human face or body is found, the eyes are found
  as a pair of round blobs. The head is everything above a chin line set from
  their spacing (or MobileSAM's mask when it agrees). Ears are what sticks out
  of the skull, and the tail is what sticks out of the body.
- **Hidden eye parts are redrawn**: the iris is completed as a circle fitted
  to its visible rim, and the eye white as an ellipse. Other hidden areas are
  filled by MI-GAN when it's available, and by OpenCV inpainting otherwise.
  Flat cartoon art copies its own pixels along the hidden shape, so outlines
  continue underneath.
- **Background**:
  - The sky (blue or cloud colours connected to the top, with clouds in it)
    becomes a seamlessly looping, slowly drifting texture.
  - Green foliage becomes plants hinged where they enter the picture.
  - Wide green strips at the bottom become waving grass.
  - Everything is painted out underneath, so nothing tears when it moves.

## Limits

- Heavily stylised or non-human characters with no face the detector knows
  fall back to the cartoon rules (cartoon art) or a single "Body" layer
  (shaded art). Fix those in the editor in a minute or two.
- Shapes that aren't drawn at all, such as the back of a sleeve or the far
  side of a skirt, can't be invented. Hidden areas are extended and inpainted
  from what is visible.
- Painted eyes with no outlines are found less reliably than cel-shaded ones.
- No automatic method gets every picture right. When a part is wrong, open
  the editor, fix it with AI select or the brush, and run again. The fixed
  parts are reused.
- The best open-source layer decomposer, See-through (SIGGRAPH 2026), needs
  an NVIDIA GPU with 12–16 GB of memory. It isn't used here because it
  doesn't run on a Mac.
- Keypoints for poses the body detector doesn't understand (lying down, heavy
  foreshortening) can put arms and legs in the wrong place. AI select in the
  editor is the quick fix.

## Tests

```
python -m pytest tests
```
