"""A Live2D-style puppet built from the layers: meshes, parameters, physics.

This is our own small runtime (Live2D's .moc3 format and Cubism Core are
proprietary).  It works like a Cubism model:

* every layer becomes a textured grid mesh;
* standard parameters drive deformers: head angle X/Y/Z (parallax by depth
  plus a dome warp, so the face turns rather than slides), body angle and
  breathing, per-eye blink, eyeball X/Y, mouth open, arm sway;
* irises/pupils are clipped to their eye white (a clip group, as in Cubism),
  so they never leave the eye;
* hair, skirts, tails, ears, sleeves and plants hang from an anchor and swing
  on two-link pendulums driven by the motion of what they hang from and by
  wind; grass waves; clouds scroll in a loop.

The deformation is computed on the CPU with numpy every frame; viewer.py
draws it with OpenGL.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------- roles


def role_of(name: str, path: list[str]) -> str:
    if path and path[0].startswith("Character "):
        path = path[1:]
    n = name.lower()
    folders = " ".join(p.lower() for p in path)
    if n.startswith("background"):
        return "background"
    if n.startswith("sky"):
        return "sky"
    if n.startswith("grass"):
        return "grass"
    if n.startswith("plant"):
        return "plant"
    if n.startswith("eye white"):
        return "eye_white"
    if n.startswith(("iris", "pupil")):
        return "iris"
    if n.startswith("eye line"):
        return "eye_line"
    if n.startswith("brow"):
        return "brow"
    if n.startswith("mouth"):
        return "mouth"
    if n.startswith("hair front"):
        return "hair_front"
    if n.startswith("hair side"):
        return "hair_side"
    if n.startswith("hair back"):
        return "hair_back"
    if n.startswith("hair"):
        return "hair_side"
    if n.startswith("earring"):
        return "earring"
    if n.startswith("ear"):
        return "ear"
    if n.startswith(("headwear", "eyewear", "nose")):
        return "face_part"
    if n.startswith("wing"):
        return "wing"
    if n.startswith("tail"):
        return "tail"
    if n.startswith(("lower body", "bottom", "skirt")):
        return "skirt"
    if n.startswith("sleeve"):
        return "sleeve"
    if n.startswith(("upper arm", "forearm", "hand", "arm")) or "arm l" in folders or "arm r" in folders:
        return "arm"
    if n.startswith(("leg", "thigh", "shin", "foot")) or "leg l" in folders or "leg r" in folders:
        return "leg"
    if n.startswith("prop"):
        return "prop"
    if path[:1] == ["Head"] or n.startswith(("face", "nose")):
        return "face_part"
    return "body"


HEAD_ROLES = {"face_part", "eye_white", "iris", "eye_line", "brow", "mouth", "hair_front", "hair_side", "hair_back", "ear",
              "earring"}
# how far in front of the face each piece sits (parallax when the head turns)
DEPTH = {
    "hair_front": 1.0, "brow": 0.85, "eye_line": 0.8, "iris": 0.8, "eye_white": 0.75, "mouth": 0.75,
    "face_part": 0.45, "ear": 0.3, "earring": 0.3, "hair_side": 0.15, "hair_back": -0.8,
}


def side_of(name: str) -> str | None:
    for s in (" L", " R"):
        if name.endswith(s) or f"{s} " in name:
            return s.strip()
    return None


# --------------------------------------------------------------------------- meshes


@dataclass(eq=False)
class Mesh:
    name: str
    role: str
    side: str | None
    z: float
    rgba: np.ndarray  # texture (straight alpha)
    left: int
    top: int
    pos0: np.ndarray  # (N, 2) rest positions, canvas pixels
    uv: np.ndarray  # (N, 2)
    tri: np.ndarray  # (M, 3) uint32
    depth: float = 0.0
    clip_to: "Mesh | None" = None
    chain: "Chain | None" = None
    extra: dict = field(default_factory=dict)
    owner: int = 0  # which character (rig) of the stage drives this mesh

    @property
    def size(self):
        return self.rgba.shape[1], self.rgba.shape[0]


def grid_mesh(rgba: np.ndarray, left: int, top: int, cell: float | None = None):
    h, w = rgba.shape[:2]
    if cell is None:
        cell = max(6.0, max(w, h) / 22)
    nx, ny = max(1, int(math.ceil(w / cell))), max(1, int(math.ceil(h / cell)))
    xs = np.linspace(0, w, nx + 1)
    ys = np.linspace(0, h, ny + 1)
    a = rgba[..., 3] > 0
    keep = np.zeros((ny, nx), bool)
    for j in range(ny):
        y0, y1 = int(ys[j]), max(int(ys[j]) + 1, int(math.ceil(ys[j + 1])))
        for i in range(nx):
            x0, x1 = int(xs[i]), max(int(xs[i]) + 1, int(math.ceil(xs[i + 1])))
            keep[j, i] = a[y0:y1, x0:x1].any()
    # one cell of margin so edges never clip during deformation
    from scipy import ndimage as ndi

    keep = ndi.binary_dilation(keep) if keep.any() else keep
    vid = -np.ones((ny + 1, nx + 1), np.int64)
    used = np.zeros((ny + 1, nx + 1), bool)
    for j, i in zip(*np.nonzero(keep)):
        used[j : j + 2, i : i + 2] = True
    vid[used] = np.arange(used.sum())
    gy, gx = np.nonzero(used)
    pos = np.stack([xs[gx] + left, ys[gy] + top], 1).astype(np.float32)
    uv = np.stack([xs[gx] / w, ys[gy] / h], 1).astype(np.float32)
    tris = []
    for j, i in zip(*np.nonzero(keep)):
        a_, b_, c_, d_ = vid[j, i], vid[j, i + 1], vid[j + 1, i], vid[j + 1, i + 1]
        tris += [(a_, b_, c_), (b_, d_, c_)]
    return pos, uv, np.array(tris, np.uint32).reshape(-1, 3)


# --------------------------------------------------------------------------- physics


@dataclass(eq=False)
class Chain:
    """Two-link pendulum hanging from ``anchor`` along ``hang``."""

    anchor: np.ndarray
    hang: np.ndarray  # unit vector, rest direction root -> tip
    reach: float  # rest length root -> tip in pixels
    stiffness: float = 18.0  # spring back to rest
    damping: float = 3.5
    inertia: float = 1.0  # how much the parent's motion swings it
    wind: float = 1.0  # how much wind pushes it
    max_angle: float = 0.6  # radians
    wag: float = 0.0  # self-driven oscillation (tails)
    th: np.ndarray = field(default_factory=lambda: np.zeros(2))
    om: np.ndarray = field(default_factory=lambda: np.zeros(2))
    prev: np.ndarray | None = None
    vel: np.ndarray | None = None
    phase: float = field(default_factory=lambda: random.uniform(0, 6.28))

    def tangent(self):
        return np.array([-self.hang[1], self.hang[0]])

    def step(self, dt, anchor_now, wind_vec, t):
        if self.prev is None:
            self.prev, self.vel = anchor_now.copy(), np.zeros(2)
        vel = (anchor_now - self.prev) / max(dt, 1e-4)
        acc = (vel - self.vel) / max(dt, 1e-4)
        self.prev, self.vel = anchor_now.copy(), vel
        tang = self.tangent()
        L = max(self.reach, 1.0)
        # wind sets where the pendulum wants to rest; the parent's acceleration kicks it
        target = self.wind * float(wind_vec @ tang) * 0.35
        target += self.wag * math.sin(t * 2.2 + self.phase)
        force = -self.inertia * float(acc @ tang) / L * 0.9
        for k, (kk, cc) in enumerate(((self.stiffness, self.damping), (self.stiffness * 0.55, self.damping * 0.7))):
            goal = target if k == 0 else self.th[0] + (target - self.th[0]) * 0.5
            a = -kk * (self.th[k] - goal) - cc * self.om[k] + force * (1.0 if k == 0 else 1.6)
            self.om[k] += a * dt
            self.th[k] += self.om[k] * dt
            self.th[k] = float(np.clip(self.th[k], -self.max_angle, self.max_angle))

    def deform(self, pts):
        """Bend points by the pendulum angles (stiff at the root, free at the tip)."""
        rel = pts - self.anchor
        s = np.clip((rel @ self.hang) / max(self.reach, 1.0), 0, 1.2)
        ang = self.th[0] * s + (self.th[1] - self.th[0]) * s * s
        c, si = np.cos(ang), np.sin(ang)
        x = rel[:, 0] * c - rel[:, 1] * si
        y = rel[:, 0] * si + rel[:, 1] * c
        return np.stack([x, y], 1) + self.anchor


# --------------------------------------------------------------------------- rig


@dataclass
class Rig:
    width: int
    height: int
    meshes: list[Mesh]
    head_center: np.ndarray
    neck: np.ndarray
    hip: np.ndarray
    face_size: float
    shoulders: dict
    eyes: dict  # side -> (cx, cy, w, h, lid_y)
    mouth: tuple | None
    notes: list = field(default_factory=list)


def _bbox(m: Mesh):
    a = m.rgba[..., 3] > 8
    ys, xs = np.nonzero(a)
    if len(ys) == 0:
        return (m.left, m.top, m.left + 1, m.top + 1)
    return (m.left + xs.min(), m.top + ys.min(), m.left + xs.max() + 1, m.top + ys.max() + 1)


def _nearest_point(m: Mesh, target):
    a = m.rgba[..., 3] > 64
    ys, xs = np.nonzero(a)
    if len(ys) == 0:
        return np.array([m.left, m.top], np.float32)
    pts = np.stack([xs + m.left, ys + m.top], 1).astype(np.float32)
    i = np.argmin(((pts - target) ** 2).sum(1))
    return pts[i]


def _centroid(m: Mesh):
    a = m.rgba[..., 3].astype(np.float32)
    ys, xs = np.nonzero(a > 8)
    if len(ys) == 0:
        return np.array([m.left, m.top], np.float32)
    return np.array([xs.mean() + m.left, ys.mean() + m.top], np.float32)


def build_rig(layers, width: int, height: int, creature: bool = False) -> Rig:
    meshes: list[Mesh] = []
    for L in sorted(layers, key=lambda l: l.z):
        role = role_of(L.name, list(L.path))
        if role in ("sky", "background"):
            h, w = L.rgba.shape[:2]
            pos = np.array([[L.left, L.top], [L.left + w, L.top], [L.left, L.top + h], [L.left + w, L.top + h]], np.float32)
            uv = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], np.float32)
            tri = np.array([[0, 1, 2], [1, 3, 2]], np.uint32)
        else:
            pos, uv, tri = grid_mesh(L.rgba, L.left, L.top)
        if len(tri) == 0:
            continue
        m = Mesh(L.name, role, side_of(L.name), L.z, L.rgba, L.left, L.top, pos, uv, tri)
        m.depth = DEPTH.get(role, 0.0)
        m.extra.update(getattr(L, "extra", {}) or {})
        meshes.append(m)

    by_role = {}
    for m in meshes:
        by_role.setdefault(m.role, []).append(m)
    char = [m for m in meshes if m.role not in ("background", "sky", "plant", "grass")]
    if not char:
        char = meshes
    xs0 = min(_bbox(m)[0] for m in char)
    ys0 = min(_bbox(m)[1] for m in char)
    xs1 = max(_bbox(m)[2] for m in char)
    ys1 = max(_bbox(m)[3] for m in char)

    face = [m for m in meshes if m.role == "face_part" and m.name.lower().startswith("face") and "detail" not in m.name.lower()]
    head_parts = [m for m in meshes if m.role in HEAD_ROLES]
    if face:
        fb = _bbox(max(face, key=lambda m: m.rgba[..., 3].sum()))
    elif head_parts:
        bbs = [_bbox(m) for m in head_parts]
        fb = (min(b[0] for b in bbs), min(b[1] for b in bbs), max(b[2] for b in bbs), max(b[3] for b in bbs))
    else:
        fb = (xs0, ys0, xs1, ys0 + (ys1 - ys0) * 0.35)
    head_center = np.array([(fb[0] + fb[2]) / 2, (fb[1] + fb[3]) / 2], np.float32)
    face_size = float(max(fb[2] - fb[0], fb[3] - fb[1]))
    neck = np.array([head_center[0], fb[3] - 0.08 * (fb[3] - fb[1])], np.float32)

    lower = by_role.get("skirt", [])
    if lower:
        hb = _bbox(max(lower, key=lambda m: m.rgba[..., 3].sum()))
        hip = np.array([(hb[0] + hb[2]) / 2, hb[1] + 0.1 * (hb[3] - hb[1])], np.float32)
    else:
        body = [m for m in meshes if m.role == "body"]
        if body:
            bb = _bbox(max(body, key=lambda m: m.rgba[..., 3].sum()))
            hip = np.array([(bb[0] + bb[2]) / 2, bb[1] + 0.85 * (bb[3] - bb[1])], np.float32)
        else:
            hip = np.array([head_center[0], ys1], np.float32)
    if hip[1] <= neck[1]:
        hip[1] = neck[1] + 0.5 * max(1.0, ys1 - neck[1])

    body_center = np.array([(xs0 + xs1) / 2, (neck[1] + hip[1]) / 2], np.float32)

    # ---- eyes (clip groups, blink geometry)
    eyes = {}
    for side in ("L", "R"):
        ew = [m for m in meshes if m.role == "eye_white" and m.side == side]
        iris = [m for m in meshes if m.role == "iris" and m.side == side]
        lines = [m for m in meshes if m.role == "eye_line" and m.side == side]
        ref = ew or lines or iris
        if not ref:
            continue
        bbs = [_bbox(m) for m in ref]
        x0, y0, x1, y1 = min(b[0] for b in bbs), min(b[1] for b in bbs), max(b[2] for b in bbs), max(b[3] for b in bbs)
        eyes[side] = (float((x0 + x1) / 2), float((y0 + y1) / 2), float(x1 - x0), float(y1 - y0), float(y0 + 0.62 * (y1 - y0)))
        if ew:
            for m in iris:
                m.clip_to = ew[0]
    mouth = None
    mm = by_role.get("mouth", [])
    if mm:
        b = _bbox(mm[0])
        mouth = (float((b[0] + b[2]) / 2), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1]))

    # ---- shoulders: where each arm group meets the body
    shoulders = {}
    for side in ("L", "R"):
        arm = [m for m in meshes if m.role in ("arm", "sleeve") and m.side == side]
        if arm:
            top = min(arm, key=lambda m: _bbox(m)[1])
            shoulders[side] = _nearest_point(top, np.array([body_center[0], neck[1]], np.float32))

    # ---- physics chains
    for m in meshes:
        b = _bbox(m)
        bh = b[3] - b[1]
        c = _centroid(m)
        ch = None
        if m.role in ("hair_front", "hair_side", "hair_back"):
            anchor = np.array([c[0], b[1] + 0.15 * bh], np.float32)
            if m.role == "hair_back":
                anchor = np.array([head_center[0], max(b[1], head_center[1] - 0.3 * face_size)], np.float32)
            ch = Chain(anchor, np.array([0, 1.0], np.float32), max(bh * 0.85, 10), stiffness=16, damping=3.2,
                       inertia=1.0, wind=0.7 if m.role != "hair_front" else 0.45, max_angle=0.22)
        elif m.role == "skirt":
            anchor = np.array([c[0], b[1]], np.float32)
            # gentle: the same role covers flowing skirts and tight shorts
            ch = Chain(anchor, np.array([0, 1.0], np.float32), max(bh, 10), stiffness=24, damping=4.5, inertia=0.6,
                       wind=0.35, max_angle=0.07)
        elif m.role == "sleeve":
            anchor = np.array([c[0], b[1]], np.float32)
            ch = Chain(anchor, np.array([0, 1.0], np.float32), max(bh, 10), stiffness=26, damping=4, inertia=0.6,
                       wind=0.4, max_angle=0.15)
        elif m.role == "tail":
            anchor = _nearest_point(m, body_center)
            d = c - anchor
            ch = Chain(anchor, (d / (np.linalg.norm(d) + 1e-6)).astype(np.float32), max(float(np.linalg.norm(d)) * 2, 10),
                       stiffness=10, damping=2.2, inertia=1.0, wind=0.5, max_angle=0.5, wag=0.18 if creature else 0.08)
        elif m.role == "ear":
            anchor = _nearest_point(m, head_center)
            d = c - anchor
            ch = Chain(anchor, (d / (np.linalg.norm(d) + 1e-6)).astype(np.float32), max(float(np.linalg.norm(d)) * 2, 10),
                       stiffness=30, damping=4, inertia=0.8, wind=0.15, max_angle=0.25)
            m.extra["twitch"] = True
        elif m.role == "earring":
            anchor = np.array([c[0], b[1]], np.float32)
            ch = Chain(anchor, np.array([0, 1.0], np.float32), max(bh, 6), stiffness=12, damping=1.8, inertia=1.4,
                       wind=0.3, max_angle=0.5)
        elif m.role == "wing":
            anchor = _nearest_point(m, body_center)
            d = c - anchor
            ch = Chain(anchor, (d / (np.linalg.norm(d) + 1e-6)).astype(np.float32), max(float(np.linalg.norm(d)) * 2, 10),
                       stiffness=9, damping=2.0, inertia=0.6, wind=0.3, max_angle=0.25, wag=0.08)
        elif m.role == "plant":
            anchor = np.array(m.extra.get("anchor", (c[0], b[3])), np.float32)
            d = c - anchor
            n = float(np.linalg.norm(d)) or 1.0
            ch = Chain(anchor, (d / n).astype(np.float32), max(n * 2, 10), stiffness=6, damping=1.6, inertia=0.0,
                       wind=1.6, max_angle=0.12)
        m.chain = ch
    # props: held by the closest hand, moved rigidly from the grip
    hands = [m for m in meshes if m.role == "arm" and m.name.lower().startswith("hand")] or [m for m in meshes if m.role == "arm"]
    for m in meshes:
        if m.role != "prop":
            continue
        best, grip = None, None
        for hm in hands:
            q = _nearest_point(m, _centroid(hm))
            d = float(np.linalg.norm(q - _centroid(hm)))
            if best is None or d < best:
                best, grip, side = d, q, hm.side
        m.extra["grip"] = grip if grip is not None else _centroid(m)
        if grip is not None and best < 0.6 * face_size:
            m.extra["held_by"] = side
    rig = Rig(width, height, meshes, head_center, neck, hip, face_size, shoulders, eyes, mouth)
    rig.body_center = body_center
    return rig


# --------------------------------------------------------------------------- animation


def _noise(t, seeds):
    # smooth pseudo-random signal in [-1, 1] from a few incommensurate sines
    return sum(math.sin(t * f + p) * a for f, p, a in seeds) / sum(a for _, _, a in seeds)


class Animator:
    """Drives the parameters (idle or mouse follow) and turns them into vertex positions."""

    PARAMS = ("AngleX", "AngleY", "AngleZ", "BodyX", "BodyZ", "Breath", "EyeOpenL", "EyeOpenR", "EyeBallX", "EyeBallY",
              "MouthOpen", "BrowY", "ArmL", "ArmR")

    def __init__(self, rig: Rig, seed: int = 1):
        self.rig = rig
        self.t = 0.0
        self.mode = "idle"
        self.wind = 0.6
        self.physics = True
        self.p = {k: 0.0 for k in self.PARAMS}
        self.p.update(EyeOpenL=1.0, EyeOpenR=1.0)
        self.target = dict(self.p)
        rnd = random.Random(seed)
        self.seeds = {k: [(rnd.uniform(0.15, 0.6), rnd.uniform(0, 6.3), rnd.uniform(0.5, 1.0)) for _ in range(3)] for k in self.PARAMS}
        self.next_blink = rnd.uniform(1.0, 3.0)
        self.blink_t = -1.0
        self.next_saccade = 0.5
        self.saccade = np.zeros(2)
        self.next_twitch = rnd.uniform(3, 7)
        self.twitch_t = -1.0
        self.mouse = None  # canvas coords
        self.rnd = rnd
        self.wind_dir = np.array([1.0, 0.0])
        self.wind_vec = np.zeros(2)
        self.sky_offset = 0.0

    # ---------------------------------------------------------------- params
    def update(self, dt):
        self.t += dt
        t = self.t
        n = lambda k: _noise(t, self.seeds[k])
        if self.mode == "follow" and self.mouse is not None:
            r = self.rig
            dx = (self.mouse[0] - r.head_center[0]) / max(r.width, 1) * 2.4
            dy = (self.mouse[1] - r.head_center[1]) / max(r.height, 1) * 2.4
            dx, dy = float(np.clip(dx, -1, 1)), float(np.clip(dy, -1, 1))
            self.target.update(AngleX=30 * dx, AngleY=-30 * dy, AngleZ=-6 * dx * dy, BodyX=10 * dx, BodyZ=-3 * dx,
                               EyeBallX=dx, EyeBallY=dy)
        else:
            if t > self.next_saccade:
                self.saccade = np.array([self.rnd.uniform(-0.6, 0.6), self.rnd.uniform(-0.3, 0.3)])
                self.next_saccade = t + self.rnd.uniform(0.8, 2.8)
            self.target.update(AngleX=12 * n("AngleX"), AngleY=7 * n("AngleY"), AngleZ=5 * n("AngleZ"),
                               BodyX=4 * n("BodyX"), BodyZ=2.5 * n("BodyZ"),
                               EyeBallX=float(np.clip(0.4 * n("AngleX") + self.saccade[0], -1, 1)),
                               EyeBallY=float(np.clip(-0.25 * n("AngleY") + self.saccade[1], -1, 1)))
        self.target["ArmL"] = 3 * n("ArmL")
        self.target["ArmR"] = 3 * n("ArmR")
        self.target["Breath"] = 0.5 + 0.5 * math.sin(t * 2 * math.pi / 3.6)
        self.target["MouthOpen"] = max(0.0, 0.12 * n("MouthOpen"))
        # blinking
        if t > self.next_blink and self.blink_t < 0:
            self.blink_t = 0.0
        open_ = 1.0
        if self.blink_t >= 0:
            self.blink_t += dt
            ph = self.blink_t / 0.16
            open_ = abs(1 - 2 * min(ph, 1.0)) if ph < 1 else 1.0
            if ph >= 1:
                self.blink_t = -1
                self.next_blink = t + (self.rnd.uniform(0.15, 0.3) if self.rnd.random() < 0.2 else self.rnd.uniform(2.0, 5.5))
        self.target["EyeOpenL"] = self.target["EyeOpenR"] = open_
        self.target["BrowY"] = 0.4 * (1 - open_) + 0.2 * n("BrowY")
        # smooth toward the targets (eyelids snap, head eases)
        for k, v in self.target.items():
            rate = 40.0 if k.startswith("EyeOpen") else (14.0 if k.startswith("EyeBall") else 5.0)
            self.p[k] += (v - self.p[k]) * min(1.0, rate * dt)
        # wind: steady breeze with gusts
        g = 0.55 + 0.3 * math.sin(t * 0.45) + 0.25 * max(0.0, math.sin(t * 1.3 + 1.0)) ** 3
        self.wind_vec = self.wind_dir * self.wind * g
        self.sky_offset += dt * 0.004 * (0.3 + self.wind)
        # ear twitches
        if t > self.next_twitch and self.twitch_t < 0:
            self.twitch_t = 0.0
        if self.twitch_t >= 0:
            self.twitch_t += dt
            if self.twitch_t > 0.25:
                self.twitch_t = -1
                self.next_twitch = t + self.rnd.uniform(3, 8)

    # ---------------------------------------------------------------- transforms
    def _head(self, pts, m: Mesh):
        r, p = self.rig, self.p
        ax, ay = p["AngleX"] / 30.0, p["AngleY"] / 30.0
        P = 0.07 * r.face_size
        d = m.depth
        rel = pts - r.head_center
        R = 0.6 * r.face_size
        dome = np.clip(1 - (rel ** 2).sum(1) / (R * R), 0, 1)
        k = d + (0.55 * dome if m.role in ("face_part", "eye_white", "iris", "eye_line", "brow", "mouth") else 0.0)
        out = pts.copy()
        out[:, 0] += ax * P * k
        out[:, 1] -= ay * P * 0.7 * k
        # the face narrows a little on the side turning away
        side = np.sign(rel[:, 0])
        out[:, 0] -= np.abs(ax) * 0.04 * rel[:, 0] * (side == -np.sign(ax)) * (m.role != "hair_back")
        return _rotate(out, r.neck, math.radians(-p["AngleZ"]))

    def _body(self, pts, m: Mesh):
        r, p = self.rig, self.p
        out = pts.copy()
        # rigid things (a staff) move as one piece: use the sway at their grip
        ref = m.extra["grip"][None] if "grip" in m.extra else pts
        above = np.clip((r.hip[1] - ref[:, 1]) / max(r.hip[1] - r.neck[1], 1), 0, 1.3)
        out[:, 1] -= p["Breath"] * 0.012 * (r.hip[1] - r.neck[1]) * above
        out[:, 0] += p["BodyX"] / 10.0 * 0.03 * r.face_size * above
        return _rotate(out, r.hip, math.radians(-p["BodyZ"]) * np.minimum(above, 1.0))

    def _eyes(self, pts, m: Mesh):
        e = self.rig.eyes.get(m.side)
        if not e:
            return pts
        cx, cy, w, h, lid = e
        out = pts.copy()
        if m.role == "iris":
            out[:, 0] += self.p["EyeBallX"] * 0.22 * w
            out[:, 1] += self.p["EyeBallY"] * 0.16 * h
        o = max(0.06, self.p[f"EyeOpen{m.side}"])
        out[:, 1] = lid + (out[:, 1] - lid) * o
        return out

    def vertices(self, m: Mesh) -> np.ndarray:
        r, p = self.rig, self.p
        pts = m.pos0.astype(np.float64)
        if m.role in ("sky", "background"):
            return pts.astype(np.float32)
        if m.chain is not None and self.physics:
            pts = m.chain.deform(pts)
        if m.role == "grass":
            h = max(1.0, float(m.pos0[:, 1].max() - m.pos0[:, 1].min()))
            s = np.clip((m.pos0[:, 1].max() - pts[:, 1]) / h, 0, 1) ** 1.5
            pts[:, 0] += s * h * 0.06 * (self.wind_vec[0] + 0.15) * (1 + np.sin(self.t * 2.3 + pts[:, 0] * 0.05)) * 0.5
            return pts.astype(np.float32)
        if m.role == "plant":
            return pts.astype(np.float32)
        if m.role in ("eye_white", "iris", "eye_line"):
            pts = self._eyes(pts, m)
        elif m.role == "brow":
            pts[:, 1] += p["BrowY"] * 0.04 * r.face_size
        elif m.role == "mouth" and r.mouth:
            mx, my, mw, mh = r.mouth
            pts[:, 1] = my + (pts[:, 1] - my) * (1 + 0.6 * p["MouthOpen"])
        elif m.role == "ear" and self.twitch_t >= 0:
            a = 0.05 * math.sin(self.twitch_t / 0.25 * math.pi * 2)
            pts = _rotate(pts, m.chain.anchor if m.chain else r.head_center, a)
        elif m.role in ("arm", "sleeve") and m.side in r.shoulders:
            pts = _rotate(pts, r.shoulders[m.side], math.radians(p["Arm" + m.side]))
        elif m.role == "prop" and m.extra.get("held_by") in r.shoulders:
            pts = _rotate(pts, r.shoulders[m.extra["held_by"]], math.radians(p["Arm" + m.extra["held_by"]]))
        if m.role in HEAD_ROLES:
            pts = self._head(pts, m)
        if m.role not in ("leg", "plant", "grass"):
            pts = self._body(pts, m)
        return pts.astype(np.float32)

    def anchor_now(self, m: Mesh):
        """Current position of what a dangly mesh hangs from (for inertia)."""
        a = m.chain.anchor[None].astype(np.float64)
        if m.role in HEAD_ROLES:
            a = self._head(a, m)
        if m.role not in ("leg", "plant", "grass"):
            a = self._body(a, m)
        return a[0]

    def step_physics(self, dt):
        if not self.physics:
            return
        # several small steps keep the springs stable at low frame rates
        n = max(1, int(math.ceil(dt / (1 / 120))))
        for m in self.rig.meshes:
            if m.chain is None:
                continue
            a = self.anchor_now(m)
            for _ in range(n):
                m.chain.step(dt / n, a, self.wind_vec, self.t)


def _rotate(pts, c, ang):
    if np.isscalar(ang) and ang == 0:
        return pts
    ca, sa = np.cos(ang), np.sin(ang)
    rel = pts - c
    x = rel[:, 0] * ca - rel[:, 1] * sa
    y = rel[:, 0] * sa + rel[:, 1] * ca
    return np.stack([x, y], 1) + c


# --------------------------------------------------------------------------- several characters


@dataclass
class Stage:
    """All the characters of a picture (one rig each) plus the scenery."""

    width: int
    height: int
    rigs: list

    @property
    def meshes(self):
        return sorted((m for r in self.rigs for m in r.meshes), key=lambda m: m.z)


def build_stage(layers, width: int, height: int, creature: bool = False) -> Stage:
    groups: dict = {}
    loose = []
    for L in layers:
        p = list(L.path)
        if p and p[0].startswith("Character "):
            groups.setdefault(p[0], []).append(L)
        else:
            loose.append(L)
    if not groups:
        return Stage(width, height, [build_rig(layers, width, height, creature=creature)])
    rigs = []
    for i, key in enumerate(sorted(groups, key=lambda k: int(k.split()[-1]))):
        # the scenery rides on the first character's animator (wind, clouds)
        rig = build_rig(groups[key] + (loose if i == 0 else []), width, height, creature=creature)
        for m in rig.meshes:
            m.owner = i
        rigs.append(rig)
    return Stage(width, height, rigs)
