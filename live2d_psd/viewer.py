"""OpenGL renderer for the puppet, a preview window and a video recorder.

    python -m live2d_psd animate picture.png            # window (macOS, Windows, Linux)
    python -m live2d_psd animate picture.png --record out.mp4 --seconds 8

Window keys: M idle/mouse-follow · Q/W wind -/+ · P physics on/off ·
B background on/off · Space pause · C screenshot · R start/stop recording ·
+/- zoom · Esc quit.
"""
from __future__ import annotations

import math
import os
import time

import numpy as np

from .puppet import Animator, Rig

_VS = """
#version 330
uniform vec2 u_scale;
uniform vec2 u_offset;
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    v_uv = in_uv;
    gl_Position = vec4(in_pos * u_scale + u_offset, 0.0, 1.0);
}
"""
_FS = """
#version 330
uniform sampler2D u_tex;
uniform float u_opacity;
in vec2 v_uv;
out vec4 f_color;
void main() { f_color = texture(u_tex, v_uv) * u_opacity; }
"""
# sky: the colour texture scrolls (and wraps) while the mask stays put
_FS_SKY = """
#version 330
uniform sampler2D u_tex;
uniform sampler2D u_mask;
uniform float u_scroll;
in vec2 v_uv;
out vec4 f_color;
void main() {
    vec4 c = texture(u_tex, vec2(v_uv.x + u_scroll, v_uv.y));
    float m = texture(u_mask, v_uv).r;
    f_color = vec4(c.rgb, 1.0) * m;
}
"""
_FS_COPY = """
#version 330
uniform sampler2D u_tex;
in vec2 v_uv;
out vec4 f_color;
void main() { f_color = texture(u_tex, v_uv); }
"""


def _premul(rgba):
    a = rgba[..., 3:4].astype(np.float32) / 255
    out = rgba.astype(np.float32)
    out[..., :3] *= a
    return np.ascontiguousarray(np.clip(out + 0.5, 0, 255).astype(np.uint8))


class Renderer:
    def __init__(self, ctx, rig: Rig):
        import moderngl

        self.mgl = moderngl
        self.ctx = ctx
        self.rig = rig
        self.prog = ctx.program(vertex_shader=_VS, fragment_shader=_FS)
        self.sky_prog = ctx.program(vertex_shader=_VS, fragment_shader=_FS_SKY)
        self.copy_prog = ctx.program(vertex_shader=_VS, fragment_shader=_FS_COPY)
        self.items = []
        for m in rig.meshes:
            tex = ctx.texture(m.size, 4, _premul(m.rgba).tobytes())
            tex.build_mipmaps()
            tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            extra = {}
            prog = self.prog
            if m.role == "sky":
                tex.repeat_x = True
                mask = m.extra["mask"]
                mt = ctx.texture((mask.shape[1], mask.shape[0]), 1, np.ascontiguousarray(mask).tobytes())
                mt.filter = (moderngl.LINEAR, moderngl.LINEAR)
                extra["mask"] = mt
                prog = self.sky_prog
            else:
                tex.repeat_x = tex.repeat_y = False
            vbo = ctx.buffer(reserve=len(m.pos0) * 16)
            ibo = ctx.buffer(m.tri.astype("u4").tobytes())
            vao = ctx.vertex_array(prog, [(vbo, "2f 2f", "in_pos", "in_uv")], index_buffer=ibo)
            self.items.append(dict(mesh=m, tex=tex, vbo=vbo, vao=vao, prog=prog, **extra))
        quad = np.array([[0, 0, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 1, 1]], np.float32)
        self.quad_vbo = ctx.buffer(quad.tobytes())
        self.quad = ctx.vertex_array(self.copy_prog, [(self.quad_vbo, "2f 2f", "in_pos", "in_uv")])
        self.clip_fbo = None
        self.show_background = True

    def _clip_target(self, size):
        if self.clip_fbo is None or self.clip_fbo.size != size:
            if self.clip_fbo is not None:
                self.clip_fbo.release()
                self.clip_tex.release()
            self.clip_tex = self.ctx.texture(size, 4)
            self.clip_fbo = self.ctx.framebuffer(color_attachments=[self.clip_tex])
        return self.clip_fbo

    def draw(self, anim: Animator, target, viewport, zoom=1.0, pan=(0.0, 0.0), clear=(0.16, 0.17, 0.2)):
        """Draw the puppet into ``target`` (a framebuffer) fitted into ``viewport`` (x, y, w, h)."""
        ctx, rig = self.ctx, self.rig
        vx, vy, vw, vh = viewport
        s = min(vw / rig.width, vh / rig.height) * zoom
        # canvas pixels -> NDC of the viewport, y down
        sx, sy = 2 * s / vw, -2 * s / vh
        ox = -rig.width * s / vw + pan[0]
        oy = rig.height * s / vh + pan[1]
        for prog in (self.prog, self.sky_prog):
            prog["u_scale"].value = (sx, sy)
            prog["u_offset"].value = (ox, oy)
        target.use()
        ctx.viewport = viewport
        ctx.clear(*clear, 1.0, viewport=viewport)
        ctx.enable(self.mgl.BLEND)
        ctx.blend_func = self.mgl.ONE, self.mgl.ONE_MINUS_SRC_ALPHA
        clipped = {}
        for it in self.items:
            if it["mesh"].clip_to is not None:
                clipped.setdefault(id(it["mesh"].clip_to), []).append(it)
        for it in self.items:
            m = it["mesh"]
            if m.clip_to is not None:
                continue  # drawn with its eye white
            if not self.show_background and m.role in ("background", "sky", "plant", "grass"):
                continue
            if id(m) in clipped:
                fbo = self._clip_target(target.size)
                fbo.use()
                ctx.viewport = viewport
                fbo.clear(0, 0, 0, 0, viewport=viewport)
                self._draw_item(it, anim)
                # clipped layers only show where the eye white already is
                ctx.blend_func = self.mgl.DST_ALPHA, self.mgl.ONE_MINUS_SRC_ALPHA
                for c in clipped[id(m)]:
                    self._draw_item(c, anim)
                ctx.blend_func = self.mgl.ONE, self.mgl.ONE_MINUS_SRC_ALPHA
                target.use()
                ctx.viewport = viewport
                self.copy_prog["u_scale"].value = (2.0, 2.0)
                self.copy_prog["u_offset"].value = (-1.0, -1.0)
                self.clip_tex.use(0)
                self.copy_prog["u_tex"].value = 0
                self.quad.render(self.mgl.TRIANGLE_STRIP)
                continue
            self._draw_item(it, anim)

    def _draw_item(self, it, anim):
        m = it["mesh"]
        v = anim.vertices(m)
        it["vbo"].write(np.hstack([v, m.uv]).astype("f4").tobytes())
        it["tex"].use(0)
        prog = it["prog"]
        prog["u_tex"].value = 0
        if prog is self.sky_prog:
            it["mask"].use(1)
            prog["u_mask"].value = 1
            prog["u_scroll"].value = anim.sky_offset
        else:
            prog["u_opacity"].value = 1.0
        it["vao"].render()


# --------------------------------------------------------------------------- window


def run_window(rig: Rig, title="Live2D preview", fps=60):
    import moderngl
    import pygame

    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, True)  # required on macOS
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
    info = pygame.display.Info()
    max_h = int((info.current_h or 900) * 0.85)
    max_w = int((info.current_w or 1400) * 0.85)
    s = min(max_w / rig.width, max_h / rig.height, 1.0)
    size = (max(320, int(rig.width * s)), max(320, int(rig.height * s)))
    flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
    try:
        pygame.display.set_mode(size, flags)
    except pygame.error:  # no multisampling available
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 0)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 0)
        pygame.display.set_mode(size, flags)
    ctx = moderngl.create_context()
    ren = Renderer(ctx, rig)
    anim = Animator(rig)
    clock = pygame.time.Clock()
    zoom, paused = 1.0, False
    recorder = None
    print(__doc__.split("\n\n")[2] if __doc__ else "")
    last = time.perf_counter()
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.VIDEORESIZE:
                size = ev.size
            elif ev.type == pygame.MOUSEMOTION:
                w, h = pygame.display.get_window_size()
                sc = min(w / rig.width, h / rig.height) * zoom
                cx = (ev.pos[0] - (w - rig.width * sc) / 2) / sc
                cy = (ev.pos[1] - (h - rig.height * sc) / 2) / sc
                anim.mouse = (cx, cy)
            elif ev.type == pygame.KEYDOWN:
                k = ev.key
                if k == pygame.K_ESCAPE:
                    running = False
                elif k == pygame.K_m:
                    anim.mode = "follow" if anim.mode == "idle" else "idle"
                elif k == pygame.K_w:
                    anim.wind = min(3.0, anim.wind + 0.25)
                elif k == pygame.K_q:
                    anim.wind = max(0.0, anim.wind - 0.25)
                elif k == pygame.K_p:
                    anim.physics = not anim.physics
                elif k == pygame.K_b:
                    ren.show_background = not ren.show_background
                elif k == pygame.K_SPACE:
                    paused = not paused
                elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    zoom = min(4.0, zoom * 1.15)
                elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    zoom = max(0.25, zoom / 1.15)
                elif k == pygame.K_c:
                    _save_png(ctx, f"puppet_{int(time.time())}.png")
                elif k == pygame.K_r:
                    if recorder is None:
                        recorder = _Recorder(f"puppet_{int(time.time())}.mp4", fps)
                        print("recording…")
                    else:
                        print("saved", recorder.close())
                        recorder = None
        now = time.perf_counter()
        dt = min(0.05, now - last)
        last = now
        if not paused:
            anim.update(dt)
            anim.step_physics(dt)
        w, h = pygame.display.get_window_size()
        fb = ctx.detect_framebuffer()
        fw, fh = fb.size  # may be larger than the window on Retina screens
        ren.draw(anim, fb, (0, 0, fw, fh), zoom=zoom)
        if recorder is not None:
            recorder.add(np.frombuffer(fb.read(components=3), np.uint8).reshape(fh, fw, 3)[::-1])
        pygame.display.flip()
        pygame.display.set_caption(
            f"{title} — {'mouse follow' if anim.mode == 'follow' else 'idle'} · wind {anim.wind:.2f} · "
            f"physics {'on' if anim.physics else 'off'} · {clock.get_fps():.0f} fps   [M Q W P B Space C R +/-]"
        )
        clock.tick(fps)
    if recorder is not None:
        print("saved", recorder.close())
    pygame.quit()


def _save_png(ctx, path):
    from PIL import Image

    fb = ctx.detect_framebuffer()
    w, h = fb.size
    img = np.frombuffer(fb.read(components=3), np.uint8).reshape(h, w, 3)[::-1]
    Image.fromarray(img).save(path)
    print("saved", path)


class _Recorder:
    def __init__(self, path, fps):
        self.path, self.fps, self.frames = path, fps, []

    def add(self, frame):
        self.frames.append(frame.copy())

    def close(self):
        write_video(self.path, self.frames, self.fps)
        return self.path


def write_video(path, frames, fps):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".gif":
        from PIL import Image

        ims = [Image.fromarray(f).convert("P", palette=Image.ADAPTIVE, colors=255) for f in frames]
        ims[0].save(path, save_all=True, append_images=ims[1:], duration=int(1000 / fps), loop=0, disposal=2)
        return
    try:
        import imageio.v2 as iio

        h, w = frames[0].shape[:2]
        frames = [f[: h - h % 2, : w - w % 2] for f in frames]
        iio.mimwrite(path, frames, fps=fps, quality=8, macro_block_size=2)
    except Exception as e:  # no ffmpeg: fall back to GIF next to it
        gif = os.path.splitext(path)[0] + ".gif"
        print(f"  mp4 needs imageio-ffmpeg ({e}); writing {gif}")
        write_video(gif, frames, fps)


# --------------------------------------------------------------------------- headless


def record(rig: Rig, out: str, seconds=8.0, fps=30, mode="idle", max_side=900, wind=0.8):
    """Render the animation without a window (uses EGL on Linux) and save it as video."""
    import moderngl

    try:
        ctx = moderngl.create_standalone_context(backend="egl")
    except Exception:
        ctx = moderngl.create_standalone_context()
    s = min(1.0, max_side / max(rig.width, rig.height))
    w, h = int(rig.width * s) // 2 * 2, int(rig.height * s) // 2 * 2
    color = ctx.renderbuffer((w, h), 4, samples=4 if ctx.max_samples >= 4 else 0)
    fbo = ctx.framebuffer(color_attachments=[color])
    resolve_tex = ctx.renderbuffer((w, h), 4)
    resolve = ctx.framebuffer(color_attachments=[resolve_tex])
    ren = Renderer(ctx, rig)
    anim = Animator(rig)
    anim.mode = mode
    anim.wind = wind
    frames = []
    dt = 1.0 / fps
    n = int(seconds * fps)
    for i in range(n):
        t = i * dt
        if mode == "follow":  # a pretend mouse drawing a slow figure-eight around the head
            anim.mouse = (rig.head_center[0] + math.sin(t * 0.9) * rig.width * 0.45,
                          rig.head_center[1] + math.sin(t * 1.8) * rig.height * 0.25)
        anim.update(dt)
        anim.step_physics(dt)
        ren.draw(anim, fbo, (0, 0, w, h))
        ctx.copy_framebuffer(resolve, fbo)
        frames.append(np.frombuffer(resolve.read(components=3), np.uint8).reshape(h, w, 3)[::-1].copy())
    write_video(out, frames, fps)
    return out
