"""Minimal layered PSD writer (RGB, 8 bit, raw channels, layer groups).

Only what Live2D Cubism Editor and Photopea/Photoshop need: plain pixel
layers with transparency, nested group folders and Unicode layer names.
No clipping masks, layer masks or effects, which Cubism cannot import.
"""
from __future__ import annotations

import struct

import numpy as np


def _pascal(name: str) -> bytes:
    b = name.encode("ascii", "replace")[:255]
    data = bytes([len(b)]) + b
    data += b"\0" * ((4 - len(data) % 4) % 4)
    return data


def _block(key: bytes, payload: bytes) -> bytes:
    if len(payload) % 2:
        payload += b"\0"
    return b"8BIM" + key + struct.pack(">I", len(payload)) + payload


def _luni(name: str) -> bytes:
    u = name.encode("utf-16-be")
    return _block(b"luni", struct.pack(">I", len(u) // 2) + u)


def _record(name, top, left, bottom, right, chan_lens, blend=b"norm", flags=0x08, extra=b""):
    rec = struct.pack(">iiiiH", top, left, bottom, right, len(chan_lens))
    for cid, ln in chan_lens:
        rec += struct.pack(">hI", cid, ln)
    rec += b"8BIM" + blend + struct.pack(">BBBB", 255, 0, flags, 0)
    ex = struct.pack(">I", 0) + struct.pack(">I", 0) + _pascal(name) + _luni(name) + extra
    rec += struct.pack(">I", len(ex)) + ex
    return rec


class _Pixel:
    def __init__(self, name, rgba, left, top, visible=True):
        self.name, self.rgba, self.left, self.top, self.visible = name, rgba, left, top, visible


class _Group:
    def __init__(self, name):
        self.name = name
        self.children: list = []  # top-most first


def build_tree(layers) -> _Group:
    """layers: objects with name, path, z, rgba, left, top (any order)."""
    root = _Group("")
    for L in sorted(layers, key=lambda l: -l.z):
        g = root
        for folder in L.path:
            sub = next((c for c in g.children if isinstance(c, _Group) and c.name == folder), None)
            if sub is None:
                sub = _Group(folder)
                g.children.append(sub)
            g = sub
        g.children.append(_Pixel(L.name, L.rgba, L.left, L.top))
    return root


def write_psd(path: str, width: int, height: int, layers, merged: np.ndarray) -> None:
    root = build_tree(layers)
    records: list[bytes] = []
    chandata: list[bytes] = []

    def emit_pixel(p: _Pixel):
        h, w = p.rgba.shape[:2]
        chans = [(-1, p.rgba[..., 3]), (0, p.rgba[..., 0]), (1, p.rgba[..., 1]), (2, p.rgba[..., 2])]
        lens = []
        data = b""
        for cid, c in chans:
            raw = struct.pack(">H", 0) + np.ascontiguousarray(c).tobytes()
            lens.append((cid, len(raw)))
            data += raw
        flags = 0x08 | (0 if p.visible else 0x02)
        records.append(_record(p.name, p.top, p.left, p.top + h, p.left + w, lens, flags=flags))
        chandata.append(data)

    def emit_empty(name, kind):
        lens = [(-1, 2), (0, 2), (1, 2), (2, 2)]
        if kind == 3:
            extra = _block(b"lsct", struct.pack(">I", 3))
            records.append(_record(name, 0, 0, 0, 0, lens, flags=0x18, extra=extra))
        else:
            extra = _block(b"lsct", struct.pack(">I", 1) + b"8BIM" + b"pass")
            records.append(_record(name, 0, 0, 0, 0, lens, blend=b"pass", flags=0x18, extra=extra))
        chandata.append(struct.pack(">H", 0) * 4)

    # PSD lists layers bottom-up; a folder is "end marker, children, folder header".
    def emit(node):
        for child in reversed(node.children):
            if isinstance(child, _Group):
                emit_empty("</Layer group>", 3)
                emit(child)
                emit_empty(child.name, 1)
            else:
                emit_pixel(child)

    emit(root)

    layer_info = struct.pack(">h", -len(records)) + b"".join(records) + b"".join(chandata)
    if len(layer_info) % 2:
        layer_info += b"\0"
    layer_info = struct.pack(">I", len(layer_info)) + layer_info
    lmi = layer_info + struct.pack(">I", 0)  # empty global layer mask

    with open(path, "wb") as f:
        f.write(b"8BPS" + struct.pack(">H", 1) + b"\0" * 6)
        f.write(struct.pack(">HIIHH", 4, height, width, 8, 3))
        f.write(struct.pack(">I", 0))  # colour mode data
        f.write(struct.pack(">I", 0))  # image resources
        f.write(struct.pack(">I", len(lmi)) + lmi)
        f.write(struct.pack(">H", 0))  # raw merged image
        for c in (0, 1, 2, 3):
            f.write(np.ascontiguousarray(merged[..., c]).tobytes())
