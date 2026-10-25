# Minimal stand-ins for the See-through helpers extend_sam.py uses.
import numpy as np
import torch


def fix_params(model):
    for _, p in model.named_parameters():
        p.requires_grad = False


def img2tensor(img, dtype=torch.float32, device="cpu"):
    a = np.asarray(img)
    if a.ndim == 2:
        a = a[..., None]
    return torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))[None].to(device=device, dtype=dtype)


def tensor2img(t):
    return t.detach().cpu().numpy()
