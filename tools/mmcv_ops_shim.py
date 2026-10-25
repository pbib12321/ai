"""Stand-in for mmcv.ops (mmcv-lite has no compiled ops): NMS via torchvision,
everything else a placeholder that fails only if actually used."""
import sys, types, torch, torchvision

m = types.ModuleType("mmcv.ops")

def nms(boxes, scores, iou_threshold, offset=0, score_threshold=0, max_num=-1):
    keep = torchvision.ops.nms(boxes.float(), scores.float(), float(iou_threshold))
    if max_num > 0:
        keep = keep[:max_num]
    return torch.cat([boxes[keep], scores[keep, None]], 1), keep

def batched_nms(boxes, scores, idxs, nms_cfg, class_agnostic=False):
    nms_cfg_ = dict(nms_cfg or {})
    thr = nms_cfg_.get("iou_threshold", nms_cfg_.get("iou_thr", 0.5))
    max_num = nms_cfg_.get("max_num", -1)
    if class_agnostic:
        keep = torchvision.ops.nms(boxes.float(), scores.float(), thr)
    else:
        keep = torchvision.ops.batched_nms(boxes.float(), scores.float(), idxs, thr)
    if max_num > 0:
        keep = keep[:max_num]
    return torch.cat([boxes[keep], scores[keep, None]], -1), keep

class _Missing:
    def __init__(self, *a, **k):
        raise RuntimeError("mmcv op not available in mmcv-lite")

def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    return type(name, (_Missing,), {})

m.nms = nms
m.batched_nms = batched_nms
m.__getattr__ = __getattr__
sys.modules["mmcv.ops"] = m
import mmcv
mmcv.ops = m

# submodules (mmcv.ops.roi_align, ...) resolve to the same stand-in
import importlib.abc, importlib.machinery
m.__path__ = []

class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name.startswith("mmcv.ops."):
            return importlib.machinery.ModuleSpec(name, self)
        return None
    def create_module(self, spec):
        sub = types.ModuleType(spec.name)
        sub.__getattr__ = __getattr__
        sub.nms, sub.batched_nms = nms, batched_nms
        return sub
    def exec_module(self, module):
        pass

sys.meta_path.insert(0, _Finder())
