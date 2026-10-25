"""Export CartoonSegmentation's anime instance segmenter (RTMDet-Ins-L,
https://github.com/CartoonSegmentation/CartoonSegmentation, weights MIT at
huggingface.co/dreMaz/AnimeInstanceSegmentation) to ONNX.

Needs the CartoonSegmentation checkout plus mmengine, mmdet 3.3 and mmcv
(mmcv-lite works with tools/mmcv_ops_shim.py).  The runtime only needs the
.onnx file: post-processing is reimplemented in live2d_psd/instances.py.

    python tools/export_animeins.py CartoonSegmentation/ rtmdetl_e60.ckpt models/animeins_rtmdet.onnx
"""
import os
import sys

import numpy as np

np.float_ = np.float64  # mmdet 3.3 still uses the NumPy 1 alias
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mmcv_ops_shim  # noqa: E402,F401

import torch  # noqa: E402


def main(repo, ckpt, out):
    sys.path.insert(0, repo)
    from animeinsseg.models.rtmdet_inshead_custom import RTMDetInsSepBNHeadCustom  # noqa: F401 (registers the head)
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    from mmengine import Config
    from mmengine.model.utils import revert_sync_batchnorm

    register_all_modules()
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = Config.fromstring(ck["meta"]["cfg"].replace("file_client_args", "backend_args"), file_format=".py")
    model = revert_sync_batchnorm(MODELS.build(cfg.model))
    model.load_state_dict(ck["state_dict"], strict=False)
    model.eval()
    head = model.bbox_head
    print("weight_nums", head.weight_nums, "bias_nums", head.bias_nums, "strides", head.prior_generator.strides)

    class Net(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):  # x: (1, 3, 640, 640) BGR, already normalised
            cls, box, ker, feat = self.m.bbox_head(self.m.extract_feat(x))
            return (*cls, *box, *ker, feat)

    names = [f"cls{i}" for i in range(3)] + [f"box{i}" for i in range(3)] + [f"ker{i}" for i in range(3)] + ["mask_feat"]
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        ref = Net(model)(x)
    full = out + ".full.onnx"
    torch.onnx.export(Net(model), x, full, input_names=["image"], output_names=names, opset_version=17, dynamo=False,
                      do_constant_folding=False, training=torch.onnx.TrainingMode.EVAL)
    # Ship the graph only: every weight becomes a named graph input that the
    # runtime fills from the original checkpoint (downloaded from Hugging Face).
    import onnx

    g = onnx.load(full)
    sd = {k: v for k, v in ck["state_dict"].items()}
    keep, moved = [], 0
    for init in g.graph.initializer:
        key = init.name[2:] if init.name.startswith("m.") else None
        if key in sd and tuple(sd[key].shape) == tuple(init.dims):
            g.graph.input.append(onnx.helper.make_tensor_value_info(init.name, init.data_type, list(init.dims)))
            moved += 1
        else:
            keep.append(init)
    del g.graph.initializer[:]
    g.graph.initializer.extend(keep)
    onnx.save(g, out)
    os.remove(full)
    print(f"moved {moved} weights out; graph file {os.path.getsize(out) / 1e6:.1f} MB")

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from live2d_psd import instances

    sess = instances._session_from(out, ckpt)
    got = sess.run(None, {"image": x.numpy()})
    print("max diff", max(float(np.abs(a.numpy() - b).max()) for a, b in zip(ref, got)))
    print("shapes", [tuple(g.shape) for g in got])


if __name__ == "__main__":
    main(*sys.argv[1:4])
