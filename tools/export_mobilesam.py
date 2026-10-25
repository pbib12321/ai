"""Export MobileSAM to two ONNX files used by live2d_psd (run once, needs torch).

    pip install torch timm segment-anything onnx
    curl -LO https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/weights/mobile_sam.pt
    curl -LO https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/mobile_sam/modeling/tiny_vit_sam.py
    python tools/export_mobilesam.py mobile_sam.pt models/

MobileSAM (https://github.com/ChaoningZhang/MobileSAM) is Apache-2.0.
The decoder export takes a batch of point prompts at once, which the
official SAM export doesn't, so "segment everything" runs quickly.
"""
import os
import sys

import torch
from segment_anything.modeling import MaskDecoder, PromptEncoder, Sam, TwoWayTransformer

sys.path.insert(0, os.getcwd())
from tiny_vit_sam import TinyViT  # noqa: E402


def build(ckpt):
    sam = Sam(
        image_encoder=TinyViT(
            img_size=1024, in_chans=3, num_classes=1000, embed_dims=[64, 128, 160, 320], depths=[2, 2, 6, 2],
            num_heads=[2, 4, 5, 10], window_sizes=[7, 7, 14, 7], mlp_ratio=4.0, drop_rate=0.0, drop_path_rate=0.0,
            use_checkpoint=False, mbconv_expand_ratio=4.0, local_conv_size=3, layer_lr_decay=0.8,
        ),
        prompt_encoder=PromptEncoder(embed_dim=256, image_embedding_size=(64, 64), input_image_size=(1024, 1024), mask_in_chans=16),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8),
            transformer_dim=256, iou_head_depth=3, iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return sam.eval()


class Encoder(torch.nn.Module):
    def __init__(self, sam):
        super().__init__()
        self.enc = sam.image_encoder

    def forward(self, image):  # already normalised and padded to 1024x1024
        return self.enc(image)


class Decoder(torch.nn.Module):
    def __init__(self, sam):
        super().__init__()
        self.pe = sam.prompt_encoder
        self.dec = sam.mask_decoder

    def forward(self, embeddings, points, labels):
        b = points.shape[0]
        # same as PromptEncoder._embed_points(pad=True), written without boolean
        # indexing so the batch and point counts stay dynamic in ONNX
        points = torch.cat([points + 0.5, torch.zeros_like(points[:, :1])], 1)
        labels = torch.cat([labels, -torch.ones_like(labels[:, :1])], 1)[..., None]
        pe = self.pe.pe_layer.forward_with_coords(points, self.pe.input_image_size)
        sparse = (
            pe * (labels != -1).float()
            + self.pe.not_a_point_embed.weight * (labels == -1).float()
            + self.pe.point_embeddings[0].weight * (labels == 0).float()
            + self.pe.point_embeddings[1].weight * (labels == 1).float()
        )
        dense = self.pe.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(b, -1, 64, 64)
        low, iou = self.dec.predict_masks(
            image_embeddings=embeddings, image_pe=self.pe.get_dense_pe(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
        )
        return low[:, 1:], iou[:, 1:]


def main(ckpt, out):
    os.makedirs(out, exist_ok=True)
    sam = build(ckpt)
    with torch.no_grad():
        torch.onnx.export(
            Encoder(sam), torch.zeros(1, 3, 1024, 1024), os.path.join(out, "mobilesam_encoder.onnx"),
            input_names=["image"], output_names=["embeddings"], opset_version=17, dynamo=False,
        )
        torch.onnx.export(
            Decoder(sam), (torch.zeros(1, 256, 64, 64), torch.zeros(4, 1, 2), torch.ones(4, 1)),
            os.path.join(out, "mobilesam_decoder.onnx"),
            input_names=["embeddings", "points", "labels"], output_names=["masks", "scores"],
            dynamic_axes={"points": {0: "batch", 1: "n"}, "labels": {0: "batch", 1: "n"}, "masks": {0: "batch"}, "scores": {0: "batch"}},
            opset_version=17, dynamo=False,
        )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
