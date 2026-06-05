"""
convert_weights.py — Convert official Darknet YOLOv3 weights to PyTorch (.pth)

Official weights file: yolov3.weights (236MB)
Download: wget https://pjreddie.com/media/files/yolov3.weights

Usage:
    python convert_weights.py --weights yolov3.weights --output yolov3_coco.pth
"""

import argparse
import torch
import numpy as np
from model import YOLOv3


def load_darknet_weights(model, weights_path):
    """
    Parse and load Darknet binary weight file into PyTorch YOLOv3 model.

    Darknet weight file format:
    - Header: 5 x int32  (major, minor, revision, seen_images x2)
    - Weights: float32 values in layer order
      For each conv layer: biases → bn_weights → bn_mean → bn_var → conv_weights
                           (or just biases + conv_weights if no BN)
    """
    with open(weights_path, "rb") as f:
        # Read header (5 int32 values)
        header = np.frombuffer(f.read(5 * 4), dtype=np.int32)
        print(f"[Header] Major:{header[0]} Minor:{header[1]} "
              f"Revision:{header[2]} Images seen:{header[3]}")

        # Read remaining weights as float32
        weights = np.frombuffer(f.read(), dtype=np.float32)

    ptr = 0  # pointer into weights array

    def load_conv_bn(conv, bn):
        """Load Conv + BatchNorm layer weights."""
        nonlocal ptr
        num_bn_biases = bn.bias.numel()

        # BatchNorm: bias, weight, running_mean, running_var
        bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases]); ptr += num_bn_biases
        bn_weights = torch.from_numpy(weights[ptr:ptr + num_bn_biases]); ptr += num_bn_biases
        bn_mean    = torch.from_numpy(weights[ptr:ptr + num_bn_biases]); ptr += num_bn_biases
        bn_var     = torch.from_numpy(weights[ptr:ptr + num_bn_biases]); ptr += num_bn_biases

        bn.bias.data.copy_(bn_biases.view_as(bn.bias))
        bn.weight.data.copy_(bn_weights.view_as(bn.weight))
        bn.running_mean.copy_(bn_mean.view_as(bn.running_mean))
        bn.running_var.copy_(bn_var.view_as(bn.running_var))

        # Conv weights (no bias when BN is used)
        num_weights = conv.weight.numel()
        conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights]); ptr += num_weights
        conv.weight.data.copy_(conv_weights.view_as(conv.weight))

    def load_conv_bias(conv):
        """Load Conv layer with bias (detection head final conv)."""
        nonlocal ptr
        num_biases = conv.bias.numel()
        conv_biases = torch.from_numpy(weights[ptr:ptr + num_biases]); ptr += num_biases
        conv.bias.data.copy_(conv_biases.view_as(conv.bias))

        num_weights = conv.weight.numel()
        conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights]); ptr += num_weights
        conv.weight.data.copy_(conv_weights.view_as(conv.weight))

    # ── Load backbone (Darknet-53) ──
    print("[*] Loading backbone weights...")

    def load_conv_bn_block(block):
        """Load a ConvBNLeaky block."""
        conv = block.conv[0]
        bn   = block.conv[1]
        load_conv_bn(conv, bn)

    # Stem
    load_conv_bn_block(model.backbone.stem)

    # Stages
    for stage in [model.backbone.stage1, model.backbone.stage2,
                  model.backbone.stage3, model.backbone.stage4,
                  model.backbone.stage5]:
        for layer in stage:
            if hasattr(layer, 'conv'):  # ConvBNLeaky
                load_conv_bn_block(layer)
            elif hasattr(layer, 'block'):  # ResidualBlock
                for sublayer in layer.block:
                    load_conv_bn_block(sublayer)

    print(f"[✓] Backbone loaded | Weights consumed: {ptr:,} / {len(weights):,}")
    print("[*] Loading detection head weights...")

    # ── Load detection heads ──
    for head in [model.head1, model.head2, model.head3]:
        # Conv sequence (5 ConvBNLeaky blocks)
        for block in head.conv:
            load_conv_bn_block(block)
        # out_conv: ConvBNLeaky + final Conv (with bias)
        load_conv_bn_block(head.out_conv[0])
        load_conv_bias(head.out_conv[1])

    print(f"[✓] All weights loaded | Total consumed: {ptr:,} / {len(weights):,}")
    if ptr != len(weights):
        print(f"[!] Warning: {len(weights) - ptr} weights unused (may be COCO vs custom mismatch)")

    return model


def main():
    parser = argparse.ArgumentParser(description="Convert Darknet YOLOv3 weights to PyTorch")
    parser.add_argument("--weights", default="yolov3.weights", help="Path to .weights file")
    parser.add_argument("--output",  default="yolov3_coco.pth", help="Output .pth file")
    parser.add_argument("--classes", type=int, default=80, help="Number of COCO classes (80)")
    args = parser.parse_args()

    print(f"[*] Initialising YOLOv3 (num_classes={args.classes})...")
    model = YOLOv3(num_classes=args.classes)

    print(f"[*] Reading: {args.weights}")
    model = load_darknet_weights(model, args.weights)

    torch.save(model.state_dict(), args.output)
    print(f"[✓] Saved PyTorch weights → {args.output}")
    print(f"    Use with: python detect_official.py --weights {args.output}")


if __name__ == "__main__":
    main()
