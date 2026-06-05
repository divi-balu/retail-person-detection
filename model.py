"""
YOLOv3 Implementation in PyTorch
Based on: "YOLOv3: An Incremental Improvement" - Redmon & Farhadi (2018)
https://arxiv.org/abs/1804.02767

Applied to: Retail person detection use case
Author: divi-balu
"""

import torch
import torch.nn as nn
import numpy as np


# ── Darknet-53 Building Blocks ──────────────────────────────────────────────

class ConvBNLeaky(nn.Module):
    """Conv → BatchNorm → LeakyReLU block (standard YOLOv3 unit)"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    """Darknet-53 residual unit: 1x1 conv → 3x3 conv + skip connection"""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNLeaky(channels, channels // 2, kernel_size=1),
            ConvBNLeaky(channels // 2, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)


# ── Darknet-53 Backbone ──────────────────────────────────────────────────────

class Darknet53(nn.Module):
    """
    Darknet-53 feature extractor (Table 1, YOLOv3 paper).
    Outputs feature maps at 3 scales for multi-scale detection.
    """
    def __init__(self):
        super().__init__()

        self.stem = ConvBNLeaky(3, 32, kernel_size=3, padding=1)

        # Downsampling + residual stages
        self.stage1 = self._make_stage(32, 64, num_blocks=1)    # /2
        self.stage2 = self._make_stage(64, 128, num_blocks=2)   # /4
        self.stage3 = self._make_stage(128, 256, num_blocks=8)  # /8  → scale3
        self.stage4 = self._make_stage(256, 512, num_blocks=8)  # /16 → scale2
        self.stage5 = self._make_stage(512, 1024, num_blocks=4) # /32 → scale1

    def _make_stage(self, in_ch, out_ch, num_blocks):
        layers = [ConvBNLeaky(in_ch, out_ch, kernel_size=3, stride=2, padding=1)]
        for _ in range(num_blocks):
            layers.append(ResidualBlock(out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        s3 = self.stage3(x)   # large objects  (52x52 for 416 input)
        s2 = self.stage4(s3)  # medium objects (26x26)
        s1 = self.stage5(s2)  # small objects  (13x13)
        return s1, s2, s3


# ── Detection Head ────────────────────────────────────────────────────────────

class DetectionHead(nn.Module):
    """
    YOLOv3 detection head for one scale.
    Outputs: (batch, num_anchors, grid_h, grid_w, 5 + num_classes)
    """
    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        out_channels = num_anchors * (5 + num_classes)

        self.conv = nn.Sequential(
            ConvBNLeaky(in_channels, in_channels // 2, kernel_size=1),
            ConvBNLeaky(in_channels // 2, in_channels, kernel_size=3, padding=1),
            ConvBNLeaky(in_channels, in_channels // 2, kernel_size=1),
            ConvBNLeaky(in_channels // 2, in_channels, kernel_size=3, padding=1),
            ConvBNLeaky(in_channels, in_channels // 2, kernel_size=1),
        )
        self.out_conv = nn.Sequential(
            ConvBNLeaky(in_channels // 2, in_channels, kernel_size=3, padding=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        )

    def forward(self, x):
        feat = self.conv(x)
        out = self.out_conv(feat)
        # Reshape: (B, A*(5+C), H, W) → (B, A, H, W, 5+C)
        B, _, H, W = out.shape
        out = out.view(B, self.num_anchors, 5 + self.num_classes, H, W)
        out = out.permute(0, 1, 3, 4, 2).contiguous()
        return out, feat  # feat used for upsampling path


# ── Full YOLOv3 ───────────────────────────────────────────────────────────────

class YOLOv3(nn.Module):
    """
    Full YOLOv3 with multi-scale detection (Section 2.3, paper).
    Default config: COCO person class only (num_classes=1 for retail use).
    """
    def __init__(self, num_classes=1, num_anchors=3):
        super().__init__()
        self.backbone = Darknet53()

        # Scale 1: 13x13 (detects small people far away)
        self.head1 = DetectionHead(1024, num_anchors, num_classes)
        self.upsample1 = nn.Sequential(
            ConvBNLeaky(512, 256, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

        # Scale 2: 26x26 (detects medium-distance people)
        self.head2 = DetectionHead(768, num_anchors, num_classes)  # 256+512
        self.upsample2 = nn.Sequential(
            ConvBNLeaky(384, 128, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

        # Scale 3: 52x52 (detects nearby people / large bboxes)
        self.head3 = DetectionHead(384, num_anchors, num_classes)  # 128+256

    def forward(self, x):
        s1, s2, s3 = self.backbone(x)

        out1, feat1 = self.head1(s1)
        up1 = self.upsample1(feat1)

        out2, feat2 = self.head2(torch.cat([up1, s2], dim=1))
        up2 = self.upsample2(feat2)

        out3, _ = self.head3(torch.cat([up2, s3], dim=1))

        return out1, out2, out3  # 3 scale predictions


if __name__ == "__main__":
    model = YOLOv3(num_classes=1)
    x = torch.randn(1, 3, 416, 416)
    o1, o2, o3 = model(x)
    print(f"Scale1 (13x13): {o1.shape}")
    print(f"Scale2 (26x26): {o2.shape}")
    print(f"Scale3 (52x52): {o3.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
