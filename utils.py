"""
Utility functions for YOLOv3 retail person detection.
Covers: bbox decoding, IoU, Non-Maximum Suppression (NMS)
"""

import torch
import numpy as np
import cv2


# ── Anchor boxes (COCO-pretrained YOLOv3 defaults) ───────────────────────────
# Format: (width, height) in pixels for 416x416 input
ANCHORS = {
    "scale1": [(116, 90), (156, 198), (373, 326)],   # large  - 13x13
    "scale2": [(30, 61),  (62, 45),   (59, 119)],    # medium - 26x26
    "scale3": [(10, 13),  (16, 30),   (33, 23)],     # small  - 52x52
}
INPUT_SIZE = 416


# ── IoU ───────────────────────────────────────────────────────────────────────

def compute_iou(box1, box2):
    """
    Compute Intersection over Union between two boxes.
    Boxes format: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / (union + 1e-6)


# ── Bbox Decoding ─────────────────────────────────────────────────────────────

def decode_predictions(raw_output, anchors, input_size=416):
    """
    Decode raw YOLOv3 output to absolute bounding boxes.

    Args:
        raw_output: tensor (B, num_anchors, H, W, 5+C)
        anchors: list of (w, h) tuples for this scale
        input_size: model input size (default 416)

    Returns:
        boxes: (B, num_anchors*H*W, 4) in [x1,y1,x2,y2] abs coords
        scores: (B, num_anchors*H*W)
        classes: (B, num_anchors*H*W)
    """
    B, A, H, W, _ = raw_output.shape
    stride = input_size // H

    # Sigmoid objectness + class probabilities
    pred = raw_output.clone()
    pred[..., 0:2] = torch.sigmoid(pred[..., 0:2])  # tx, ty
    pred[..., 4]   = torch.sigmoid(pred[..., 4])    # objectness
    pred[..., 5:]  = torch.sigmoid(pred[..., 5:])   # class probs

    # Grid offsets
    grid_x = torch.arange(W).repeat(H, 1).view(1, 1, H, W).float()
    grid_y = torch.arange(H).repeat(W, 1).t().view(1, 1, H, W).float()

    anchor_w = torch.tensor([a[0] for a in anchors]).view(1, A, 1, 1).float()
    anchor_h = torch.tensor([a[1] for a in anchors]).view(1, A, 1, 1).float()

    # Decode centre coords and dimensions
    bx = (pred[..., 0] + grid_x) * stride
    by = (pred[..., 1] + grid_y) * stride
    bw = torch.exp(pred[..., 2]) * anchor_w
    bh = torch.exp(pred[..., 3]) * anchor_h

    # Convert to [x1, y1, x2, y2]
    x1 = (bx - bw / 2).unsqueeze(-1)
    y1 = (by - bh / 2).unsqueeze(-1)
    x2 = (bx + bw / 2).unsqueeze(-1)
    y2 = (by + bh / 2).unsqueeze(-1)

    boxes   = torch.cat([x1, y1, x2, y2], dim=-1).view(B, -1, 4)
    obj     = pred[..., 4].view(B, -1)
    cls_prob = pred[..., 5:].view(B, -1, pred.shape[-1] - 5)
    scores, classes = (obj.unsqueeze(-1) * cls_prob).max(dim=-1)

    return boxes, scores, classes


# ── Non-Maximum Suppression ───────────────────────────────────────────────────

def nms(boxes, scores, iou_threshold=0.45, score_threshold=0.5):
    """
    Non-Maximum Suppression to remove duplicate detections.

    Args:
        boxes: (N, 4) tensor [x1, y1, x2, y2]
        scores: (N,) confidence scores
        iou_threshold: suppress if IoU > this
        score_threshold: keep only boxes above this confidence

    Returns:
        keep: indices of surviving boxes
    """
    mask = scores > score_threshold
    boxes  = boxes[mask]
    scores = scores[mask]
    indices = torch.where(mask)[0]

    if len(boxes) == 0:
        return []

    order = scores.argsort(descending=True)
    keep  = []

    while order.numel() > 0:
        i = order[0].item()
        keep.append(indices[i].item())

        if order.numel() == 1:
            break

        rest = order[1:]
        ious = torch.tensor([
            compute_iou(boxes[i].tolist(), boxes[j].tolist())
            for j in rest
        ])
        order = rest[ious < iou_threshold]

    return keep


# ── Image Preprocessing ───────────────────────────────────────────────────────

def preprocess(image_path, size=416):
    """Load and resize image to model input format."""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (size, size))
    tensor  = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0), img  # (1, 3, H, W), original for drawing


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_boxes(image, boxes, scores, orig_size, model_size=416, color=(0, 255, 80)):
    """Draw detected person bounding boxes on original image."""
    h, w = orig_size
    scale_x = w / model_size
    scale_y = h / model_size

    count = 0
    for box, score in zip(boxes, scores):
        x1 = int(box[0] * scale_x)
        y1 = int(box[1] * scale_y)
        x2 = int(box[2] * scale_x)
        y2 = int(box[3] * scale_y)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"Person {score:.2f}"
        cv2.putText(image, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        count += 1

    cv2.putText(image, f"Count: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    return image, count
