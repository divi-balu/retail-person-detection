"""
detect.py — Run YOLOv3 person detection on image or video.

Usage:
    # Image
    python detect.py --source demo/store.jpg --output demo/out.jpg

    # Webcam
    python detect.py --source 0

    # Video file
    python detect.py --source demo/store_video.mp4 --output demo/out.mp4

    # Use pretrained weights
    python detect.py --source demo/store.jpg --weights yolov3_person.pth
"""

import argparse
import torch
import cv2
import os
import time

from model import YOLOv3
from utils import (
    ANCHORS, preprocess, decode_predictions, nms, draw_boxes
)

CONF_THRESHOLD = 0.5
NMS_THRESHOLD  = 0.45
INPUT_SIZE     = 416


def load_model(weights_path=None, device="cpu"):
    model = YOLOv3(num_classes=1).to(device)
    model.eval()

    if weights_path and os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"[✓] Loaded weights: {weights_path}")
    else:
        print("[!] No weights loaded — using random init (demo mode)")
        print("    Download pretrained weights: see README.md")

    return model


def run_inference(model, tensor, device="cpu"):
    """Forward pass + decode all 3 scales → merge → NMS."""
    tensor = tensor.to(device)

    with torch.no_grad():
        o1, o2, o3 = model(tensor)

    anchor_sets = [ANCHORS["scale1"], ANCHORS["scale2"], ANCHORS["scale3"]]
    all_boxes, all_scores = [], []

    for out, anchors in zip([o1, o2, o3], anchor_sets):
        boxes, scores, _ = decode_predictions(out, anchors, INPUT_SIZE)
        all_boxes.append(boxes[0])
        all_scores.append(scores[0])

    boxes  = torch.cat(all_boxes,  dim=0)
    scores = torch.cat(all_scores, dim=0)

    keep = nms(boxes, scores, NMS_THRESHOLD, CONF_THRESHOLD)
    return boxes[keep], scores[keep]


def detect_image(model, source, output, device):
    tensor, img = preprocess(source, INPUT_SIZE)
    boxes, scores = run_inference(model, tensor, device)

    h, w = img.shape[:2]
    result, count = draw_boxes(img, boxes.cpu(), scores.cpu(), (h, w))

    cv2.imwrite(output, result)
    print(f"[✓] Detected {count} person(s) → saved: {output}")


def detect_video(model, source, output, device):
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output, fourcc, fps, (width, height))

    frame_count = 0
    total_time  = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE))
        tensor  = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0

        t0 = time.time()
        boxes, scores = run_inference(model, tensor, device)
        total_time += time.time() - t0

        result, count = draw_boxes(frame, boxes.cpu(), scores.cpu(), (height, width))

        frame_count += 1
        avg_fps = frame_count / (total_time + 1e-6)
        cv2.putText(result, f"FPS: {avg_fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        if writer:
            writer.write(result)

        cv2.imshow("Retail Person Detection", result)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"[✓] Processed {frame_count} frames | Avg FPS: {frame_count/(total_time+1e-6):.1f}")


def main():
    parser = argparse.ArgumentParser(description="YOLOv3 Retail Person Detection")
    parser.add_argument("--source",  required=True, help="Image/video path or 0 for webcam")
    parser.add_argument("--output",  default="demo/output.jpg", help="Output path")
    parser.add_argument("--weights", default=None, help="Path to .pth weights file")
    parser.add_argument("--device",  default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model  = load_model(args.weights, device)

    # Detect
    is_video = args.source.isdigit() or args.source.endswith((".mp4", ".avi", ".mov"))
    if is_video:
        detect_video(model, args.source, args.output, device)
    else:
        detect_image(model, args.source, args.output, device)


if __name__ == "__main__":
    main()
