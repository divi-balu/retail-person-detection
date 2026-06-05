"""
detect_official.py — YOLOv3 person detection using converted official Darknet weights.

Steps before running:
    1. Download weights:  wget https://pjreddie.com/media/files/yolov3.weights
    2. Convert weights:   python convert_weights.py --weights yolov3.weights --output yolov3_coco.pth
    3. Run detection:     python detect_official.py --source mall.jpg --weights yolov3_coco.pth

Usage:
    python detect_official.py --source mall.jpg --weights yolov3_coco.pth --output output.jpg
    python detect_official.py --source store.mp4 --weights yolov3_coco.pth --output out.mp4
    python detect_official.py --source 0 --weights yolov3_coco.pth
"""

import argparse
import torch
import cv2
import time
from pathlib import Path

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model import YOLOv3
from utils import ANCHORS, decode_predictions, nms, draw_boxes, preprocess

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_SIZE     = 416
NUM_CLASSES    = 80        # full COCO (person is class index 0)
PERSON_CLASS   = 0
CONF_THRESHOLD = 0.5
NMS_THRESHOLD  = 0.45
BOX_COLOR      = (0, 255, 80)
TEXT_COLOR     = (255, 255, 0)


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(weights_path, device):
    model = YOLOv3(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print(f"[✓] Loaded: {weights_path}")
    return model


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(model, tensor, device):
    """Forward pass all 3 scales → merge → filter person class → NMS."""
    tensor = tensor.to(device)

    with torch.no_grad():
        o1, o2, o3 = model(tensor)

    all_boxes, all_scores = [], []

    for out, anchor_key in zip([o1, o2, o3], ["scale1", "scale2", "scale3"]):
        anchors = ANCHORS[anchor_key]
        boxes, scores, classes = decode_predictions(out, anchors, INPUT_SIZE)

        # Filter person class only
        person_mask = (classes[0] == PERSON_CLASS)
        all_boxes.append(boxes[0][person_mask])
        all_scores.append(scores[0][person_mask])

    boxes  = torch.cat(all_boxes,  dim=0)
    scores = torch.cat(all_scores, dim=0)

    keep = nms(boxes, scores, NMS_THRESHOLD, CONF_THRESHOLD)
    return boxes[keep], scores[keep]


# ── Image ─────────────────────────────────────────────────────────────────────

def detect_image(model, source, output, device):
    tensor, img = preprocess(source, INPUT_SIZE)
    boxes, scores = run_inference(model, tensor, device)

    h, w = img.shape[:2]
    result, count = draw_boxes(img, boxes.cpu(), scores.cpu(), (h, w))

    cv2.imwrite(output, result)
    print(f"[✓] Detected {count} person(s) → saved: {output}")


# ── Video ─────────────────────────────────────────────────────────────────────

def detect_video(model, source, output, device):
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output, fourcc, fps, (width, height))

    frame_idx  = 0
    total_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE))
        tensor  = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0

        t0 = time.time()
        boxes, scores = run_inference(model, tensor, device)
        total_time += time.time() - t0

        result, count = draw_boxes(frame, boxes.cpu(), scores.cpu(), (height, width))

        avg_fps = (frame_idx + 1) / (total_time + 1e-6)
        cv2.putText(result, f"FPS: {avg_fps:.1f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

        if writer:
            writer.write(result)

        cv2.imshow("Retail Person Detection (Official Weights)", result)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"[✓] Done — {frame_idx} frames | Avg FPS: {frame_idx/(total_time+1e-6):.1f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="YOLOv3 Retail Person Detection (Official Weights)")
    parser.add_argument("--source",  required=True)
    parser.add_argument("--weights", required=True, help="Path to yolov3_coco.pth")
    parser.add_argument("--output",  default="output.jpg")
    parser.add_argument("--device",  default="cpu")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else args.device)
    model  = load_model(args.weights, device)

    is_video = args.source.isdigit() or Path(args.source).suffix in [".mp4", ".avi", ".mov", ".mkv"]
    if is_video:
        detect_video(model, args.source, args.output, device)
    else:
        detect_image(model, args.source, args.output, device)


if __name__ == "__main__":
    main()
