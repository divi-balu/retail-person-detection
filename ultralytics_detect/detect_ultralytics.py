"""
detect_ultralytics.py — YOLOv3 person detection using Ultralytics (pretrained COCO weights)

This script uses official pretrained YOLOv3 weights via the Ultralytics library.
Weights auto-download on first run (~230MB).

Usage:
    # Image
    python detect_ultralytics.py --source mall.jpg --output output.jpg

    # Video
    python detect_ultralytics.py --source store_video.mp4 --output out.mp4

    # Webcam
    python detect_ultralytics.py --source 0

    # Set confidence threshold
    python detect_ultralytics.py --source mall.jpg --conf 0.4
"""

import argparse
import cv2
import time
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("Run: pip install ultralytics")


# ── Config ────────────────────────────────────────────────────────────────────
PERSON_CLASS  = 0        # COCO class 0 = person
MODEL_WEIGHTS = "yolov3.pt"   # auto-downloaded by ultralytics on first run
BOX_COLOR     = (0, 255, 80)
TEXT_COLOR    = (255, 255, 0)


# ── Helpers ───────────────────────────────────────────────────────────────────

def draw_count(frame, count):
    cv2.putText(frame, f"Count: {count}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, TEXT_COLOR, 3)


def process_image(model, source, output, conf):
    """Run detection on a single image."""
    results = model(source, classes=[PERSON_CLASS], conf=conf, verbose=False)
    result  = results[0]

    # Draw on original image
    frame = cv2.imread(source)
    count = 0

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        score = float(box.conf[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
        cv2.putText(frame, f"Person {score:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, BOX_COLOR, 2)
        count += 1

    draw_count(frame, count)
    cv2.imwrite(output, frame)
    print(f"[✓] Detected {count} person(s) → saved: {output}")
    return count


def process_video(model, source, output, conf):
    """Run detection on video or webcam."""
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

        t0 = time.time()
        results = model(frame, classes=[PERSON_CLASS], conf=conf, verbose=False)
        total_time += time.time() - t0

        count = 0
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            score = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
            cv2.putText(frame, f"Person {score:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, BOX_COLOR, 2)
            count += 1

        draw_count(frame, count)
        avg_fps = (frame_idx + 1) / (total_time + 1e-6)
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

        if writer:
            writer.write(frame)

        cv2.imshow("Retail Person Detection (Ultralytics)", frame)
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
    parser = argparse.ArgumentParser(description="YOLOv3 Retail Person Detection (Ultralytics)")
    parser.add_argument("--source",  required=True, help="Image/video path or 0 for webcam")
    parser.add_argument("--output",  default="output.jpg", help="Output file path")
    parser.add_argument("--conf",    type=float, default=0.5, help="Confidence threshold")
    args = parser.parse_args()

    print(f"[*] Loading YOLOv3 pretrained weights (auto-download if first run)...")
    model = YOLO(MODEL_WEIGHTS)
    print(f"[✓] Model ready")

    is_video = args.source.isdigit() or Path(args.source).suffix in [".mp4", ".avi", ".mov", ".mkv"]
    if is_video:
        process_video(model, args.source, args.output, args.conf)
    else:
        process_image(model, args.source, args.output, args.conf)


if __name__ == "__main__":
    main()
