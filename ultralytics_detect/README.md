# YOLOv3 — Ultralytics Pretrained Weights

Person detection using official pretrained YOLOv3 COCO weights via Ultralytics.  
Weights (~230MB) are **auto-downloaded** on first run — no manual setup needed.

## Install
```bash
pip install -r requirements.txt
```

## Run

**Image:**
```bash
python detect_ultralytics.py --source store.jpg --output output.jpg
```

**Video:**
```bash
python detect_ultralytics.py --source store.mp4 --output out.mp4
```

**Webcam:**
```bash
python detect_ultralytics.py --source 0
```

**Lower confidence threshold (detect more people):**
```bash
python detect_ultralytics.py --source mall.jpg --conf 0.3
```

## Output
- Bounding boxes around each detected person
- Confidence score per detection
- Live person count (top-left corner)
- FPS display for video

## Notes
- Uses COCO class 0 (person) only — ignores all other objects
- `yolov3.pt` downloads automatically to your working directory on first run
- Press `q` to quit webcam/video window
