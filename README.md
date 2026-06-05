# Retail Person Detection — YOLOv3 (PyTorch)

Implementation of **YOLOv3** from the original paper applied to retail person detection.  
Two complete detection pipelines included.

> Redmon & Farhadi, *"YOLOv3: An Incremental Improvement"*, arXiv 2018  
> https://arxiv.org/abs/1804.02767

---

## Repository Structure

```
retail-person-detection/
│
├── model.py                        # YOLOv3 + Darknet-53 — implemented from paper
├── utils.py                        # IoU, NMS, bbox decoding, draw boxes
├── train.py                        # Fine-tune on COCO person class
├── requirements.txt                # Base dependencies
│
├── ultralytics_detect/             # ── Method 1: Ultralytics pretrained weights
│   ├── detect_ultralytics.py       #    Easiest — weights auto-download
│   ├── requirements.txt
│   └── README.md
│
├── official_weights/               # ── Method 2: Official Darknet weights
│   ├── convert_weights.py          #    Parses Darknet binary format → PyTorch
│   ├── detect_official.py          #    Detection with converted weights
│   ├── requirements.txt
│   └── README.md
│
└── demo/
    └── output.jpg                  # Sample detection result
```

---

## Architecture

```
Input (416×416)
      │
  Darknet-53 Backbone
  (53 layers + residual blocks — Table 1, paper)
      │
  ┌───┴──────────────────────┐
  │           │              │
Scale1      Scale2        Scale3
13×13       26×26         52×52
small       medium        large/close
people      people        people
  │           │              │
  └───────────┴──────────────┘
              │
     Decode → NMS → Output
```

**Key paper concepts implemented:**
- Darknet-53 backbone with residual connections (Section 2.2)
- Multi-scale detection across 3 feature map sizes (Section 2.3)
- 9 anchors total — 3 per scale (k-means on COCO)
- Logistic objectness per anchor (Section 2.1)
- Bbox decoding: `bx = σ(tx) + cx`, `bw = pw·exp(tw)`

---

## Quick Start

### Method 1 — Ultralytics (Easiest, real detections immediately)
```bash
cd ultralytics_detect
pip install -r requirements.txt
python detect_ultralytics.py --source mall.jpg --output output.jpg
```
Weights auto-download (~230MB) on first run.

### Method 2 — Official Darknet Weights (Manual conversion)
```bash
cd official_weights
pip install -r requirements.txt

# Download official weights
wget https://pjreddie.com/media/files/yolov3.weights

# Convert Darknet → PyTorch
python convert_weights.py --weights yolov3.weights --output yolov3_coco.pth

# Run detection
python detect_official.py --source mall.jpg --weights yolov3_coco.pth
```

---

## Retail Applications

| Application | How YOLOv3 Helps |
|---|---|
| Footfall counting | Count person detections per frame |
| Queue detection | Alert when count > threshold in zone |
| Heatmap generation | Aggregate bbox centres over time |
| Dwell time analysis | Track person presence duration per zone |

---

## Results

| Metric | Value |
|---|---|
| Input size | 416 × 416 |
| CPU inference | ~2–3 FPS |
| GPU inference | ~30+ FPS |
| COCO person AP | ~51% (paper) |

---

## References

1. Redmon & Farhadi (2018). *YOLOv3: An Incremental Improvement.* arXiv:1804.02767  
2. Lin et al. (2014). *Microsoft COCO: Common Objects in Context.*  
3. He et al. (2016). *Deep Residual Learning for Image Recognition.*

---

## Author
divi_balu
