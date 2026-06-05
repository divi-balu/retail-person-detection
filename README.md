# Retail Person Detection — YOLOv3 (PyTorch)

Implementation of **YOLOv3** from the original paper, applied to retail person detection.

> Redmon & Farhadi, *"YOLOv3: An Incremental Improvement"*, arXiv 2018  
> https://arxiv.org/abs/1804.02767

## Use Case

Retail video analytics — detecting and counting customers in store environments.  
Directly applicable to: footfall tracking, queue detection, heatmap generation.

---

## Architecture

```
Input (416×416)
      │
  Darknet-53 Backbone  (53 conv layers + residual blocks)
      │
  ┌───┴───────────────┐
  │                   │
Scale 1 (13×13)   Scale 2 (26×26)   Scale 3 (52×52)
Small people      Medium people     Large/close people
      │
  Multi-scale Detection Heads → Decode → NMS → Output
```

**Key design choices from the paper:**
- Darknet-53: deeper than Darknet-19 (YOLOv2), uses residual connections
- 3 detection scales: handles people at varying distances in retail scenes
- 9 anchors total (k-means clustered on COCO person bboxes)
- Logistic regression for objectness (not softmax) → better multi-label support

---

## Project Structure

```
retail-person-detection/
├── model.py          # YOLOv3 + Darknet-53 architecture (PyTorch)
├── utils.py          # IoU, NMS, bbox decoding, preprocessing
├── detect.py         # Inference on image / video / webcam
├── train.py          # Fine-tuning on COCO person class
├── requirements.txt
├── demo/
│   └── output.jpg    # Sample detection result
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
git clone https://github.com/YOUR_USERNAME/retail-person-detection
cd retail-person-detection
pip install -r requirements.txt
```

### 2. Run on an image (demo mode — random weights)
```bash
python detect.py --source demo/store.jpg --output demo/output.jpg
```

### 3. Run on webcam
```bash
python detect.py --source 0
```

### 4. Run with pretrained weights
Download YOLOv3 COCO weights (person class):
```bash
wget https://pjreddie.com/media/files/yolov3.weights
# Convert to PyTorch: use darknet2pytorch (see below)
python detect.py --source demo/store.jpg --weights yolov3_person.pth
```

### 5. Fine-tune on COCO
```bash
# Download COCO 2017: https://cocodataset.org/#download
python train.py --data_dir ./coco --epochs 30 --batch_size 8
```

---

## Implementation Details

### IoU Calculation
Standard bounding box IoU used for NMS and anchor matching:
```
IoU = Intersection / Union
    = (overlap area) / (area1 + area2 - overlap)
```

### Non-Maximum Suppression (NMS)
- Sort boxes by confidence score (descending)
- Greedily keep highest-confidence box
- Suppress any box with IoU > 0.45 with a kept box
- Threshold: confidence > 0.5

### Bbox Decoding (Section 2.1, paper)
```
bx = sigmoid(tx) + cx    # centre x
by = sigmoid(ty) + cy    # centre y
bw = pw * exp(tw)        # width
bh = ph * exp(th)        # height
```
Where cx, cy = grid offsets; pw, ph = anchor dimensions.

---

## Results (Demo)

| Metric | Value |
|--------|-------|
| Input resolution | 416 × 416 |
| Inference speed (CPU) | ~2–3 FPS |
| Inference speed (GPU) | ~30+ FPS |
| Person class AP (COCO) | ~51% (paper baseline) |

---

## Retail Applications

- **Footfall counting** — track customer entries/exits
- **Queue detection** — alert when person count exceeds threshold
- **Zone analytics** — detect dwell time in specific shelf zones
- **Heatmap generation** — aggregate bounding box centres over time

---

## References

1. Redmon & Farhadi (2018). *YOLOv3: An Incremental Improvement.* arXiv:1804.02767
2. Lin et al. (2014). *Microsoft COCO: Common Objects in Context.*
3. He et al. (2016). *Deep Residual Learning for Image Recognition.*

---

## Author

**Divya Balasubramanian**  
PhD Candidate, IIT Madras | ML Engineer  
[divi.b21@gmail.com](mailto:divi.b21@gmail.com)
