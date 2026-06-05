# YOLOv3 — Official Darknet Weights (PyTorch)

Person detection using official YOLOv3 weights converted from Darknet binary format to PyTorch.  
This demonstrates manual weight parsing from the original Darknet format — directly from the paper authors.

## Install
```bash
pip install -r requirements.txt
```

## Step 1 — Download Official Weights
```bash
wget https://pjreddie.com/media/files/yolov3.weights
```
File size: ~236MB

## Step 2 — Convert to PyTorch
```bash
python convert_weights.py --weights yolov3.weights --output yolov3_coco.pth
```

This parses the Darknet binary format:
- Header: major, minor, revision, images_seen
- Weights: BN biases → BN weights → BN mean → BN var → Conv weights (per layer)

## Step 3 — Run Detection

**Image:**
```bash
python detect_official.py --source mall.jpg --weights yolov3_coco.pth --output output.jpg
```

**Video:**
```bash
python detect_official.py --source store.mp4 --weights yolov3_coco.pth --output out.mp4
```

**Webcam:**
```bash
python detect_official.py --source 0 --weights yolov3_coco.pth
```

## Why This Matters

Unlike Ultralytics (which wraps everything), this approach:
- Manually parses the Darknet binary weight format
- Loads weights layer-by-layer into our custom PyTorch model
- Demonstrates deep understanding of the YOLOv3 architecture internals
- Person filtered from COCO 80-class output (class index 0)

## Weight File Format (Darknet)

```
[int32 x5]  Header: major, minor, revision, seen_low, seen_high
[float32]*  For each layer:
              ConvBNLeaky: bn_bias, bn_weight, bn_mean, bn_var, conv_weight
              Conv+bias:   bias, conv_weight
```
