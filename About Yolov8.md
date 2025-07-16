
# 🦾 YOLOv8 Comprehensive Guide
A hands‑on reference for Ultralytics’ **YOLOv8** object‑detection family  
_Last updated: July 2025_

![What’s new in YOLOv8](https://blog.roboflow.com/content/images/size/w1200/2024/04/image-1791.webp)

## 📑 Table of Contents
1. [Overview](#overview)  
2. [Key Improvements over YOLOv5](#key-improvements)  
3. [Architecture Break‑down](#architecture)  
4. [Model Zoo & Benchmarks](#model-zoo)  
5. [Supported Tasks](#tasks)  
6. [Quick‑start Workflow](#workflow)  
7. [Tips for Custom Training](#tips)  
8. [Resources & References](#resources)  

---

<a id="overview"></a>
## 1. Overview  
* **Release date:** 10 Jan 2023  
* **Maintainer:** [Ultralytics](https://github.com/ultralytics)  
* **License:** AGPL‑3.0 for open‑source use, commercial licensing available  
* **Vision:** Provide a single, production‑ready repo that supports real‑time **detection, segmentation, key‑point, oriented‑box & classification** from desktop to edge.

---

<a id="key-improvements"></a>
## 2. Key Improvements over YOLOv5  

| Area | YOLOv5 | **YOLOv8** |
|------|--------|-----------|
| Backbone | `C3` blocks | **`C2f`** blocks with wider residual paths |
| First stem conv | `6×6 Conv` | **3×3 Conv** → faster & lighter |
| Head | Anchor‑based, coupled | **Anchor‑free, decoupled** head |
| Loss | BCE + CIoU | **VariFocal / DFL + Task‑aware IoU** |
| Augmentation | Mosaic, mix‑up | Same + smarter auto‑scale and Albumentations hooks |
| Export | PyTorch, ONNX | **Adds** CoreML, TF‑Lite, TensorRT, RT‑DETR JSON |

![YOLOv5 vs YOLOv8 architecture (C2f blocks & decoupled head)](https://yolov8.org/wp-content/uploads/2024/01/YOLOv5-vs-YOLOv8-3.webp)

---

<a id="architecture"></a>
## 3. Architecture Break‑down  

```text
Input → 🏗  CSPDarknet‑53 (C2f)  
      → 🔀  PANet‑FPN Neck (C2f blocks)  
      → 🎯  Anchor‑free Head  
            • Classification branch  
            • Bounding‑box regression branch
```

* **Backbone (CSPDarknet + C2f)** – keeps gradient flow efficient while cutting parameters.  
* **Neck (PANet)** – fuses multi‑scale features (P3–P5) with path aggregation.  
* **Head** – predicts (cx, cy, w, h, cls) per cell, removing objectness branch for speed.

---

<a id="model-zoo"></a>
## 4. Model Zoo & Benchmarks (COCO validation)

| Model | Params | FLOPs | mAP<sub>50‑95</sub> | A100‑TensorRT (ms) |
|-------|-------:|------:|--------------------:|-------------------:|
| **yolov8n** | 3.1 M | 23.3 B | 37.3 % | **0.99** |
| **yolov8s** | 11.4 M | 76.3 B | 44 % | 1.91 |
| **yolov8m** | 26.4 M | 208 B | 50.2 % | 3.57 |
| **yolov8l** | 44.5 M | 434 B | 52.9 % | 5.42 |
| **yolov8x** | 69.5 M | 677 B | 53.9 % | 6.88 |

> *Numbers from Ultralytics docs & benchmark runs on NVIDIA A100.*

---

<a id="tasks"></a>
## 5. Supported Tasks & Weights  

| Task | File suffix | Variants available |
|------|-------------|--------------------|
| Detection | `*.pt` | `yolov8{n,s,m,l,x}.pt` |
| Instance Segmentation | `*-seg.pt` | `yolov8{n,s,m,l,x}-seg.pt` |
| Key‑point / Pose | `*-pose.pt` | `yolov8{n,s,m,l,x}-pose.pt` (+`x-p6`) |
| Oriented BBox | `*-obb.pt` | `yolov8{n,s,m,l,x}-obb.pt` |
| Classification | `*-cls.pt` | `yolov8{n,s,m,l,x}-cls.pt` |

---

<a id="workflow"></a>
## 6. Quick‑start Workflow  

### ① Installation  

```bash
pip install ultralytics
```

### ② Inference  

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model('bus.jpg')
results.show()
```

![Sample input](https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg)

### ③ Training (COCO‑8 demo)

```bash
yolo train model=yolov8s.pt data=coco8.yaml epochs=100 imgsz=640
```

### ④ Export  

```bash
yolo export model=yolov8s.pt format=onnx
```

---

<a id="tips"></a>
## 7. Tips for Custom Training  

* **Start small ➜ large** – begin with `yolov8n` for rapid over‑fit checks.  
* **Augment** – enable mosaic, HSV, and flip; disable if dataset already heavily varied.  
* **Freeze backbone** for small datasets; unfreeze after initial convergence.  
* **Use mixed‑precision (FP16)** for faster GPU training (`--half`).  
* **Validate regularly** with `yolo val` to watch over‑fit.

---

<a id="resources"></a>
## 8. Resources & Further Reading  

* Official docs – <https://docs.ultralytics.com/models/yolov8/>  
* Roboflow deep‑dive blog – <https://blog.roboflow.com/what-is-yolov8/>  
* Community architecture diagram – Issue #189 on Ultralytics GitHub  
* Benchmarks & release notes – <https://github.com/ultralytics/ultralytics/releases>

---

> © 2025 Compiled for educational use.  
