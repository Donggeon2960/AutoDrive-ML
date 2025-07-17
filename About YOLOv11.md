# Ultralytics YOLO11 완전 정복 (2025년 7월 기준)

> **요약**: YOLO11은 Ultralytics가 2024년 9월경 YOLO Vision 2024 행사와 함께 공개한 차세대 YOLO 계열 모델로, 이전 YOLOv8 대비 더 높은 정확도(mAP), 더 적은 파라미터 수, 더 빠른 추론 속도를 목표로 설계되었습니다. 동일한 워크플로우(데이터 포맷, CLI/Python API)를 유지하면서도 백본·넥(Neck) 개선, C3k2 / C2PSA 모듈 도입 등 구조적 최적화를 통해 효율을 높인 것이 특징입니다. 또한 Detection, Instance Segmentation, Pose(키포인트), Oriented Bounding Box(OBB), Image Classification 등 멀티 태스크를 모두 지원합니다. citeturn10view0turn10view1turn3view0

---

## 목차

* [1. YOLO11 한눈에 보기](#1-yolo11-한눈에-보기)
* [2. 다른 YOLO 버전과 무엇이 다른가? (YOLOv8 대비)](#2-다른-yolo-버전과-무엇이-다른가-yolov8-대비)
* [3. 모델 패밀리 & 지원 태스크](#3-모델-패밀리--지원-태스크)
* [4. 주요 성능 지표 요약](#4-주요-성능-지표-요약)
* [5. 설치 & 환경 세팅](#5-설치--환경-세팅)
* [6. 데이터 준비 (YOLO PyTorch TXT 포맷)](#6-데이터-준비-yolo-pytorch-txt-포맷)
* [7. 가장 빠른 시작: CLI Quickstart](#7-가장-빠른-시작-cli-quickstart)
* [8. Python API Quickstart](#8-python-api-quickstart)
* [9. 커스텀 데이터셋 학습 Step-by-Step](#9-커스텀-데이터셋-학습-step-by-step)
* [10. 추론 (이미지/폴더/웹캠/동영상)](#10-추론-이미지폴더웹캠동영상)
* [11. 결과 다루기: 박스/마스크/키포인트/세그멘테이션](#11-결과-다루기-박스마스크키포인트세그멘테이션)
* [12. 검증(Val) & 평가 지표 읽기](#12-검증val--평가-지표-읽기)
* [13. 모델 내보내기 & 배포 (ONNX, TensorRT 등)](#13-모델-내보내기--배포-onnx-tensorrt-등)
* [14. 추적(Track) 모드 빠른 예시](#14-추적track-모드-빠른-예시)
* [15. 초보자 체크리스트 & 자주 하는 실수](#15-초보자-체크리스트--자주-하는-실수)
* [16. 하이퍼파라미터 핵심만!](#16-하이퍼파라미터-핵심만)
* [17. 엣지 디바이스 최적화 팁 (Jetson / CPU)](#17-엣지-디바이스-최적화-팁-jetson--cpu)
* [18. 아키텍처 깊이 보기 (C3k2, C2PSA, SPPF, PAFPN)](#18-아키텍처-깊이-보기-c3k2-c2psa-sppf-pafpn)
* [19. 모델 사이즈 고르기 가이드](#19-모델-사이즈-고르기-가이드)
* [20. 학습 중단/재개 & 실험 관리](#20-학습-중단재개--실험-관리)
* [21. FAQ](#21-faq)
* [22. 참고자료](#22-참고자료)

---

## 1. YOLO11 한눈에 보기

* **공식 공개**: 2024년 9월 말 *YOLO Vision 2024* 행사와 함께 발표, 2024-09-30 공식 런칭. citeturn10view0turn10view1
* **목표**: 더 높은 정확도, 더 빠른 속도, 더 적은 파라미터(특히 mAP 향상대비 파라미터 절감). citeturn10view1turn3view0
* **백워드 호환 워크플로우**: YOLOv8 사용자가 거의 동일한 CLI / Python API 패턴으로 전환 가능. citeturn10view0turn10view2
* **멀티태스크 지원**: Detect, Segment, Pose, OBB, Classify. citeturn3view0turn10view1turn9view5
* **오픈소스 + 엔터프라이즈 옵션**: AGPL-3.0 기반 오픈소스, 상용 라이선싱 선택 가능. citeturn9view5turn4view0turn10view0

---

## 2. 다른 YOLO 버전과 무엇이 다른가? (YOLOv8 대비)

YOLO11은 YOLOv8과 거의 같은 사용 경험을 제공하지만 내부 아키텍처 최적화로 성능/효율이 개선되었습니다. 특히 mAP가 전반적으로 증가하고 파라미터 수가 줄었으며 CPU 추론 속도가 빨라졌다는 보고가 있습니다. 예: YOLO11l이 YOLOv8l 대비 mAP↑(53.4 vs 52.9) 이면서 파라미터 약 42% 감소. citeturn9view1turn10view1

Ultralytics 커뮤니티 벤치마크에 따르면 Nano~~X 전 범위에서 YOLO11이 YOLOv8 대비 mAP 0.5~~2.2pt 상승, 파라미터 22%까지 절감한 케이스가 보고되었습니다. citeturn10view1

YOLO11은 YOLOv8 워크플로우와의 호환성을 유지하므로 사용자는 기존 데이터 포맷과 스크립트를 거의 그대로 재사용할 수 있습니다. citeturn10view2turn3view0

---

## 3. 모델 패밀리 & 지원 태스크

YOLO11은 5가지 크기( n / s / m / l / x )가 각 태스크별로 제공됩니다. Detection, Segmentation, Pose(Keypoint), Oriented Bounding Box(OBB), Classification용 가중치(.pt)가 각각 존재합니다. 파일명 예: `yolo11n.pt`, `yolo11s-seg.pt`, `yolo11m-pose.pt`, `yolo11l-obb.pt`, `yolo11x-cls.pt`. citeturn3view0turn5view0turn9view2

모든 Detect/Seg/Pose 모델은 기본적으로 COCO 기반 사전학습(80 클래스; Pose는 person 키포인트), Classification 모델은 ImageNet 기반 사전학습(1K 클래스)을 사용합니다. citeturn5view0turn3view0

---

## 4. 주요 성능 지표 요약

아래는 COCO val2017 기준 YOLO11 Detect 모델의 공식 표 일부(단일 모델·단일 스케일)입니다.

| 모델      | 입력(px) | mAP50-95 | Params(M) | FLOPs(B) | 비고   |
| ------- | ------ | -------- | --------- | -------- | ---- |
| YOLO11n | 640    | 39.5     | 2.6       | 6.5      | 초경량  |
| YOLO11s | 640    | 47.0     | 9.4       | 21.5     | 소형   |
| YOLO11m | 640    | 51.5     | 20.1      | 68.0     | 범용   |
| YOLO11l | 640    | 53.4     | 25.3      | 86.9     | 고정확  |
| YOLO11x | 640    | 54.7     | 56.9      | 194.9    | 최고성능 |

(속도 측정: COCO val, Amazon EC2 P4d 기준 ONNX CPU / TensorRT10 FP16 T4 지표 제공. 표 전체는 공식 문서 참조.) citeturn5view0turn3view0

Glenn Jocher 인터뷰에 따르면 YOLO11n은 약 2.6M 파라미터, YOLO11x는 약 56M 파라미터 수준으로 경량\~대형 스펙트럼을 커버합니다. citeturn10view0

---

## 5. 설치 & 환경 세팅

가장 간단한 설치:

```bash
pip install ultralytics
```

공식 패키지는 Python>=3.8, PyTorch>=1.8 환경에서 동작하며, 최신 버전을 설치하면 YOLO11 모델 가중치가 자동으로 다운로드됩니다. citeturn4view0turn1view3

Conda, Docker, 소스에서 설치하는 방법은 Ultralytics Quickstart 문서를 참고하세요. citeturn4view0

---

## 6. 데이터 준비 (YOLO PyTorch TXT 포맷)

YOLO11은 YOLO PyTorch TXT 라벨 포맷(다크넷 포맷 기반 변형)을 사용합니다. 각 이미지당 동일한 이름의 .txt 파일에 `class cx cy w h` (정규화 좌표) 라인들을 기록합니다. citeturn9view5turn5view0

데이터셋 구성용 `data.yaml` 파일에는 `path`, `train`, `val`, (선택) `test`, `names`(클래스 목록)를 정의합니다. Roboflow 등 도구로 COCO/다른 포맷을 자동 변환할 수 있습니다. citeturn9view5turn5view0

---

## 7. 가장 빠른 시작: CLI Quickstart

사전학습된 모델로 추론:

```bash
yolo predict model=yolo11n.pt source='https://ultralytics.com/images/bus.jpg'
```

citeturn4view0

커스텀 훈련 (COCO8 예시):

```bash
yolo detect train data=coco8.yaml model=yolo11n.pt epochs=100 imgsz=640
```

사전학습 가중치를 사용하면 수렴이 빨라집니다. citeturn5view0turn4view0

학습한 모델로 추론:

```bash
yolo detect predict model=path/to/best.pt source='path/to/images/' conf=0.25
```

citeturn5view0

---

## 8. Python API Quickstart

파이썬에서 바로 사용:

```python
from ultralytics import YOLO

# 사전학습 모델 로드
model = YOLO('yolo11n.pt')

# 커스텀 데이터셋으로 훈련
results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

# 검증
metrics = model.val()

# 단일 이미지 추론
res = model('path/to/image.jpg')
res[0].show()
```

citeturn4view0turn5view0

---

## 9. 커스텀 데이터셋 학습 Step-by-Step

### 9.1 폴더 구조 예

```
datasets/mydata/
 ├── images/
 │    ├── train/*.jpg
 │    ├── val/*.jpg
 │    └── test/*.jpg  # 선택
 ├── labels/
 │    ├── train/*.txt
 │    ├── val/*.txt
 │    └── test/*.txt
 └── mydata.yaml
```

YOLO 포맷 / data.yaml 구성은 위 참조. citeturn5view0turn9view5

### 9.2 data.yaml 최소 예시

```yaml
path: datasets/mydata  # 선택
train: images/train
val: images/val
test: images/test  # 선택

names:
  0: person
  1: car
  2: traffic_light
```

데이터 경로와 클래스 이름만 맞으면 바로 학습 가능합니다. citeturn5view0turn9view5

### 9.3 훈련 실행

```bash
yolo detect train data=datasets/mydata/mydata.yaml model=yolo11s.pt epochs=100 imgsz=640 batch=16 device=0
```

* `model=yolo11s.pt` : 사전학습 가중치 로드
* `epochs=100` : 반복 횟수
* `imgsz=640` : 리사이즈 해상도
* `batch=16` : GPU VRAM에 맞게 조정
* `device=0` 또는 `'cpu'` 선택

추후 고급 설정은 `lr0`, `lrf`, `weight_decay`, `augment` 등 하이퍼파라미터에서 조정합니다. citeturn5view0turn4view0

### 9.4 재훈련(전이학습) vs 스크래치

YAML로 새 모델을 빌드하고 사전학습 가중치 전이:

```bash
yolo detect train data=mydata.yaml model=yolo11n.yaml pretrained=yolo11n.pt epochs=100 imgsz=640
```

스크래치 학습은 데이터가 매우 많거나 특수 도메인일 때에만 권장. citeturn5view0

---

## 10. 추론 (이미지/폴더/웹캠/동영상)

### 10.1 단일 이미지

```python
from ultralytics import YOLO
model = YOLO('yolo11x.pt')
results = model('path/to/img.jpg', conf=0.25)
```

결과 객체에서 박스 좌표, 클래스, confidence에 접근할 수 있습니다 (아래 11절 참조). citeturn7view0turn5view0

### 10.2 폴더 일괄 처리

```bash
yolo detect predict model=best.pt source='datasets/mydata/images/val'
```

citeturn5view0

### 10.3 웹캠 / 동영상 스트림

```bash
yolo detect predict model=best.pt source=0  # 웹캠
yolo detect predict model=best.pt source='video.mp4'
```

citeturn5view0

### 10.4 Roboflow Inference로 배포 후 추론 (선택)

```python
from inference import get_model
import supervision as sv
image = cv2.imread('image.jpg')
model = get_model(model_id='yolo11x-640')
results = model.infer(image)[0]
```

Roboflow Inference는 Jetson 등 다양한 디바이스에서 YOLO11 추론을 쉽게 해줍니다. citeturn6view0turn9view5

---

## 11. 결과 다루기: 박스/마스크/키포인트/세그멘테이션

결과 객체 순회 예:

```python
for r in results:
    boxes = r.boxes.xyxy      # (N,4)
    confs = r.boxes.conf      # (N,)
    clses = r.boxes.cls.int() # (N,)
    names = [r.names[c.item()] for c in clses]
```

세그멘테이션 결과는 `r.masks`, 키포인트는 `r.keypoints`에서 접근합니다. citeturn5view0turn7view0

---

## 12. 검증(Val) & 평가 지표 읽기

학습 후:

```python
metrics = model.val()  # 이전 학습 데이터셋 자동 기억
print(metrics.box.map)     # mAP50-95
print(metrics.box.map50)   # mAP50
print(metrics.box.maps)    # 클래스별 mAP
```

CLI:

```bash
yolo detect val model=best.pt
```

Ultralytics는 COCO 형식 평가 루틴을 사용하며 박스, 마스크, 키포인트 각각 전용 mAP를 출력합니다. citeturn5view0turn4view0

---

## 13. 모델 내보내기 & 배포 (ONNX, TensorRT 등)

Python:

```python
model = YOLO('best.pt')
model.export(format='onnx')
```

CLI:

```bash
yolo export model=best.pt format=onnx
```

지원 포맷: TorchScript, ONNX, OpenVINO, TensorRT 엔진, CoreML, TF SavedModel/TF Lite/EdgeTPU, TF.js, Paddle, MNN, NCNN, Sony IMX, RKNN 등(일부 옵션별 인자 지원). citeturn5view0turn4view0

---

## 14. 추적(Track) 모드 빠른 예시

YOLO11 Detect/Seg/Pose 모델은 추적 모드와 결합해 멀티 객체 추적(MOT)을 구현할 수 있습니다.

```bash
yolo track model=yolo11s.pt source='video.mp4' tracker=bytetrack.yaml
```

(트래커 구성은 ByteTrack, StrongSORT 등 사용 가능; 상세는 Ultralytics Tracking 문서 참조.) citeturn4view0turn10view1

---

## 15. 초보자 체크리스트 & 자주 하는 실수

| 체크         | 설명                                  |
| ---------- | ----------------------------------- |
| 라벨 누락      | 이미지 수와 라벨 수 일치? 빈 .txt 허용(배경)?      |
| 클래스 인덱스    | 0부터 시작? data.yaml names와 정렬?        |
| 경로         | data.yaml 상대/절대 경로 오류 흔함            |
| 이미지 크기     | 너무 큰 경우 VRAM 부족 → `imgsz` 줄이기       |
| 배치 사이즈     | CUDA OOM 나면 batch↓ 또는 `--cache` 사용  |
| Epoch 수    | 작은 데이터는 100 epoch 과잉; Early stop 고려 |
| Conf / IoU | 추론 시 `conf=`, `iou=` 조정해 결과 제어      |

이 체크리스트는 커뮤니티에서 반복 보고된 초보자 이슈를 바탕으로 정리했습니다. citeturn10view1turn9view5

---

## 16. 하이퍼파라미터 핵심만!

일반적으로 다음부터 조정:

* `epochs`: 데이터 크기에 맞춰 조정.
* `imgsz`: 정확도 vs 속도 트레이드오프.
* `batch`: GPU 메모리 한계 내 최대.
* `lr0`, `lrf`: 학습률 스케줄.
* `augment`: 강한 증강은 소량 데이터에 유리하나 과적합 주의.
  상세 구성은 Ultralytics Configuration 문서를 참고. citeturn5view0turn4view0

---

## 17. 엣지 디바이스 최적화 팁 (Jetson / CPU)

* Nano/소형 GPU: `yolo11n.pt` 또는 `yolo11s.pt` 추천.
* INT8 / FP16 변환으로 속도 향상 (TensorRT, OpenVINO, TFLite).
* 파이프라인 경량화를 위해 이미지 해상도 축소, 배치=1 스트림 추론.
* Roboflow Inference 또는 Ultralytics export된 엔진 사용 시 배포 단순화. citeturn6view0turn10view0turn5view0

---

## 18. 아키텍처 깊이 보기 (C3k2, C2PSA, SPPF, PAFPN)

YOLO11 백본은 합성곱 블록과 **C3k2(Cross Stage Partial, kernel size=2 변형)** 블록을 번갈아 쌓으며 총 5회 다운샘플링을 수행합니다. 마지막 피처는 SPPF(Spatial Pyramid Pooling-Fast)로 전송되어 가변 크기 입력을 처리합니다. citeturn9view3turn9view4

또한 **C2PSA / C2PSA(Convolutional block with Parallel Spatial Attention)** 모듈이 도입되어 관심 영역에 더 집중하도록 공간 주의(spatial attention)를 강화합니다. 일부 문서에서는 C2PSA 또는 C2PSA로 표기되며, 경량성을 유지하면서 특징 강조에 기여합니다. citeturn9view4turn6view0

넥(Neck)은 PAFPN(Path Aggregation FPN) 아이디어를 채용해 상향(bottom-up) + 하향(top-down) 피처 융합을 수행, 다중 스케일 정보를 효과적으로 결합합니다. citeturn9view3

이러한 모듈화와 경량 설계가 YOLOv8 대비 파라미터 수 감소와 정확도 개선의 핵심 배경으로 지목됩니다. citeturn10view1turn9view2

---

## 19. 모델 사이즈 고르기 가이드

간단한 권장:

* **n (Nano)**: 임베디드/Jetson Nano/라즈베리파이, 실시간 요구 낮음.
* **s (Small)**: 엣지 + 적당한 정확도.
* **m (Medium)**: 대부분 프로젝트의 출발점; 정확도/속도 균형.
* **l (Large)**: 더 높은 정확도, GPU 자원 충분할 때.
* **x (X-Large)**: 연구/최고 성능; 자원 충분 시.
  모델 크기별 용도는 LearnOpenCV 튜토리얼 및 Ultralytics 문서에서 정리됨. citeturn9view2turn3view0

---

## 20. 학습 중단/재개 & 실험 관리

Ultralytics는 기본적으로 `runs/detect/train*` 폴더에 체크포인트(`last.pt`, `best.pt`)를 저장합니다. 학습을 중단했다면 `model=path/to/last.pt`로 재개할 수 있습니다. (훈련 설정은 모델 객체에 저장되므로 val/predict 시 반복 지정 불필요.) citeturn5view0

---

## 21. FAQ

**Q1. YOLO11 기본 가중치는 어떤 데이터로 학습되었나요?**
A. Detect/Seg/Pose는 COCO, Classify는 ImageNet. citeturn5view0turn3view0

**Q2. 라이선스는? 상업적 사용 가능?**
A. 코드 AGPL-3.0; Ultralytics / 파트너(예: Roboflow) 경로를 통해 상용 라이선싱 가능. citeturn9view5turn4view0turn10view0

**Q3. 얼마나 가벼운가?**
A. YOLO11n \~2.6M 파라미터 수준, JPEG 크기 정도라고 표현됨; YOLO11 전 범위에서 v8보다 파라미터 절감 및 mAP 향상 보고. citeturn10view0turn10view1

**Q4. 내 데이터셋으로 학습하려면?**
A. YOLO 포맷 라벨 준비 → data.yaml 작성 → `yolo detect train ...` 실행. citeturn5view0turn9view5

**Q5. Edge 디바이스에서 돌아가나?**
A. Nano/Small 모델 + TensorRT/ONNX 최적화 사용; Roboflow Inference로 간단 배포 가능. citeturn6view0turn10view0

---

## 22. 참고자료

* Ultralytics **YOLO11 공식 문서**(모델, 특징, 성능, 태스크). citeturn3view0
* Ultralytics **Object Detection / Train / Predict 문서**(CLI & Python 예시, Export 포맷 등). citeturn5view0
* Ultralytics **GitHub 저장소**(설치, 사용 예시, 전체 생태계). citeturn4view0
* Ultralytics 블로그 *All you need to know about YOLO11* (출시, 파라미터 규모, 응용). citeturn10view0
* Ultralytics 커뮤니티 *YOLO11 Released* (mAP 개선, 파라미터 절감, 벤치마크). citeturn10view1
* Roboflow 블로그 *What is YOLOv11?* (라벨 포맷, 라이선스, 배포, Roboflow Inference). citeturn9view5
* PLOS One 연구 *Enhanced YOLO11 for Drone SAR* (C3k2, C2PSA 모듈). citeturn9view4
* MDPI *Improved Space Object Detection Based on YOLO11* (C3k2 스택, SPPF, PAFPN). citeturn9view3
* LearnOpenCV *YOLO11 Tutorial* (모델 크기 개요, 응용). citeturn9view2

---

### 업데이트 노트

이 문서는 Asia/Seoul 기준 2025-07-17 시점 최신 공개 정보를 반영했습니다. 이후 패키지/문서 업데이트로 세부 수치가 변할 수 있습니다. 필요하면 `pip show ultralytics` 또는 공식 문서를 다시 확인하세요.
