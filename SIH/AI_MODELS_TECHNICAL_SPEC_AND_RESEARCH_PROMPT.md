# AI & Computer Vision Models: Technical Specifications & Research Prompt
**Project: IBVAP (Intelligent Border Video Analytics Platform) / SIH PS-26187**

---

## Part 1: Architecture Overview & Current Model Inventory

Frigate operates a multi-stage, cascading inference pipeline:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              FRIGATE AI INFERENCE PIPELINE                             │
└────────────────────────────────────────────────────────────────────────────────────────┘

[ Video Stream (RTSP / MP4) ] ──► [ OpenCV Motion Gating (absdiff + dynamic contrast) ]
                                                │ (Only when motion detected)
                                                ▼
                             ┌──────────────────────────────────────┐
                             │ 1. PRIMARY OBJECT DETECTOR           │
                             │    • YOLOv8 / RT-DETR / D-FINE       │
                             └──────────────────┬───────────────────┘
                                                │ (Bounding Box Crops)
        ┌───────────────────────────────────────┼────────────────────────────────────────┐
        ▼                                       ▼                                        ▼
┌───────────────────────────────┐ ┌──────────────────────────────┐ ┌─────────────────────────────┐
│ 2. FACE RECOGNITION PIPELINE  │ │ 3. ANPR / LPR PIPELINE       │ │ 4. SEMANTIC / VISUAL SEARCH │
│ • Stage A: Landmark Alignment │ │ • Stage A: Plate Detector    │ │ • Jina CLIP v1/v2 Embedder  │
│ • Stage B: ArcFace/FaceNet    │ │ • Stage B: PaddleOCR PP-OCR  │ │ • Text & Image Vector DB    │
└───────────────────────────────┘ └──────────────────────────────┘ └─────────────────────────────┘
```

---

## Part 2: Technical Specifications of Current Models

### 1. Primary Object Detection (Human, Vehicle, Truck, Bus)

| Parameter | Current Specification |
|---|---|
| **Current Models** | `yolov8n.onnx` (Nano), `dfine`, `rfdetr` (RT-DETR), `yolox`, `ssd` |
| **Input Shape** | `[1, 3, 320, 320]` (or `[1, 3, 640, 640]`) |
| **Tensor Layout** | `NCHW` (Batch, Channels, Height, Width) |
| **Color Format** | RGB |
| **Data Type & Normalization** | `float32`, normalized to `[0.0, 1.0]` |
| **Output Tensor** | `[1, 84, 2100]` for YOLOv8 (4 bbox coords `[cx, cy, w, h]` + 80 class confidence scores) |
| **Post-Processing** | Built-in Non-Maximum Suppression (NMS) + Kalman spatial tracker |
| **Input Expected** | Whole cropped motion regions or downscaled full video frames |

---

### 2. Automatic Number Plate Recognition (ANPR / LPR)

| Parameter | Current Specification |
|---|---|
| **Pipeline Architecture** | **Two-Stage Cascaded:** Plate Localization &rarr; Text OCR Recognition |
| **Stage 1 (Plate Detector)** | `YOLOv9-LPR` ONNX (`yolov9-small.onnx` / `yolov9-large.onnx`) |
| **Plate Detector Input** | `[1, 3, 640, 640]` RGB, `float32` `[0.0, 1.0]` cropped from `car`/`motorcycle` |
| **Plate Detector Output** | Bounding box of the license plate |
| **Stage 2 (Text Recognition)** | `PaddleOCR` ONNX (`detection_v5-small.onnx`, `recognition_v5-small.onnx`) |
| **OCR Input Shape** | `[1, 3, 48, 320]` (standard PP-OCR text line crop, normalized to `[-1.0, 1.0]`) |
| **OCR Output** | CTC Greedy Decoder output &rarr; alphanumeric string + per-character probability |
| **Indian Plate Regex** | `^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$` with fuzzy Levenshtein distance matching |

---

### 3. Face Detection & Recognition

| Parameter | Current Specification |
|---|---|
| **Pipeline Architecture** | **Two-Stage Cascaded:** Landmark Detection/Alignment &rarr; Deep Feature Embedding |
| **Stage 1 (Alignment)** | OpenCV LBF Facemark (`landmarkdet.yaml`) / OpenCV DNN Face Detector |
| **Alignment Output** | 5 facial keypoints (eyes, nose, mouth corners) &rarr; Affine Warp (`cv2.warpAffine`) |
| **Stage 2 (Small Model)** | **FaceNet** (`facenet.tflite`): Input `[1, 160, 160, 3]`, RGB, `[-1.0, 1.0]` |
| **Stage 2 (Large Model)** | **ArcFace / InsightFace** (`arcface.onnx`): Input `[1, 3, 112, 112]`, RGB, `[-1.0, 1.0]` |
| **Embedding Output** | **512-dimensional L2-normalized float vector** |
| **Matching Algorithm** | Cosine distance against SQLite saved identity vector centroid (`min_score: 0.60`) |

---

### 4. Semantic / Visual Search (Natural Language Queries)

| Parameter | Current Specification |
|---|---|
| **Current Model** | **Jina CLIP v1 / v2** ONNX (`jina_v1_embedding.py` / `jina_v2_embedding.py`) |
| **Image Input** | `[1, 3, 224, 224]`, ImageNet mean/std normalized |
| **Text Input** | Tokenized phrase (`WordPiece` / `BPE`), max sequence length 64 tokens |
| **Output** | Joint **512-d or 768-d multimodal embedding vector** stored in `sqlite-vec` |
| **Function** | Allows typing "person in yellow raincoat carrying box" to query past thumbnails |

---

```
========================================================================================
             COPY & PASTE RESEARCH PROMPT FOR YOUR ML / RESEARCH TEAM
========================================================================================
```

# AI Research & Model Benchmark Brief
**Target System:** High-Throughput Edge Video Analytics Platform (IBVAP / Frigate Architecture)  
**Deployment Constraints:** Edge Intel Core i3/i5/i7, Intel UHD/Iris Xe (OpenVINO), NVIDIA Jetson / RTX GPU (TensorRT), Google Coral TPU.

### Objective
Our platform uses modular ONNX neural network pipelines to run real-time surveillance analytics on CCTV streams. We need you to identify, benchmark, and evaluate state-of-the-art (SOTA) 2025/2026 models that deliver **higher accuracy, better low-light robustness, and lower latency** than our current baseline models.

---

### Task 1: Primary Object Detection & Tracking (Human / Vehicle)
* **Current Baseline:** YOLOv8n ONNX (`[1, 3, 320, 320]` and `[1, 3, 640, 640]`, RGB, `float32`).
* **Evaluation Targets:**
  1. **YOLO11 / YOLOv10 ONNX:** NMS-free architectures for faster inference.
  2. **D-FINE (D-FINE-N / D-FINE-S):** SOTA real-time bounding box regressor.
  3. **RT-DETR v2:** Real-time transformer detector for dense border perimeters.
* **Requirements:**
  * Must export cleanly to ONNX opset 17+ with static/dynamic batch shape `[1, 3, H, W]`.
  * Quantizable to INT8/FP16 with minimal mAP drop.
  * Evaluate mAP@50-95 specifically on low-resolution, distant surveillance targets (people < 30px height).

---

### Task 2: High-Speed License Plate Recognition (ANPR / LPR)
* **Current Baseline:** YOLOv9 Plate Detector (`640x640`) + PaddleOCR PP-OCRv3/v5 (`48x320` input).
* **Pain Points to Solve:** Blurry plates from moving vehicles, high oblique angles (up to 45°), dirty/faded Indian High-Security Registration Plates (HSRP), night headlight glare.
* **Evaluation Targets:**
  1. **Plate Localization:** YOLO11-LPR or Edge-SAM tiny plate segmenter.
  2. **Plate Super-Resolution:** Fast real-time super-resolution (e.g., Real-ESRGAN-Compact / SwinIR-light) for low-pixel plate crops.
  3. **OCR Engine:** **PP-OCRv4 / PP-OCRv5**, **ParseQ** (SOTA permutation-autoregressive scene text reader), or **SVTR-v2**.
* **Requirements:**
  * Must output character sequence and confidence scores per character.
  * Zero reliance on cloud APIs; must run entirely on local ONNX runtime in < 15ms.

---

### Task 3: Face Detection, Landmark Alignment & Recognition
* **Current Baseline:** OpenCV LBF Landmark detector + ArcFace ResNet50 (`112x112` input &rarr; 512-d embedding).
* **Pain Points to Solve:** Extreme yaw/pitch face angles (security camera looking down from 4m height), low illumination, motion blur, partial occlusion (masks, sunglasses, turbans).
* **Evaluation Targets:**
  1. **Face Detector & Alignment:** **SCRFD** (Sample and Computation Redistribution for Efficient Face Detection) or **RetinaFace-MobileNet0.25**.
  2. **Face Feature Embedder:** **AdaFace** (Adaptive Margin Loss, designed specifically for low-quality / blurry surveillance video), **GhostFaceNetV2** (ultra-lightweight SOTA), or **InsightFace ArcFace-r100**.
* **Requirements:**
  * Input tensor: `[1, 3, 112, 112]` RGB normalized.
  * Output tensor: L2-normalized 512-dimensional embedding vector compatible with cosine similarity.

---

### Task 4: Missing Intelligence Capabilities (New Modules to Research)
1. **Human Pose & Skeleton Estimation (Suspicious Behavior):**
   * *Target:* **YOLOv8-Pose / YOLO11-Pose** or **RTMPose**.
   * *Input:* `[1, 3, 256, 192]` person crop.
   * *Output:* 17 COCO keypoints `[x, y, confidence]` for detecting crouching, fence climbing, running, or fighting.
2. **Cross-Camera Person Re-Identification (Re-ID):**
   * *Target:* **OSNet (Omni-Scale Network)**, **FastReID**, or **TransReID**.
   * *Input:* `[1, 3, 256, 128]` full-body person crop.
   * *Output:* 512-d appearance embedding vector to track the same unknown individual across multiple non-overlapping border cameras.

---

### Deliverables Required from Research Team:
1. Exported `.onnx` model files (FP32 and INT8/FP16 quantized).
2. Accuracy benchmark table (mAP, Top-1 accuracy, and F1-score) on standard surveillance datasets (Market-1501, WiderFace, LFW, CCPD/Indian-Plate).
3. Latency benchmarks (milliseconds per frame) tested across:
   * Intel CPU (OpenVINO ONNX provider)
   * NVIDIA TensorRT (FP16 engine)
   * Google Coral Edge TPU (INT8 TFLite)
