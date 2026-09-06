# Master Technical Integration & Testing Plan: 3-Tier AI Model Zoo (IBVAP / SIH PS-26187)
**Target Agencies:** Ministry of Home Affairs (MHA) & Sashastra Seema Bal (SSB)  
**System Integrity Mandate:** Zero code breakage, seamless config-driven model selection, and complete unit/integration test coverage.

---

## 1. Executive Summary & Hardware Topology

The IBVAP surveillance architecture supports three distinct deployment tiers. **Tier 1 (Edge CPU) is the active build-now demo target**, while **Tier 2 (Station GPU)** and **Tier 3 (Command HQ Server)** form the official scalability roadmap.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        3-TIER OPERATIONAL DEPLOYMENT ARCHITECTURE                      │
├───────────────────────┬────────────────────────────────┬───────────────────────────────┤
│ TIER 1: LOW (Edge)    │ TIER 2: MEDIUM (Station)       │ TIER 3: HIGH (Command HQ)     │
│ [🟢 BUILD NOW - DEMO] │ [🟡 ROADMAP / SPEC ONLY]       │ [🟡 ROADMAP / SPEC ONLY]       │
├───────────────────────┼────────────────────────────────┼───────────────────────────────┤
│ • Remote Border Post  │ • Battalion Checkpost          │ • Central Command Hub         │
│ • Intel i3 / N100 CPU │ • Intel i5/i7 + Iris Xe iGPU   │ • Enterprise Server Rack      │
│ • Coral Edge TPU      │ • NVIDIA Jetson Orin / RTX3050 │ • Multi-GPU (RTX 4090/A5000)  │
│ • Power: 15–25 Watts  │ • Power: 45–100 Watts          │ • Power: 450+ Watts           │
│ • Target Latency <15ms│ • Target Latency <35ms         │ • Target Latency <80ms        │
└───────────────────────┴────────────────────────────────┴───────────────────────────────┘
```

---

## 2. Audited Master 3-Tier Model Zoo

| Functional Module | Tier 1 (Edge CPU - BUILD NOW) | Tier 2 (Station GPU - ROADMAP) | Tier 3 (Command HQ - ROADMAP) | License Status |
|---|---|---|---|---|
| **Task 1: Object Detection** | **D-FINE-N (Nano)** (3.8M params, 11.2ms) | **D-FINE-S / M** / **RT-DETRv2-S** (10–20M params) | **D-FINE-X / L** / **Co-DETR (Swin-L)** (55.8%+ mAP) | ✅ **Apache-2.0** |
| **Task 2: ANPR / LPR** | **D-FINE-Tiny + CLAHE + PP-OCRv5-Mobile** | **D-FINE Plate + STN + ParseQ-Tiny** | **STN + Real-ESRGAN + ParseQ-Large / TrOCR** | ✅ **Apache-2.0** |
| **Task 3: Face Analytics** | **SCRFD-500M + GhostFaceNetV2** (512-d) | **SCRFD-2.5G + AdaFace (IResNet50)** | **SCRFD-34G + AdaFace-ResNet100 + SDD-FIQA** | ✅ **MIT / Apache-2.0** |
| **Task 4: Perimeter Pose** | **RTMPose-S + Geometric Rules** (17 keypoints) | **RTMPose-M + Joint Angles** | **RTMPose-L + ST-GCN Action Classifier** | ✅ **Apache-2.0** |
| **Task 5: Person Re-ID** | **OSNet-x0_25** (512-d vector, ~7.4ms) | **OSNet-IBN-x1_0** (Lighting-invariant) | **TransReID (ViT-Base) + FAISS GPU Index** | ✅ **MIT / Apache-2.0** |

---

## 3. Standardized Tensor Interface Seams (Zero-Breakage Architecture)

To ensure that switching models across Tier 1, Tier 2, and Tier 3 requires **zero modifications to Frigate's core Python backend**, we enforce strict contract adapters at five defined public seams:

```
                            [ CAMERA RAW FRAME ]
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
    ┌─────────────────────────┐             ┌─────────────────────────┐
    │ SEAM 1: Object Detector │             │ Background Event Stream │
    │ Output Contract:        │             └────────────┬────────────┘
    │ [N, 6] -> (x1,y1,x2,y2, │                          │ (MQTT Person / Plate Crops)
    │            score, class)│                          ▼
    └────────────┬────────────┘             ┌─────────────────────────┐
                 │                          │ SEAM 4: Pose Analytics  │
                 ▼                          │ Input: [1, 3, 256, 192] │
    ┌─────────────────────────┐             │ Output: 17 Keypoints    │
    │ Frigate Tracking Engine │             └─────────────────────────┘
    └────────────┬────────────┘                          │
                 │                                       ▼
                 ├──────────────────────────┬─────────────────────────┐
                 ▼                          ▼                         ▼
    ┌─────────────────────────┐┌─────────────────────────┐┌─────────────────────────┐
    │ SEAM 2: ANPR / OCR      ││ SEAM 3: Face Analytics  ││ SEAM 5: Person Re-ID    │
    │ Input: [1, 3, 48, 320]  ││ Input: [1, 3, 112, 112] ││ Input: [1, 3, 256, 128] │
    │ Output: (text, conf)    ││ Output: 512-d L2 Vector ││ Output: 512-d L2 Vector │
    └─────────────────────────┘└─────────────────────────┘└─────────────────────────┘
```

### Seam Specifications:

#### Seam 1: Object Detector (`frigate/detectors/plugins/onnx.py`)
* **Input:** `[1, 3, H, W]` normalized float tensor ($[0, 1]$ or $[-1, 1]$).
* **Output Adapter:** Maps raw model logits (NMS-free distribution vectors in D-FINE or RT-DETR) into a standardized `numpy.ndarray` of shape `[N, 6]` representing `[x_min, y_min, x_max, y_max, confidence, class_id]`.
* **Frigate Compatibility:** Frigate natively supports `model_type: dfine` and `model_type: yolo-generic` via [`frigate/util/model.py`](file:///c:/Users/thang/Downloads/frigate-dev/frigate-dev/frigate/util/model.py).

#### Seam 2: License Plate OCR (`frigate/embeddings/onnx/lpr_embedding.py`)
* **Input:** `[1, 3, 48, 320]` plate crop.
* **Pre-processing:** Classical OpenCV CLAHE tile equalization ($8 \times 8$ grid, clip limit $2.0$) executes in $<0.8\text{ms}$ on CPU before neural inference.
* **Output Adapter:** Decodes CTC / autoregressive character logits into `(plate_text: str, confidence: float)`.

#### Seam 3: Face Recognition (`frigate/embeddings/onnx/face_embedding.py`)
* **Input:** `[1, 3, 112, 112]` aligned BGR/RGB crop (via 5-point affine transformation from SCRFD / RetinaFace).
* **Output Adapter:** Invariant 512-dimensional L2-normalized float32 vector $\vec{z} \in \mathbb{R}^{512}$ where $\Vert \vec{z} \Vert_2 = 1.0$. Drop-in replacement for ArcFace/FaceNet.

#### Seam 4: Pose Estimation (`services/pose_service.py` - Standalone Microservice)
* **Input:** `[1, 3, 256, 192]` person bounding-box crop consumed via MQTT event bus (`frigate/events`).
* **Output:** $\mathbf{K} \in \mathbb{R}^{17 \times 3}$ containing $(x, y, \text{confidence})$ per joint &rarr; Evaluated against deterministic geometric heuristics for fence climbing and crouching.

#### Seam 5: Cross-Camera Re-ID (`services/reid_service.py` - Standalone Microservice)
* **Input:** `[1, 3, 256, 128]` full-body crop.
* **Output:** 512-dimensional appearance embedding &rarr; Indexed into an in-memory spatial-temporal vector database (TTL: 15 minutes).

---

## 4. Config-Driven Model Selection (No Code Modification)

Field operators can switch between model tiers purely by changing parameters in `config/config.yml`:

```yaml
# ==============================================================================
# IBVAP MULTI-TIER SURVEILLANCE CONFIGURATION
# ==============================================================================

model:
  # Tier 1 (Low / Edge CPU):
  path: /config/model_cache/dfine_n.onnx
  # Tier 2 (Station GPU): path: /config/model_cache/dfine_s.onnx
  # Tier 3 (Command HQ):  path: /config/model_cache/dfine_x.onnx
  model_type: dfine
  width: 320
  height: 320
  input_tensor: nchw
  input_dtype: float

detectors:
  primary_detector:
    type: onnx
    device: AUTO # Accelerates via OpenVINO VNNI / GPU on Intel or NVIDIA

face_recognition:
  enabled: true
  # Drop-in AdaFace / GhostFaceNetV2 embedding model
  model_path: /config/model_cache/facedet/adaface_ir50.onnx

lpr:
  enabled: true
  # PP-OCRv5 / ParseQ recognition engine
  model_path: /config/model_cache/lpr/ppocr_v5_rec.onnx
  enhancement: clahe # Enables sub-millisecond CLAHE for night IR feeds
```

---

## 5. Test-Driven Development (TDD) & Verification Plan

Following `/tdd` and `/karpathy-guidelines`, we test each seam independently using isolated unit and integration tests.

### Test Matrix:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              TDD VERIFICATION SUITE                                    │
├────────────────────────────┬─────────────────────────────┬─────────────────────────────┤
│ Test Module                │ Seam Under Test             │ Verifiable Success Criteria │
├────────────────────────────┼─────────────────────────────┼─────────────────────────────┤
│ test_dfine_adapter.py      │ Seam 1: Detection Output    │ Raw D-FINE logits parse     │
│                            │                             │ cleanly to [N, 6] array.    │
├────────────────────────────┼─────────────────────────────┼─────────────────────────────┤
│ test_lpr_clahe.py          │ Seam 2: LPR Pre-proc & CTC  │ CLAHE boosts low-contrast   │
│                            │                             │ plate; CTC decodes string.  │
├────────────────────────────┼─────────────────────────────┼─────────────────────────────┤
│ test_face_embedding.py     │ Seam 3: Face Embeddings     │ Output is exactly 512-d and │
│                            │                             │ L2-normalized (norm = 1.0). │
├────────────────────────────┼─────────────────────────────┼─────────────────────────────┤
│ test_pose_heuristics.py    │ Seam 4: Pose Rule Engine    │ Climbing heuristic triggers │
│                            │                             │ when wrist > shoulder plane.│
├────────────────────────────┼─────────────────────────────┼─────────────────────────────┤
│ test_reid_matching.py      │ Seam 5: Cosine Similarity   │ Same identity cosine > 0.75,│
│                            │                             │ different identity < 0.35.  │
└────────────────────────────┴─────────────────────────────┴─────────────────────────────┘
```

### Execution Commands:
```bash
# Run backend unit tests across all model adapter seams
python3 -u -m unittest frigate.test.test_dfine_adapter
python3 -u -m unittest frigate.test.test_lpr_clahe
python3 -u -m unittest frigate.test.test_face_embedding
```
