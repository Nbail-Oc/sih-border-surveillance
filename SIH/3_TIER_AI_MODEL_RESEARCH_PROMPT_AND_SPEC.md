# 3-Tier Edge-to-Cloud AI Model Zoo: Research Specification & Prompt
**Project: IBVAP (Intelligent Border Video Analytics Platform) / SIH PS-26187**
**For:** Ministry of Home Affairs (MHA) & Sashastra Seema Bal (SSB)

---

## Part 1: Current Frigate Baseline vs. Research Upgrades

### Why Frigate Uses Its Current Baseline:
Frigate was designed for low-power consumer Home Assistant setups (Raspberry Pis & Coral TPUs), using lightweight models that sacrifice high-end military surveillance accuracy for minimum power draw:
1. **Object Detection:** `yolov8n.onnx` (`320x320`, 3.2M params) &rarr; Fast (10ms on CPU), but drops small/distant human targets ($<30\text{px}$) and carries restrictive **AGPL-3.0** licensing.
2. **ANPR / LPR:** `YOLOv9-LPR` (`640x640`) + `PaddleOCR PP-OCRv3` (`48x320`) &rarr; Functional on flat plates, but fails on severe 45° perspective skews, night headlight glare, and degraded Indian HSRP plates.
3. **Face Recognition:** `cv2.face` LBF Landmark + `ArcFace ResNet50` (`112x112`) &rarr; Uses non-commercial research weights (`buffalo_l`), degrades under active-IR noise, and lacks demographic tuning for South Asian faces.
4. **Behavior / Pose / Re-ID:** **Not present in Frigate core** &rarr; Must be built as external event-driven microservices.

---

## Part 2: The 3-Tier Operational Hardware Framework

Our deployment strategy covers three distinct military operational tiers:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        3-TIER OPERATIONAL HARDWARE PROFILES                            │
├───────────────────────┬────────────────────────────────┬───────────────────────────────┤
│ TIER 1: LOW (Edge)    │ TIER 2: MEDIUM (Station)       │ TIER 3: HIGH (Command HQ)     │
│ [🟢 BUILD NOW - DEMO] │ [🟡 ROADMAP / SPEC ONLY]       │ [🟡 ROADMAP / SPEC ONLY]       │
├───────────────────────┼────────────────────────────────┼───────────────────────────────┤
│ • Remote Border Post  │ • Battalion Checkpost          │ • Central Command Hub         │
│ • Intel i3 / N100 CPU │ • Intel i5/i7 + Iris Xe iGPU   │ • Enterprise Server Rack      │
│ • Coral Edge TPU      │ • NVIDIA Jetson Orin / RTX3050 │ • Multi-GPU (RTX 4090/A5000)  │
│ • Power: 15–25 Watts  │ • Power: 45–100 Watts          │ • Power: 450+ Watts           │
│ • Stream Cap: 2–4 Cams│ • Stream Cap: 8–16 Cams        │ • Stream Cap: 32+ 4K Cams     │
│ • Latency Cap: <15ms  │ • Latency Cap: <35ms           │ • Latency Cap: <80ms (Batched)│
└───────────────────────┴────────────────────────────────┴───────────────────────────────┘
```

> **Operational Scope Definition:**  
> **Tier 1 (Edge) is the ONLY tier being implemented and benchmarked hands-on for the working prototype/demo.**  
> Tier 2 and Tier 3 are researched and documented as the **Scalability Architecture Section** in our official submission report (model names, hardware profiles, expected accuracy gains) — not built hands-on, not blocking the demo timeline.

---

```
========================================================================================
             COPY & PASTE RESEARCH PROMPT FOR YOUR ML / RESEARCH TEAM
========================================================================================
```

# High-Precision AI Model Zoo Research Brief: 3-Tier Multi-Modal Surveillance
**Target Agency:** Ministry of Home Affairs (MHA) / Sashastra Seema Bal (SSB)  
**Primary Objective:** Deliver a legally compliant, high-accuracy model zoo for edge-to-cloud border surveillance across 3 operational tiers.

---

### RULE 1: Hard Licensing Gate (AGPL-3.0 & Non-Commercial Prohibition)
* **Strict Restriction:** Any Ultralytics-family model (**YOLOv8, YOLOv10, YOLO11, YOLO26, YOLO-World v2, YOLOv8-Pose/Face**) is governed by **GNU AGPL-3.0**. Under AGPL-3.0, deploying this software requires open-sourcing the *entire connected application stack, proprietary scripts, and internal configs* unless an expensive enterprise license is acquired. Government/internal deployments are not exempt.
* **Standing Directive:** **Do NOT select AGPL-3.0 or non-commercial research models (e.g. InsightFace `buffalo_l`) as default recommendations.** 
* Default strictly to permissively licensed architectures (**Apache-2.0, MIT, BSD**). D-FINE and RT-DETRv2 already cover detection across all tiers, and RTMPose (OpenMMLab, Apache-2.0) covers pose. Only surface a YOLO variant if zero viable Apache/MIT alternative exists, and flag it explicitly as a licensing risk.

---

### RULE 2: Sensor Scope Definition (Active-IR CCTV vs. True Thermal)
* **Scope Clarification:** The problem statement specifies **existing CCTV infrastructure**, which consists of standard visible-light IP cameras equipped with **built-in IR-LED illumination (monochrome low-light video)**, *not* expensive Long-Wave Infrared (LWIR) / FLIR thermal sensors.
* Do not evaluate models on thermal-specific datasets (e.g., FLIR ADAS). Evaluate candidates strictly on **monochrome low-light, IR-illuminated grain, headlight glare, and distance blur**.

---

### RULE 3: Implementation Tiering & Tensor Adapter Engineering
* **Tier 1 (Build Now / Demo):** Prefer models adhering to standard tensor interfaces (`[1, 3, H, W]` RGB, `[1, 3, 112, 112] -> 512-d` for face, etc.) to minimize custom C++/Python adapter overhead on a hackathon timeline. Custom adapters are acceptable only if the accuracy gain is substantial.
* **Tier 2 & Tier 3 (Roadmap / Architecture):** Exotic architectures (Vision Transformers, Neural Super-Resolution, Multi-Stage Ensembles) are encouraged without implementation penalty since they serve as the theoretical scalability roadmap.

---

### Task 1: Primary Object Detection & Tracking (Human / Vehicle / Asset)
*Evaluate models capable of detecting distant humans (<25px tall) and vehicles across open border terrain in daylight and night IR monochrome feeds.*

* **Tier 1 (Low / Edge CPU) — [BUILD NOW]:**
  * *Target Specs:* Parameter count $\le 5\text{M}$, Latency $<15\text{ms}$ on Intel i3 / OpenVINO INT8.
  * *Primary Pick:* **D-FINE-N (Nano)** (Apache-2.0, NMS-free FDR, superior $<30\text{px}$ small-target retention).
  * *Alternate:* **MobileNetV4-Conv-S** / **YOLOX-Nano** (Apache-2.0).
* **Tier 2 (Medium / Edge GPU / Jetson) — [ROADMAP]:**
  * *Target Specs:* Parameter count $10\text{M}–25\text{M}$, Latency $<30\text{ms}$ on RTX 3050 / Jetson Orin.
  * *Candidates:* **D-FINE-S / D-FINE-M**, **RT-DETRv2-S / M** (Apache-2.0, Real-Time Detection Transformer).
* **Tier 3 (High / Command Server GPU) — [ROADMAP]:**
  * *Target Specs:* Maximum mAP@50-95, Latency $<50\text{ms}$ on RTX 4090 / TensorRT FP16.
  * *Candidates:* **D-FINE-X / D-FINE-L** (Apache-2.0, 54.0%+ COCO mAP), **RT-DETR-X**, **Co-DETR (Swin-L)**.

---

### Task 2: High-Speed License Plate Recognition (ANPR / LPR)
*Evaluate systems capable of resolving high oblique angles (up to 45°), night headlight glare, motion blur, and worn/dusty Indian High-Security Registration Plates (HSRP).*

* **Tier 1 (Low / Edge CPU) — [BUILD NOW]:**
  * *Pipeline:* YOLOv9-LPR / D-FINE Tiny Plate Detector + **OpenCV CLAHE (<0.8ms)** + **PP-OCRv5-Mobile / SVTR-v2-tiny** (Apache-2.0, $<12\text{ms}$ on CPU).
  * *Engineering Note:* Neural Super-Resolution (Real-ESRGAN/SwinIR) is rejected for Tier 1 due to 45–110ms CPU bottleneck; classical CLAHE contrast stretching is mandatory.
* **Tier 2 (Medium / Workstation GPU) — [ROADMAP]:**
  * *Pipeline:* D-FINE Plate Detector + Spatial Transformer Network (STN) + **ParseQ-Tiny** (Apache-2.0, Permutation Autoregressive Transformer).
* **Tier 3 (High / Command Server GPU) — [ROADMAP]:**
  * *Pipeline:* 3-Stage Heavy Pipeline:
    1. Plate Localization & 4-Corner Landmark Quad Detection (STN).
    2. **Real-ESRGAN-Compact / SwinIR-Light** (2x Neural Super-Resolution on low-pixel distant plates).
    3. **ParseQ-Large / TrOCR-Large** (Vision Transformer OCR with language prior error correction).

---

### Task 3: Facial Recognition & Alignment Under Night / Overhead Surveillance
*Evaluate systems for cameras mounted at 4-meter heights (downward angle $>35^\circ$), handling partial occlusions (masks/caps/scarves), and achieving demographic parity on South Asian / Indian faces.*

* **Tier 1 (Low / Edge CPU) — [BUILD NOW]:**
  * *Detector & Alignment:* **RetinaFace-MobileNet0.25** (MIT) or **SCRFD-500M** (Permissive weights).
  * *Feature Embedder:* **AdaFace-MobileFaceNet** / **GhostFaceNetV2** (MIT, ~6ms on CPU) or **AdaFace-IResNet50** (Quality-adaptive margin for IR noise).
* **Tier 2 (Medium / Workstation GPU) — [ROADMAP]:**
  * *Detector:* **SCRFD-2.5G** (5-point landmark alignment).
  * *Embedder:* **AdaFace (IResNet50 backbone)** (Apache-2.0 code / permissive weights, handles IR grain and overhead pitch).
* **Tier 3 (High / Command Server GPU) — [ROADMAP]:**
  * *Detector:* **SCRFD-34G** (High-precision occluded face localization).
  * *Embedder:* **AdaFace-ResNet100** (Trained on WebFace42M) + **SDD-FIQA** (Face Quality Assessment filter).

---

### Task 4: Perimeter Pose Estimation & Suspicious Behavior Analytics
*Evaluate human skeleton models for real-time classification of: fence-climbing, crouching/crawling, border breach, and weapon/carrying postures.*

* **Tier 1 (Low / Edge CPU) — [BUILD NOW]:**
  * **RTMPose-T (Tiny)** / **RTMPose-S** (OpenMMLab, Apache-2.0, 17 keypoints in ~6ms on CPU) + Real-time geometric heuristic rule engine (wrist-shoulder elevation, knee flexion).
* **Tier 2 (Medium / Workstation GPU) — [ROADMAP]:**
  * **RTMPose-M** (High keypoint accuracy under crowded and self-occluding postures).
* **Tier 3 (High / Command Server GPU) — [ROADMAP]:**
  * **RTMPose-L** / **ViTPose-Base** (Vision Transformer Pose) + **Spatial-Temporal Graph Convolutional Network (ST-GCN / 2s-AGCN)** for complex action recognition.

---

### Task 5: Multi-Camera Person Re-Identification (Re-ID)
*Evaluate appearance feature extractors to establish persistent identity trails across non-overlapping border cameras.*

* **Tier 1 (Low / Edge CPU) — [BUILD NOW]:**
  * **OSNet-x0_25 / OSNet-x0_5** (Omni-Scale Network, MIT, ~8ms on CPU, 512-d vector).
* **Tier 2 (Medium / Workstation GPU) — [ROADMAP]:**
  * **OSNet-IBN-x1_0** (Instance-Batch Normalization for cross-camera lighting invariance) / **FastReID-ResNet50**.
* **Tier 3 (High / Command Server GPU) — [ROADMAP]:**
  * **TransReID (ViT-Base)** / **SOLIDER** (Self-supervised learned semantic representation) + Short-term FAISS GPU Vector Index.

---

## Required Deliverables from Research Team:
1. **Verifiable Citations:** Every specific accuracy figure (mAP, AUROC, Top-1, character accuracy, parameter count) must cite the specific paper, model card, or official benchmark repository. If a metric lacks an official peer-reviewed citation, mark it explicitly as **"unverified — needs confirmation"**.
2. **Tiered Recommendation Table:** Detail Model Name, Paper Citation, License (Apache-2.0/MIT/BSD only), Benchmark Scores, and Parameter Count.
3. **Export Artifacts for Tier 1:** Deliver validated `.onnx` or OpenVINO IR files for Tier 1 candidate models.
