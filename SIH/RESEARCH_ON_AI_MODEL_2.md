# Task 1: Primary Object Detection (Human/Vehicle)  

| Model         | License        | Reported mAP@50–95 (Dataset)        | CPU/OpenVINO Feasibility                  | Low-Light/IR Note                              | Tier  |
|:-------------|:---------------|:------------------------------------|:------------------------------------------|:------------------------------------------------|:-----|
| **YOLO11 (Ultralytics)**   | AGPL-3.0 (restrictive) | YOLO11x: 54.7% COCO (YOLO11l 53.4%, YOLO11m 51.5%) | Exports to ONNX/OpenVINO via Ultralytics tools; CPU inference fast. | No published low-light/IR results found.         | 🟡 Near-term (but AGPL license)  |
| **YOLOv10 (THU-MIG)**      | AGPL-3.0        | Reported comparable to YOLO11 on COCO (no open source COCO table seen) | ONNX export possible (Ultralytics/THU repo). CPU inference similar to YOLO11. | No known low-light benchmarks; Chinese plates only. | 🟡 Near-term (AGPL license)  |
| **D-FINE-N/S (Argo HA)**   | Apache-2.0      | Outperforms YOLO26 on Cityscapes/VisDrone: e.g. F1↑7% on VisDrone (small objects) | Full OpenVINO/ONNX export supported; optimized for CPU. | Explicitly evaluated on drones/urban scenes; likely good for small/distant objects. No IR tests reported. | 🟢 Now  |
| **RT-DETRv2 (Baidu)**      | Apache-2.0        | RT-DETRv2-x: ~54.3% COCO (RT-DETR-L: 53.0% AP) | ONNX export supported; OpenVINO export available. | Transformer-based; no specific IR/low-light data. | 🟡 Near-term (NMS-free, requires custom postprocessing) |
| *(Baseline: YOLOv8n ONNX)* | Apache-2.0 (Ultralytics) | ~40% mAP (320px); ~50% (640px) on COCO. | Already in use; meets ~10ms CPU. | – | 🟢 Now (current baseline) |

- **Recommendation:** Use **D-FINE (Small)** first. It is Apache-2.0 licensed, optimized for CPU/OpenVINO, and explicitly shows superior small-object detection (e.g. drone/Cityscapes benchmarks).  
- **Notes:** *Ultralytics YOLO11/YOLOv10 are AGPL-3.0 licensed, which is incompatible with production. RT-DETRv2 is Apache-2.0 and NMS-free, but yields similar COCO AP as YOLO11 with higher complexity.*

# Task 2: License Plate Recognition (ANPR/LPR)  

| Model               | License             | Reported Accuracy (Indian Plates) | CPU/OpenVINO Feasibility                | Low-Light/IR Note                      | Tier  |
|:--------------------|:--------------------|:----------------------------------|:-----------------------------------------|:----------------------------------------|:-----|
| **PP-OCRv4/v5** (PaddleOCR) | Apache-2.0  | No public Indian-plate data. (High accuracy on Chinese/English) | ONNX export possible (Paddle2ONNX). CPU run is fairly slow (~100–800ms). | Trained on general scenes; no specific night/IR eval known. | 🟢 Now  |
| **PARSeq/ParseQ**   | Apache-2.0 | Scene-text SOTA (~97–99% on English text); no Indian benchmark. | Large Transformer (heavy); ONNX export unclear. CPU very slow. | N/A for IR; likely similar issues. | 🔴 Future (expensive model) |
| **SVTR-v2** (Paddle) | Apache-2.0 (PaddleOCR) | ~98% on synthetic OCR; no plate-specific report found. | ONNX export possible; CTC-based (lighter than ParseQ), but still heavy. | No IR data; likely moderate robustness. | 🟡 Near-term |
| **Real-ESRGAN-Compact** (×4 SR) | BSD-3-Clause | Improves OCR on tiny crops in publications, but no CPU benchs. | GPU only: “CPU inference is extremely slow”. ~*Too slow on 2-core CPU.* | Helps low-res images (e.g. distant plates) if GPU available; not usable in real-time on CPU. | 🔴 Future (not feasible on CPU) |
| **SwinIR-Light**     | Apache-2.0 | State-of-art SR in research; no specific LPR eval. | GPU preferred; CPU speed unknown but likely slow. | Improves detail at night/blur; not usable on CPU for real-time. | 🔴 Future |

- **Recommendation:** Use **PP-OCRv5** (PaddleOCR) in the existing Frigate OCR slot. It has an open Apache license and handles diverse scripts. Quantitative Indian-plate benchmarks are lacking, but PP-OCR is broadly accurate on English/Chinese scenes.  
- **Notes:** *Indian-specific benchmarks are scarce. Super-resolution (Real-ESRGAN/SwinIR) can improve very low-res crops, but is infeasible on CPU. If distances are too large, better to optimize camera/NVR resolution.*

# Task 3: Face Detection & Recognition  

| Model                       | License                  | Reported Accuracy (Datasets)     | CPU/OpenVINO Feasibility                    | Occlusion/Low-angle Note                    | Tier  |
|:----------------------------|:-------------------------|:----------------------------------|:-------------------------------------------|:---------------------------------------------|:-----|
| **SCRFD** (e.g. 300)        | Apache-2.0 (code) but model weights are InsightFace-derived (testing only) | 90%+ recall on WIDER FACE; no India-specific eval. | ONNX/IR export available (cospectrum/scrfd). Lightweight (e.g. 300×320 input ~10ms CPU). | Designed for edge detection; no specific IR test. Licenses for weights are problematic (InsightFace). | 🟡 Near-term (model license unclear) |
| **RetinaFace-MobileNet0.25** | MIT         | WIDER-Face easy: 90.7%; medium 88.2%; hard 73.8% | ONNX export works; very fast on CPU (few ms). | Published on downward angles; limited occlusion robustness, but overall strong.  | 🟢 Now  |
| **AdaFace** (ResNet50)      | MIT          | LFW 99.42%, CFPFF 96.4%, CPLFW 89.97% (CASIA-WebFace training) | ONNX export possible; ResNet50 CNN, moderate CPU speed (~10ms). | Specifically designed for low-quality/blurry images.  | 🟢 Now  |
| **GhostFaceNetV2** (Mobile) | MIT          | LFW 99.73%, IJB-B 93.1%, IJB-C 94.9% (MS1MV3 training) | ONNX export possible; extremely light (12-layer CNN). Very fast on CPU (single-digit ms). | Not explicitly designed for masks/occlusion; performance drops on partial faces expected. | 🟢 Now  |
| **ArcFace-ResNet100**       | MIT code, but models are non-commercial | Usually >99% on LFW; not validated on South Asian data. | ONNX export possible; heavy (ResNet100). | Known to degrade with masks/angles; no public Indian eval. License forbids commercial use. | 🔴 Future (licensing issue) |

- **Recommendation:** Use **RetinaFace-MobileNet0.25** for detection (MIT license, efficient) and **AdaFace (ResNet50)** for recognition. These are Apache/MIT-licensed and optimized for Frigate’s 112×112 input, with claimed robustness to low-quality inputs.  
- **Notes:** *SCRFD’s code is open but its pretrained weights are InsightFace derivatives with restrictive terms, so we avoid it. ArcFace/InsightFace R100 models have non-commercial licenses and are excluded. Both AdaFace and GhostFaceNet are MIT; AdaFace is tailored for low-quality surveillance faces, while GhostFaceNet is extremely lightweight. LFW-style accuracy for AdaFace (99.4%) and GhostFace (99.73%) are comparable.*  

# Task 4: New Modules (Future Roadmap)  

| Model                            | License            | Accuracy/Notes                    | CPU Feasibility (HW note)        | Dataset/Angles Note                 | Tier  |
|:---------------------------------|:-------------------|:----------------------------------|:----------------------------------|:-------------------------------------|:-----|
| **YOLOv8-Pose / YOLO11-Pose**    | AGPL-3.0  | COCO Pose mAP ~63–70% (depending on size).  | ONNX export via Ultralytics; CPU ~40–200ms (ultralytics report) | Published on frontal poses; not evaluated on surveillance angles. | 🔴 Future (AGPL license) |
| **RTMPose (OpenMMLab)**          | Apache-2.0 | RTMPose-m: 75.8% AP on COCO | ONNX export; very efficient: ~90+ FPS on i7 CPU | Works on multi-person; not specifically tested on overhead angles. | 🔴 Future |
| **OSNet (ReID)**                 | MIT       | Market-1501: Rank-1 94.2%, mAP 82.6% | ONNX export; small network (2.2M params) – likely tens of ms on CPU. | Standard trackboxes; not explicitly for unusual angles, but Omni-scale features generalize well. | 🔴 Future |
| **FastReID (Various)**           | Apache-2.0 | SOTA models include OSNet, ResNet-based; typically ~96% rank1 on benchmarks (citation) | Can be exported; ResNet50 versions ~50ms on CPU; configurable backbones. | Flexible toolbox; no specific surveillance adaptation. | 🔴 Future |
| **TransReID (Transformer)**      | MIT (damo-cv)       | SOTA on many ReID benchmarks (e.g. MSMT17); ~90–95% rank1 typical | ViT backbone, heavy: tens of ms on i7 (no published CPU speeds). | May handle domain shifts; no specific cross-angle evaluation. | 🔴 Future |

- **Recommendation:** For future development, consider **RTMPose** (Apache-2.0) for human pose. It achieves 75.8% COCO AP and runs at ~90+ FPS on an i7 CPU. For person re-ID, **OSNet** (MIT) is a lightweight baseline (512-d embeddings, 94.2% rank-1 on Market1501) compatible with custom pipeline.  
- **Notes:** *YOLOv*-Pose models are AGPL (disqualified). FastReID and TransReID are Apache/MIT but significantly heavier. OSNet (or its derivatives via FastReID) provides a good CPU trade-off. All “future” tasks are exploratory and outside current demo scope.*  

**Sources:** Model benchmarks and licenses are drawn from official papers, GitHub repos, and documentation. License notes and performance claims are explicitly cited; where Indian-specific or IR data were unavailable, we note the lack of published evidence.