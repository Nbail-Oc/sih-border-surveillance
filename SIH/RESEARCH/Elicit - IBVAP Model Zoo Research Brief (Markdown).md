# IBVAP 3-Tier Edge-to-Cloud AI Model Zoo

## Executive assessment

The supplied specification has a sound deployment structure—implement Tier 1 first and treat station and command-hub systems as roadmap architecture—but several model claims need correction before they appear in an official submission. The strongest directly verified Tier 1 candidates in the retrieved evidence are D-FINE-N for general object detection and SCRFD-0.5GF/2.5GF for face detection. Their published benchmarks are COCO and WIDERFace results, not evidence of performance on active-IR border CCTV, distant targets below 25 px, or the proposed Intel i3/N100 OpenVINO INT8 target.

D-FINE is especially relevant because its official repository states Apache-2.0 licensing, provides an explicit model table, and documents ONNX export. However, the repository reports D-FINE-N at 4M parameters and 42.8 COCO AP, while D-FINE-L and D-FINE-X are 31M and 62M parameters—not models within a 25M upper bound. [^1]

SCRFD has the clearest lightweight face-detection evidence in the retrieved sources. SCRFD-0.5GF and SCRFD-2.5GF are reported at 0.57M and 0.67M parameters, with 3.6 ms and 4.2 ms inference times under the repository's stated VGA evaluation setup. The repository also documents ONNX conversion, but the measurements were not made on the proposed edge hardware and do not establish active-IR or overhead-camera performance. [^2]

AdaFace is a credible face-embedding research candidate, but it should not be described as already validated for Indian active-IR surveillance. Its official repository is MIT-licensed and reports results on IJB-B, IJB-C, IJB-S, and TinyFace; it also specifies 112×112 BGR input and provides pretrained research models. A third-party MIT-licensed repository provides ONNX exports, but those exports were not independently validated here. [^3] [^4]

## Tiered recommendation table

| Tier | Task | Model or pipeline | License status | Verified benchmark / size | Implementation status and evidence gate |
|---|---|---|---|---|---|
| Tier 1 | Object detection | **D-FINE-N** | Apache-2.0 [^1] | COCO AP 42.8; 4M parameters; 2.12 ms and 472 FPS are reported by the official repository under different hardware/measurement contexts [^1] | Primary candidate. Export procedure is documented, but Intel i3/N100 OpenVINO INT8 and active-IR performance remain to be measured [^1] |
| Tier 1 alternative | Object detection | MobileNetV4-Conv-S | Not verified in the retrieved evidence | Benchmark, parameter count, and target-device latency: **unverified — needs confirmation** | Keep as a fallback only after an exact checkpoint and license audit |
| Tier 1 alternative | Object detection | YOLOX-Nano | Not verified in the retrieved evidence | Benchmark, parameter count, and target-device latency: **unverified — needs confirmation** | Do not promote to the default recommendation until the exact repository/checkpoint license is verified |
| Tier 1 | Face detection/alignment | **SCRFD-0.5GF** | Exact model/weight license not verified in the retrieved source | WIDERFace Easy/Medium/Hard 90.57/88.12/68.51; 0.57M parameters; 3.6 ms under the repository's VGA setup [^2] | Preferred lightweight face-detector candidate pending weight-license and active-IR validation. ONNX conversion is documented [^2] |
| Tier 1 | Face detection/alignment | **SCRFD-2.5GF** | Exact model/weight license not verified in the retrieved source | WIDERFace Easy/Medium/Hard 93.78/92.16/77.87; 0.67M parameters; 4.2 ms under the repository's VGA setup [^2] | Higher-capacity Tier 1 alternative pending the same license and target-hardware checks |
| Tier 1–3 | Face embedding | **AdaFace IR-18 / IR-50 / IR-100** | Official training repository MIT [^3] | IR-100/MS1MV2 reports IJB-B/IJB-C TAR@FAR 0.01% of 95.67/96.89; low-quality IJB-S results are also reported [^3] | Research candidate, not demographic-parity or active-IR validation. Exact parameter counts: **unverified — needs confirmation**. Third-party ONNX exports exist but were not independently validated [^4] |
| Tier 1 | Pose estimation | RTMPose-T / RTMPose-S | Prompt claims Apache-2.0, but exact checkpoint/license evidence was not verified here | The prompt's 17-keypoint and ~6 ms claims: **unverified — needs confirmation** | Candidate only after exact MMPose checkpoint, export, and CPU benchmark verification |
| Tier 1 | Person re-identification | OSNet-x0.25 / OSNet-x0.5 | Prompt claims MIT, but exact checkpoint/license evidence was not verified here | The prompt's ~8 ms and 512-dimensional claims: **unverified — needs confirmation** | Candidate only after exact checkpoint, embedding dimensionality, and target-device benchmark verification |
| Tier 1 | ANPR/LPR | D-FINE or other permissively licensed plate detector + CLAHE + OCR | Component-by-component license audit required | Indian HSRP accuracy, 45° skew tolerance, glare robustness, character accuracy, and stated latency: **unverified — needs confirmation** | Treat as a pipeline hypothesis. Measure detector recall, OCR character accuracy, end-to-end latency, and false-read rate on representative Indian night/IR footage |
| Tier 2 | Object detection | D-FINE-S | Apache-2.0 [^1] | COCO AP 48.5; 10M parameters; 3.49 ms on the official T4 FP16 setup [^1] | Roadmap candidate; not evidence of RTX3050/Jetson Orin performance |
| Tier 2 | Object detection | D-FINE-M | Apache-2.0 [^1] | COCO AP 52.3; 19M parameters; 5.62 ms on the official T4 FP16 setup [^1] | Roadmap candidate; not evidence of station deployment performance |
| Tier 2–3 | Object detection | RT-DETRv2-S / M / L / X | License not verified from the retrieved primary source | COCO AP 47.9/49.9/53.4/54.3; 20M/31M/42M/76M parameters; T4 FP16 FPS 217/161/108/74 [^5] | Roadmap family. It does not meet the Tier 1 ≤5M constraint, and its paper benchmark is not active-IR or Intel CPU evidence |
| Tier 3 | Object detection | D-FINE-L / D-FINE-X | Apache-2.0 [^1] | COCO AP 54.0/55.8; 31M/62M parameters; 8.07/12.89 ms on the official T4 FP16 setup [^1] | High-capacity roadmap candidates. The prompt's ≤25M framing is incorrect for both models |
| Tier 3 | Face detection | SCRFD-34GF | Exact model/weight license not verified in the retrieved source | WIDERFace Easy/Medium/Hard 96.06/94.92/85.29; 9.80M parameters; 11.7 ms under the repository's VGA setup [^2] | Roadmap candidate; no evidence yet for overhead, occlusion, active-IR, or South Asian demographic parity |

## What the evidence supports—and what it does not

### Object detection

D-FINE-N is the only retrieved candidate that simultaneously has an official permissive-license statement, a published small model size below 5M parameters, an official benchmark, and documented ONNX export. Its 42.8 COCO AP is useful for model-family comparison, but it does not support the specification's claims about retaining humans below 25–30 px in border terrain. The decisive experiment is fine-tuning and testing on labeled active-IR CCTV with separate day, IR-grain, headlight-glare, blur, occlusion, and target-size strata.

The D-FINE family scales substantially beyond Tier 1. D-FINE-S and D-FINE-M are 10M and 19M parameters, while D-FINE-L and X are 31M and 62M. The official latency figures are T4 TensorRT FP16 measurements, so they must not be copied into an Intel CPU or Jetson specification. [^1]

RT-DETRv2 is a useful roadmap comparator, particularly because its paper reports a deployment-oriented discrete-sampling option and fixed 640×640 COCO evaluations. However, even its smallest listed model is 20M parameters, so it is not a Tier 1 model under the supplied ≤5M requirement. [^5]

### Face detection and recognition

SCRFD is a better-supported edge detector than the prompt's generic “permissive weights” wording suggests, but licensing must be checked separately for code, checkpoint, and downstream use. The repository reports WIDERFace metrics and ONNX conversion instructions; it does not establish performance for 4-meter cameras, downward angles above 35°, masks, active-IR noise, or a South Asian demographic-parity target. [^2]

AdaFace's quality-adaptive margin is relevant to degraded face imagery, and its repository reports low-quality surveillance benchmarks. That is still indirect evidence for this use case: IJB-S and TinyFace are not equivalent to the project's camera geometry, illumination, population, enrollment protocol, or operational false-match requirement. The adapter must preserve AdaFace's documented BGR channel order and 112×112 preprocessing. [^3]

### Pose, behavior, re-identification, and ANPR

The supplied prompt gives useful architecture directions for RTMPose, OSNet, and the ANPR pipeline, but the retrieved evidence did not verify the stated parameter counts, latencies, licenses, OCR character accuracies, or active-IR robustness for those components. They should therefore appear in the submission as “candidate architecture” or “unverified — needs confirmation,” not as validated performance claims.

Behavior analytics should be evaluated as an event-level system rather than by pose AP alone. The Tier 1 benchmark should measure false alarms per camera-hour, missed-event rate, detection-to-alert latency, and performance separately for climbing, crouching/crawling, carrying, and ordinary patrol activity. OCR should similarly report plate-detection recall, character accuracy, full-plate accuracy, and false reads by illumination, skew, blur, and plate condition.

## Tier 1 export and validation gate

No ONNX/OpenVINO artifact is labeled validated in this brief. The retrieved D-FINE and SCRFD sources document export routes, and a third-party AdaFace repository provides ONNX downloads, but validation requires running the exact model and preprocessing on the target device. [^1] [^2] [^4]

A defensible Tier 1 release should record:

1. exact checkpoint URL, commit or release, SHA-256 hash, and license files;
2. input shape, channel order, normalization, resize/letterbox policy, and output tensor schema;
3. ONNX opset, ONNX Runtime/OpenVINO version, precision mode, and whether post-processing is inside or outside the graph;
4. cold-start and steady-state latency on the named Intel i3/N100 and Coral hardware;
5. throughput and memory use for the required 2–4 camera concurrency;
6. accuracy on a held-out active-IR dataset, with target-height and failure-condition strata; and
7. failure logs for unsupported operators, quantization drift, preprocessing mismatch, and false positives.

## Decision summary

Use D-FINE-N and SCRFD-0.5GF or SCRFD-2.5GF as the initial Tier 1 investigation set, not as already-proven border-surveillance solutions. Use AdaFace as a gated embedding experiment with careful licensing, population, and false-match evaluation. Keep D-FINE-S/M, RT-DETRv2, SCRFD-34GF, and larger face/re-identification/pose systems in the roadmap. Do not present the prompt's unverified latency, small-target, HSRP-OCR, demographic-parity, or model-license claims as established facts until the exact checkpoints and target hardware have been tested.


[^1]: Peterande/D-FINE.

[^2]: detection/scrfd/README.md at master · deepinsight/insightface.

[^3]: mk-minchul/AdaFace.

[^4]: README.md - AdaFace ONNX.

[^5]: RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time ...