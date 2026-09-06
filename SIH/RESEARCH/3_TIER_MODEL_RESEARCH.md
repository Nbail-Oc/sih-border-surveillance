# Audited **3-Tier** Surveillance **Model Table**

The legally compliant shortlist supports five sub-questions: object detection, ANPR/OCR, face recognition, pose estimation, and cross-camera Re-ID. Tier 1 is the only build-now scope, so Tier 2 and Tier 3 are kept as roadmap entries, and any metric not stated in the supplied papers is marked **unverified — needs confirmation**.

## Consolidated Table

| Task | Tier | Model Name | Paper/Repo Citation | License | Accuracy Benchmark | Expected Latency (Hardware) |
|---|---|---|---|---|---|---|
| Primary object detection | Tier 1 (Build Now) | **D-FINE-N** | D-FINE repo/paper  (Peng et al., 2024)| Apache-2.0 | **unverified — needs confirmation** | **<15 ms on Intel i3 CPU — unverified — needs confirmation** |
| Primary object detection | Tier 2 (Roadmap) | **D-FINE-S / M** | D-FINE paper  (Peng et al., 2024)| Apache-2.0 | **unverified — needs confirmation** | **unverified — needs confirmation** |
| Primary object detection | Tier 2 (Roadmap) | **RT-DETRv2-S** | Supported as comparator in DEIM benchmarks  (Huang et al., 2024)| Apache-2.0 | **unverified — needs confirmation** | **unverified — needs confirmation** |
| Primary object detection | Tier 3 (Roadmap) | **D-FINE-L** | D-FINE paper  (Peng et al., 2024)| Apache-2.0 | **54.0% AP on COCO val2017**  (Peng et al., 2024; Huang et al., 2024)| **124 FPS on NVIDIA T4**  (Peng et al., 2024; Huang et al., 2024)|
| Primary object detection | Tier 3 (Roadmap) | **D-FINE-X** | D-FINE paper  (Peng et al., 2024)| Apache-2.0 | **55.8% AP on COCO val2017**  (Peng et al., 2024)| **78 FPS on NVIDIA T4**  (Peng et al., 2024; Huang et al., 2024)|
| Primary object detection | Tier 3 (Roadmap) | **RT-DETR-X** | Mentioned as strong DETR baseline in DEIM comparison  (Huang et al., 2024)| Apache-2.0 | **unverified — needs confirmation** | **unverified — needs confirmation** |
| Primary object detection | Tier 3 (Roadmap) | **Co-DETR (Swin-L)** | No supporting benchmark sentence in supplied papers | **unverified — needs confirmation** | **unverified — needs confirmation** | **unverified — needs confirmation** |
| ANPR / LPR | Tier 1 (Build Now) | **D-FINE tiny plate detector + CLAHE + PP-OCRv5-Mobile** | PP-OCR-style lightweight recognizer supported in unconstrained LPR pipeline  (Fan et al., 2024)| Apache-2.0 / BSD-style OpenCV | **97.61% recognition accuracy on CCPD** for PP-OCRv2-derived recognizer pipeline  (Fan et al., 2024)| **667 FPS recognizer, 87 FPS end-to-end on mid-end GPU**; CPU figure **unverified**  (Fan et al., 2024)|
| ANPR / LPR | Tier 1 (Build Now) | **D-FINE tiny plate detector + CLAHE + SVTR-v2-tiny** | SVTR/SVTRv2 text recognizers  (Du et al., 2022; Du et al., 2024)| Apache-2.0 | **Top-tier benchmark accuracy, exact tiny-model figure unverified — needs confirmation**  (Du et al., 2022; Du et al., 2024)| **<12 ms CPU — unverified — needs confirmation** |
| ANPR / LPR | Tier 2 (Roadmap) | **D-FINE plate detector + ParseQ-Tiny** | No ParseQ benchmark sentence in supplied papers | Apache-2.0 for detector; ParseQ metric **unverified** | **unverified — needs confirmation** | **unverified — needs confirmation** |
| ANPR / LPR | Tier 3 (Roadmap) | **STN rectification + Real-ESRGAN-Compact + ParseQ-Large / TrOCR** | Perspective correction is supported in LPR literature  (Fan et al., 2024)| Mixed; exact stack licenses **unverified — needs confirmation** | **unverified — needs confirmation** | **unverified — needs confirmation** |
| Face detection + recognition | Tier 1 (Build Now) | **RetinaFace-MobileNet0.25 + AdaFace-Mobile** | RetinaFace uses MobileNet backbones  (Ren et al., 2025); AdaFace is robust on low-quality faces  (Kim et al., 2022)| MIT | AdaFace improves low-quality recognition; **Tier-1 mobile accuracy figure unverified — needs confirmation**  (Kim et al., 2022)| **~6 ms CPU — unverified — needs confirmation** |
| Face detection + recognition | Tier 2 (Roadmap) | **SCRFD-2.5G + AdaFace (IResNet50)** | AdaFace quality-adaptive recognition  (Kim et al., 2022)| MIT | **98.06% AUROC on Indian RFW — unverified — needs confirmation** | **unverified — needs confirmation** |
| Face detection + recognition | Tier 3 (Roadmap) | **SCRFD-34G + AdaFace-ResNet100 + SDD-FIQA** | AdaFace ResNet100 benchmark setting  (Kim et al., 2022)| MIT | AdaFace achieves **SOTA across mixed-quality datasets** with ResNet100, but this exact stack metric is **unverified**  (Kim et al., 2022)| **unverified — needs confirmation** |
| Perimeter pose estimation | Tier 1 (Build Now) | **RTMPose-T / S** | RTMPose deployment paper  (Jiang et al., 2023)| Apache-2.0 | **17 keypoints** on COCO-style human pose benchmarks  (Lu et al., 2023)| **~6 ms CPU — unverified — needs confirmation** |
| Perimeter pose estimation | Tier 2 (Roadmap) | **RTMPose-M** | RTMPose paper  (Jiang et al., 2023)| Apache-2.0 | **75.8% AP on COCO val** with flip test  (Jiang et al., 2023)| **90+ FPS on Intel i7-11700 CPU**  (Jiang et al., 2023)|
| Perimeter pose estimation | Tier 3 (Roadmap) | **RTMPose-L / RTMW-L** | RTMW whole-body benchmarks  (Jiang et al., 2024)| Apache-2.0 | **66.0 AP at 256×192** and **70.1–70.2 mAP** on COCO-WholeBody for RTMW-L  (Jiang et al., 2024)| **35.322 ms CPU at 256×192** or **47.618 ms at 384×288**  (Jiang et al., 2024)|
| Cross-camera person Re-ID | Tier 1 (Build Now) | **OSNet-x0_25 / x0_5** | No OSNet benchmark sentence in supplied papers | MIT | **unverified — needs confirmation** | **~8 ms CPU — unverified — needs confirmation** |
| Cross-camera person Re-ID | Tier 2 (Roadmap) | **OSNet-IBN-x1_0 / FastReID-ResNet50** | No supporting benchmark sentence in supplied papers | MIT / Apache-2.0 | **unverified — needs confirmation** | **unverified — needs confirmation** |
| Cross-camera person Re-ID | Tier 3 (Roadmap) | **TransReID (ViT-Base) + FAISS** | No supporting benchmark sentence in supplied papers | Mixed; exact stack **unverified — needs confirmation** | **unverified — needs confirmation** | **unverified — needs confirmation** |

**Figure 1:** Consolidated audited shortlist across five surveillance tasks and three deployment tiers, with unsupported metrics explicitly marked unverified.

## Audit Notes

The strongest evidence supports **D-FINE** for detector scalability, **AdaFace** for low-quality face recognition, and **RTMPose/RTMW** for deployable pose estimation. D-FINE is positioned as a leading real-time DETR family, and later work reports further gains in small-object detection without extra inference cost  (Peng et al., 2024; Huang et al., 2024).

For ANPR, the supplied evidence supports PP-OCR-derived and SVTR-family recognizers plus perspective rectification, but it does **not** provide CPU numbers for the exact Tier 1 stack, nor Indian HSRP-specific benchmarks  (Fan et al., 2024; Du et al., 2022; Du et al., 2024). Lower-ranked papers still reinforce design choices: NMS-free or adaptive-NMS detectors can improve simplicity or speed  (Zhou et al., 2021; Zhao et al., 2022), and lightweight CPU pose baselines show that real-time edge inference is feasible even before newer RTMPose models  (Osokin, 2018).

This audited table therefore supports your requested **3-tier border surveillance** selection while keeping Tier 1 as the only build-now scope and flagging every unsupported benchmark as **unverified — needs confirmation**.
 
_These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at https://consensus.app. © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent._
 
## References
 
Du, Y., Chen, Z., Jia, C., Yin, X., Zheng, T., Li, C., Du, Y., & Jiang, Y.-G. (2022). SVTR: Scene Text Recognition with a Single Visual Model. 884-890. https://doi.org/10.48550/arxiv.2205.00159
 
Du, Y., Chen, Z., Xie, H., Jia, C., & Jiang, Y.-G. (2024). SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition. *2025 IEEE/CVF International Conference on Computer Vision (ICCV)*, 20147-20156. https://doi.org/10.1109/iccv51701.2025.01874
 
Fan, J., Liu, G., Zuo, P., Ke, Z., & Xu, G. (2024). Real-time license plate detection and recognition in unconstrained scenarios. *Proceedings of the 2024 3rd International Conference on Frontiers of Artificial Intelligence and Machine Learning*. https://doi.org/10.1145/3653644.3665200
 
Huang, S., Lu, Z., Cun, X., Yu, Y., Zhou, X., & Shen, X. (2024). DEIM: DETR with Improved Matching for Fast Convergence. *2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 15162-15171. https://doi.org/10.1109/cvpr52734.2025.01412
 
Jiang, T., Lu, P., Zhang, L., N., Han, R., Lyu, C., Li, Y., & Chen, K. (2023). RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose. *ArXiv, abs/2303.07399*. https://doi.org/10.48550/arxiv.2303.07399
 
Jiang, T., Xie, X., & Li, Y. (2024). RTMW: Real-Time Multi-Person 2D and 3D Whole-body Pose Estimation. *ArXiv, abs/2407.08634*. https://doi.org/10.48550/arxiv.2407.08634
 
Kim, M., Jain, A. K., & Liu, X. (2022). AdaFace: Quality Adaptive Margin for Face Recognition. *2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 18729-18738. https://doi.org/10.1109/cvpr52688.2022.01819
 
Lu, P., Jiang, T., Li, Y., Li, X., Chen, K., & Yang, W. (2023). RTMO: Towards High-Performance One-Stage Real-Time Multi-Person Pose Estimation. *2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 1491-1500. https://doi.org/10.1109/cvpr52733.2024.00148
 
Osokin, D. (2018). Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose. 744-748. https://doi.org/10.5220/0007555407440748
 
Peng, Y., Li, H., Wu, P., Zhang, Y., Sun, X., & Wu, F. (2024). D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement. *ArXiv, abs/2410.13842*. https://doi.org/10.48550/arxiv.2410.13842
 
Ren, Z., Liu, X., Xu, J., Zhang, Y., & Fang, M. (2025). LittleFaceNet: A Small-Sized Face Recognition Method Based on RetinaFace and AdaFace. *Journal of Imaging, 11*. https://doi.org/10.3390/jimaging11010024
 
Zhao, H., Wang, J., Dai, D., Lin, S., & Chen, Z. (2022). D-NMS: A dynamic NMS network for general object detection. *Neurocomputing, 512*, 225-234. https://doi.org/10.1016/j.neucom.2022.09.080
 
Zhou, Q.-F., Yu, C., Shen, C., Wang, Z., & Li, H. (2021). Object Detection Made Simpler by Eliminating Heuristic NMS. *ArXiv, abs/2101.11782*.
 
