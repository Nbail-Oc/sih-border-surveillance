"""Deterministic geometric heuristic rule engine for RTMPose 17 keypoints."""

from typing import Any
import numpy as np


class PoseRuleEngine:
    """Evaluates 2D human pose skeletons for suspicious border perimeter behavior."""

    # 17 COCO Keypoint Indices:
    # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
    # 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
    # 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
    # 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

    def __init__(self, min_confidence: float = 0.3):
        self.min_confidence = min_confidence

    def evaluate(self, keypoints: np.ndarray, bbox: tuple[float, float, float, float] | None = None) -> dict[str, Any]:
        """Evaluate keypoints against perimeter heuristic rules.

        Args:
            keypoints: Array of shape (17, 3) where columns are (x, y, confidence).
            bbox: Optional bounding box (x1, y1, x2, y2).

        Returns:
            Dictionary containing alert flags and computed behavior state.
        """
        alerts = []
        is_climbing = False
        is_crouching = False

        # Extract joints
        l_shoulder, r_shoulder = keypoints[5], keypoints[6]
        l_wrist, r_wrist = keypoints[9], keypoints[10]
        l_hip, r_hip = keypoints[11], keypoints[12]
        l_ankle, r_ankle = keypoints[15], keypoints[16]

        # Check shoulder and wrist confidences
        shoulder_conf = (l_shoulder[2] + r_shoulder[2]) / 2.0
        wrist_conf = max(l_wrist[2], r_wrist[2])

        if shoulder_conf >= self.min_confidence and wrist_conf >= self.min_confidence:
            avg_shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2.0
            min_wrist_y = min(
                l_wrist[1] if l_wrist[2] >= self.min_confidence else float("inf"),
                r_wrist[1] if r_wrist[2] >= self.min_confidence else float("inf"),
            )

            # In image coordinate space, smaller y = higher elevation
            # Wrist elevated above shoulders indicates climbing / scaling motion
            if min_wrist_y < avg_shoulder_y:
                is_climbing = True
                alerts.append("climbing")

        # Check crouching / crawling
        hip_conf = (l_hip[2] + r_hip[2]) / 2.0
        ankle_conf = (l_ankle[2] + r_ankle[2]) / 2.0

        if hip_conf >= self.min_confidence and ankle_conf >= self.min_confidence:
            avg_hip_y = (l_hip[1] + r_hip[1]) / 2.0
            avg_ankle_y = (l_ankle[1] + r_ankle[1]) / 2.0
            leg_height = abs(avg_ankle_y - avg_hip_y)

            # Torso height
            avg_shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2.0
            torso_height = max(abs(avg_hip_y - avg_shoulder_y), 1.0)

            # If leg height is severely compressed relative to torso height, person is crouching
            if leg_height / torso_height < 0.65 or leg_height < 30.0:
                is_crouching = True
                alerts.append("crouching")

        return {
            "status": "alert" if alerts else "normal",
            "is_climbing": is_climbing,
            "is_crouching": is_crouching,
            "alerts": alerts,
        }
