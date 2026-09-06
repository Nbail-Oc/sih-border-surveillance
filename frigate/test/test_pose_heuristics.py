"""Unit tests for RTMPose geometric heuristic rule engine."""

import unittest
import numpy as np


class TestPoseHeuristicRules(unittest.TestCase):
    def setUp(self):
        from frigate.analytics.pose_rules import PoseRuleEngine
        self.engine = PoseRuleEngine()

    def test_standing_normal_posture(self):
        """Standard standing person should trigger NO alarms."""
        # 17 COCO keypoints: [nose, l_eye, r_eye, l_ear, r_ear, l_shoulder, r_shoulder,
        # l_elbow, r_elbow, l_wrist, r_wrist, l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle]
        keypoints = np.zeros((17, 3), dtype=np.float32)
        # Head
        keypoints[0] = [100, 50, 0.9]  # Nose
        # Shoulders (y=100)
        keypoints[5] = [80, 100, 0.9]
        keypoints[6] = [120, 100, 0.9]
        # Wrists hanging down (y=200)
        keypoints[9] = [70, 200, 0.9]
        keypoints[10] = [130, 200, 0.9]
        # Hips (y=250)
        keypoints[11] = [85, 250, 0.9]
        keypoints[12] = [115, 250, 0.9]
        # Knees (y=350)
        keypoints[13] = [85, 350, 0.9]
        keypoints[14] = [115, 350, 0.9]
        # Ankles (y=450)
        keypoints[15] = [85, 450, 0.9]
        keypoints[16] = [115, 450, 0.9]

        events = self.engine.evaluate(keypoints, bbox=(50, 30, 150, 460))
        self.assertFalse(events["is_climbing"])
        self.assertFalse(events["is_crouching"])
        self.assertEqual(events["status"], "normal")

    def test_fence_climbing_detected_when_wrists_elevated(self):
        """Fence climbing triggered when wrists are elevated above shoulders."""
        keypoints = np.zeros((17, 3), dtype=np.float32)
        # Head (y=150)
        keypoints[0] = [100, 150, 0.9]
        # Shoulders (y=180)
        keypoints[5] = [80, 180, 0.9]
        keypoints[6] = [120, 180, 0.9]
        # Wrists reaching UP above shoulders/head (y=50)
        keypoints[9] = [70, 50, 0.9]
        keypoints[10] = [130, 50, 0.9]
        # Hips (y=280)
        keypoints[11] = [85, 280, 0.9]
        keypoints[12] = [115, 280, 0.9]
        # Knees bent upwards (y=320)
        keypoints[13] = [70, 320, 0.9]
        keypoints[14] = [130, 320, 0.9]
        # Ankles on fence rung (y=380)
        keypoints[15] = [70, 380, 0.9]
        keypoints[16] = [130, 380, 0.9]

        events = self.engine.evaluate(keypoints, bbox=(50, 40, 150, 400))
        self.assertTrue(events["is_climbing"])
        self.assertIn("climbing", events["alerts"])

    def test_crouching_or_crawling_detected(self):
        """Crouching detected when hip-to-ankle vertical distance is severely compressed."""
        keypoints = np.zeros((17, 3), dtype=np.float32)
        # Head (y=200)
        keypoints[0] = [100, 200, 0.9]
        # Shoulders (y=220)
        keypoints[5] = [80, 220, 0.9]
        keypoints[6] = [120, 220, 0.9]
        # Hips (y=240)
        keypoints[11] = [85, 240, 0.9]
        keypoints[12] = [115, 240, 0.9]
        # Knees bent low (y=250)
        keypoints[13] = [85, 250, 0.9]
        keypoints[14] = [115, 250, 0.9]
        # Ankles near hips (y=260)
        keypoints[15] = [85, 260, 0.9]
        keypoints[16] = [115, 260, 0.9]

        events = self.engine.evaluate(keypoints, bbox=(50, 190, 150, 270))
        self.assertTrue(events["is_crouching"])
        self.assertIn("crouching", events["alerts"])
