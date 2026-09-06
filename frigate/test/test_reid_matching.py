"""Unit tests for OSNet Person Re-ID appearance vector matcher."""

import time
import unittest
import numpy as np


class TestReIDMatching(unittest.TestCase):
    def setUp(self):
        from frigate.analytics.reid_matcher import ReIDGallery
        self.gallery = ReIDGallery(similarity_threshold=0.70, ttl_seconds=900)

    def test_reid_same_identity_matches_across_cameras(self):
        """Identical or near-identical 512-d embeddings should match the same track ID."""
        # Create normalized random embedding vector
        np.random.seed(42)
        base_vector = np.random.randn(512).astype(np.float32)
        base_vector /= np.linalg.norm(base_vector)

        # Enroll target seen on Camera 1
        track_id_1, is_new_1 = self.gallery.query_or_enroll(
            embedding=base_vector,
            camera_name="north_gate_cam",
            timestamp=time.time()
        )
        self.assertTrue(is_new_1)

        # Same target appears 30 seconds later on Camera 2 with small visual perturbation (similarity ~0.95)
        noisy_vector = base_vector + (np.random.randn(512).astype(np.float32) * 0.01)
        noisy_vector /= np.linalg.norm(noisy_vector)

        track_id_2, is_new_2 = self.gallery.query_or_enroll(
            embedding=noisy_vector,
            camera_name="perimeter_fence_cam",
            timestamp=time.time() + 30
        )
        self.assertFalse(is_new_2)
        self.assertEqual(track_id_1, track_id_2)

    def test_reid_distinct_identity_creates_new_track(self):
        """Orthogonal or distinct 512-d embeddings should generate distinct track IDs."""
        np.random.seed(101)
        v1 = np.random.randn(512).astype(np.float32)
        v1 /= np.linalg.norm(v1)

        v2 = np.random.randn(512).astype(np.float32)
        v2 /= np.linalg.norm(v2)

        track_1, is_new_1 = self.gallery.query_or_enroll(v1, "cam_1", time.time())
        track_2, is_new_2 = self.gallery.query_or_enroll(v2, "cam_2", time.time())

        self.assertTrue(is_new_1)
        self.assertTrue(is_new_2)
        self.assertNotEqual(track_1, track_2)
