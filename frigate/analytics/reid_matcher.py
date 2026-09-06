"""In-memory appearance feature vector store and cross-camera matching gallery."""

import time
import uuid
from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass
class TrackIdentity:
    track_id: str
    embedding: np.ndarray
    last_camera: str
    last_seen: float
    history: list[dict[str, Any]]


class ReIDGallery:
    """Maintains a rolling temporal gallery of person appearance embeddings."""

    def __init__(self, similarity_threshold: float = 0.70, ttl_seconds: float = 900.0):
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.tracks: dict[str, TrackIdentity] = {}

    def _purge_expired(self, current_time: float):
        """Remove tracks older than TTL."""
        expired = [
            tid for tid, track in self.tracks.items()
            if current_time - track.last_seen > self.ttl_seconds
        ]
        for tid in expired:
            del self.tracks[tid]

    def query_or_enroll(
        self, embedding: np.ndarray, camera_name: str, timestamp: float | None = None
    ) -> tuple[str, bool]:
        """Query gallery for closest matching identity or enroll new track.

        Args:
            embedding: 1D numpy array of float32 (512-d or 768-d), L2-normalized.
            camera_name: Identifier of the camera where the target was observed.
            timestamp: Observation timestamp.

        Returns:
            (track_id, is_new_enrollment)
        """
        now = timestamp if timestamp is not None else time.time()
        self._purge_expired(now)

        # Ensure normalized
        norm = np.linalg.norm(embedding)
        if norm > 0:
            query_vec = embedding / norm
        else:
            query_vec = embedding

        best_score = -1.0
        best_match_id = None

        for tid, track in self.tracks.items():
            # Cosine similarity between L2-normalized vectors
            sim = float(np.dot(query_vec, track.embedding))
            if sim > best_score:
                best_score = sim
                best_match_id = tid

        if best_match_id and best_score >= self.similarity_threshold:
            # Update existing identity track
            matched_track = self.tracks[best_match_id]
            # Exponential moving average to update embedding
            matched_track.embedding = 0.8 * matched_track.embedding + 0.2 * query_vec
            matched_track.embedding /= np.linalg.norm(matched_track.embedding)
            matched_track.last_camera = camera_name
            matched_track.last_seen = now
            matched_track.history.append({"camera": camera_name, "time": now, "score": best_score})
            return best_match_id, False
        else:
            # Create new identity track
            new_id = f"target_{uuid.uuid4().hex[:8]}"
            self.tracks[new_id] = TrackIdentity(
                track_id=new_id,
                embedding=query_vec.copy(),
                last_camera=camera_name,
                last_seen=now,
                history=[{"camera": camera_name, "time": now, "score": 1.0}],
            )
            return new_id, True
