"""Face Detector module using OpenCV.

Provides local face detection capabilities for single and multi-face images with
deterministic face box sorting and false positive filtering.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import logging
import os
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FaceDetector:
    """Local face detector utilizing OpenCV CascadeClassifier for fast, offline face detection."""

    def __init__(self, scale_factor: float = 1.05, min_neighbors: int = 5, min_size: tuple[int, int] = (30, 30)) -> None:
        """Initialize the face detector with cascade configuration parameters.

        Args:
            scale_factor: Parameter specifying how much the image size is reduced at each image scale.
            min_neighbors: Parameter specifying how many neighbors each candidate rectangle should have to retain it (default: 5).
            min_size: Minimum possible object size. Objects smaller than that are ignored.
        """
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if not os.path.exists(cascade_path):
            raise RuntimeError(f"Haar cascade XML file not found at {cascade_path}")

        self.classifier = cv2.CascadeClassifier(cascade_path)
        if self.classifier.empty():
            raise RuntimeError("Failed to load OpenCV CascadeClassifier.")

    def detect_faces(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """Detect faces in an image file and return detected face boxes sorted by area descending.

        Args:
            image_path: Path to the image file.

        Returns:
            Structured dictionary containing detection results:
            {
                "success": bool,
                "face_count": int,
                "faces": [
                    {
                        "box": {"x": int, "y": int, "w": int, "h": int},
                        "confidence": float
                    }
                ],
                "image_size": {"width": int, "height": int},
                "error": str | None
            }
        """
        path = Path(image_path)

        if not path.is_file():
            return {
                "success": False,
                "face_count": 0,
                "faces": [],
                "image_size": {"width": 0, "height": 0},
                "error": f"Image file not found: {image_path}",
            }

        try:
            image_bytes = np.fromfile(str(path), dtype=np.uint8)
            img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        except Exception as err:
            return {
                "success": False,
                "face_count": 0,
                "faces": [],
                "image_size": {"width": 0, "height": 0},
                "error": f"Failed to read image file: {str(err)}",
            }

        if img is None or img.size == 0:
            return {
                "success": False,
                "face_count": 0,
                "faces": [],
                "image_size": {"width": 0, "height": 0},
                "error": f"Invalid or corrupt image format: {image_path}",
            }

        height, width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Primary detection attempt with configured min_neighbors
        rects, reject_weights = self._run_cascade(gray, self.min_neighbors)

        # Fallback attempt if 0 faces found and min_neighbors > 3
        if len(rects) == 0 and self.min_neighbors > 3:
            rects, reject_weights = self._run_cascade(gray, min_neighbors=3)

        detected_faces: List[Dict[str, Any]] = []

        if len(rects) > 0:
            for i, (x, y, w, h) in enumerate(rects):
                weight = float(reject_weights[i]) if i < len(reject_weights) else 1.0
                confidence = round(min(1.0, max(0.0, weight / 10.0)), 2) if weight > 0 else 0.5
                detected_faces.append(
                    {
                        "box": {
                            "x": int(x),
                            "y": int(y),
                            "w": int(w),
                            "h": int(h),
                        },
                        "confidence": confidence,
                    }
                )

            # Sort detected faces in descending order of bounding box area (w * h) and confidence
            detected_faces.sort(key=lambda f: (f["box"]["w"] * f["box"]["h"], f["confidence"]), reverse=True)

        logger.info(
            "Detected %d face(s) in %s. Boxes: %s",
            len(detected_faces),
            path.name,
            [f["box"] for f in detected_faces],
        )

        return {
            "success": True,
            "face_count": len(detected_faces),
            "faces": detected_faces,
            "image_size": {"width": int(width), "height": int(height)},
            "error": None,
        }

    def _run_cascade(self, gray_img: np.ndarray, min_neighbors: int) -> Tuple[Any, Any]:
        """Run CascadeClassifier detectMultiScale3 with fallback to detectMultiScale."""
        try:
            rects, rejects, reject_weights = self.classifier.detectMultiScale3(
                gray_img,
                scaleFactor=self.scale_factor,
                minNeighbors=min_neighbors,
                minSize=self.min_size,
                outputRejectLevels=True,
            )
            return rects, reject_weights
        except Exception:
            rects = self.classifier.detectMultiScale(
                gray_img,
                scaleFactor=self.scale_factor,
                minNeighbors=min_neighbors,
                minSize=self.min_size,
            )
            return rects, [1.0] * len(rects)
