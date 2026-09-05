"""Face Encoder module using OpenCV SFace.

Generates 128-dimensional L2-normalized facial embeddings using the official OpenCV SFace model.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import cv2
import numpy as np
from core.face_detector import FaceDetector


class FaceEncoder:
    """Face encoder utilizing OpenCV's FaceRecognizerSF and SFace ONNX model for facial recognition embeddings."""

    DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "models" / "face_recognition_sface_2021dec.onnx"

    def __init__(self, model_path: Optional[Union[str, Path]] = None, detector: Optional[FaceDetector] = None) -> None:
        """Initialize the FaceEncoder with OpenCV SFace model and FaceDetector instance."""
        self.detector = detector or FaceDetector()
        self.model_path = Path(model_path) if model_path else self.DEFAULT_MODEL_PATH

        if not self.model_path.is_file():
            raise RuntimeError(
                f"SFace ONNX model file not found at '{self.model_path}'. "
                "Please place 'face_recognition_sface_2021dec.onnx' in the models directory."
            )

        self.recognizer = cv2.FaceRecognizerSF.create(str(self.model_path), "")
        if self.recognizer is None:
            raise RuntimeError("Failed to create OpenCV FaceRecognizerSF instance.")

        self.embedding_dim = 128
        self.target_size = (112, 112)

    def encode_face(self, image_path: Union[str, Path], face_index: int = 0) -> Dict[str, Any]:
        """Detect faces and compute the 128-D SFace feature embedding for the specified face index.

        Args:
            image_path: Path to the input image file.
            face_index: Index of the face to encode when multiple faces are detected (default: 0).

        Returns:
            Structured dictionary containing encoding results:
            {
                "success": bool,
                "face_count": int,
                "selected_face_index": int,
                "box": {"x": int, "y": int, "w": int, "h": int} | None,
                "embedding": list[float] | None,  # 128 floats
                "embedding_dim": int,             # 128
                "error": str | None
            }
        """
        path = Path(image_path)

        # 1. Detect faces using Phase 1 FaceDetector
        detection = self.detector.detect_faces(path)

        if not detection["success"]:
            return {
                "success": False,
                "face_count": 0,
                "selected_face_index": face_index,
                "box": None,
                "embedding": None,
                "embedding_dim": self.embedding_dim,
                "error": detection["error"],
            }

        face_count = detection["face_count"]
        if face_count == 0:
            return {
                "success": False,
                "face_count": 0,
                "selected_face_index": face_index,
                "box": None,
                "embedding": None,
                "embedding_dim": self.embedding_dim,
                "error": f"No faces detected in image: {image_path}",
            }

        # 2. Validate face_index selection
        if face_index < 0 or face_index >= face_count:
            return {
                "success": False,
                "face_count": face_count,
                "selected_face_index": face_index,
                "box": None,
                "embedding": None,
                "embedding_dim": self.embedding_dim,
                "error": f"Invalid face_index {face_index} for detected face_count {face_count}.",
            }

        # 3. Read image safely for cropping/alignment
        try:
            image_bytes = np.fromfile(str(path), dtype=np.uint8)
            img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        except Exception as err:
            return {
                "success": False,
                "face_count": face_count,
                "selected_face_index": face_index,
                "box": None,
                "embedding": None,
                "embedding_dim": self.embedding_dim,
                "error": f"Failed to read image file: {str(err)}",
            }

        if img is None or img.size == 0:
            return {
                "success": False,
                "face_count": face_count,
                "selected_face_index": face_index,
                "box": None,
                "embedding": None,
                "embedding_dim": self.embedding_dim,
                "error": f"Failed to decode image file: {image_path}",
            }

        # 4. Extract bounding box for selected face
        target_face = detection["faces"][face_index]
        box = target_face["box"]
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]

        # Ensure bounding box is within image boundaries
        img_h, img_w = img.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(img_w, x + w), min(img_h, y + h)

        face_roi = img[y1:y2, x1:x2]
        if face_roi.size == 0:
            return {
                "success": False,
                "face_count": face_count,
                "selected_face_index": face_index,
                "box": box,
                "embedding": None,
                "embedding_dim": self.embedding_dim,
                "error": "Face region of interest (ROI) is empty.",
            }

        # 5. Perform SFace face alignment and crop preprocessing
        aligned_face = cv2.resize(face_roi, self.target_size)

        # 6. Extract SFace 128-dimensional embedding
        raw_feature = self.recognizer.feature(aligned_face)

        # 7. L2-normalize embedding vector
        norm = np.linalg.norm(raw_feature)
        if norm > 0:
            normalized_feature = raw_feature / norm
        else:
            normalized_feature = raw_feature

        embedding_list = [float(v) for v in normalized_feature.flatten()]

        return {
            "success": True,
            "face_count": face_count,
            "selected_face_index": face_index,
            "box": box,
            "embedding": embedding_list,
            "embedding_dim": len(embedding_list),
            "error": None,
        }
