# """Face Encoder module using OpenCV SFace.

# Generates 128-dimensional L2-normalized facial embeddings using the official OpenCV SFace model.
# """

# from pathlib import Path
# from typing import Any, Dict, Optional, Union
# import cv2
# import numpy as np
# from core.face_detector import FaceDetector


# class FaceEncoder:
#     """Face encoder utilizing OpenCV's FaceRecognizerSF and SFace ONNX model for facial recognition embeddings."""

#     DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "models" / "face_recognition_sface_2021dec.onnx"

#     def __init__(self, model_path: Optional[Union[str, Path]] = None, detector: Optional[FaceDetector] = None) -> None:
#         """Initialize the FaceEncoder with OpenCV SFace model and FaceDetector instance."""
#         self.detector = detector or FaceDetector()
#         self.model_path = Path(model_path) if model_path else self.DEFAULT_MODEL_PATH

#         if not self.model_path.is_file():
#             raise RuntimeError(
#                 f"SFace ONNX model file not found at '{self.model_path}'. "
#                 "Please place 'face_recognition_sface_2021dec.onnx' in the models directory."
#             )

#         self.recognizer = cv2.FaceRecognizerSF.create(str(self.model_path), "")
#         if self.recognizer is None:
#             raise RuntimeError("Failed to create OpenCV FaceRecognizerSF instance.")

#         self.embedding_dim = 128
#         self.target_size = (112, 112)

#     def encode_face(self, image_path: Union[str, Path], face_index: int = 0) -> Dict[str, Any]:
#         """Detect faces and compute the 128-D SFace feature embedding for the specified face index.

#         Args:
#             image_path: Path to the input image file.
#             face_index: Index of the face to encode when multiple faces are detected (default: 0).

#         Returns:
#             Structured dictionary containing encoding results:
#             {
#                 "success": bool,
#                 "face_count": int,
#                 "selected_face_index": int,
#                 "box": {"x": int, "y": int, "w": int, "h": int} | None,
#                 "embedding": list[float] | None,  # 128 floats
#                 "embedding_dim": int,             # 128
#                 "error": str | None
#             }
#         """
#         path = Path(image_path)

#         # 1. Detect faces using Phase 1 FaceDetector
#         detection = self.detector.detect_faces(path)

#         if not detection["success"]:
#             return {
#                 "success": False,
#                 "face_count": 0,
#                 "selected_face_index": face_index,
#                 "box": None,
#                 "embedding": None,
#                 "embedding_dim": self.embedding_dim,
#                 "error": detection["error"],
#             }

#         face_count = detection["face_count"]
#         if face_count == 0:
#             return {
#                 "success": False,
#                 "face_count": 0,
#                 "selected_face_index": face_index,
#                 "box": None,
#                 "embedding": None,
#                 "embedding_dim": self.embedding_dim,
#                 "error": f"No faces detected in image: {image_path}",
#             }

#         # 2. Validate face_index selection
#         if face_index < 0 or face_index >= face_count:
#             return {
#                 "success": False,
#                 "face_count": face_count,
#                 "selected_face_index": face_index,
#                 "box": None,
#                 "embedding": None,
#                 "embedding_dim": self.embedding_dim,
#                 "error": f"Invalid face_index {face_index} for detected face_count {face_count}.",
#             }

#         # 3. Read image safely for cropping/alignment
#         try:
#             image_bytes = np.fromfile(str(path), dtype=np.uint8)
#             img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
#         except Exception as err:
#             return {
#                 "success": False,
#                 "face_count": face_count,
#                 "selected_face_index": face_index,
#                 "box": None,
#                 "embedding": None,
#                 "embedding_dim": self.embedding_dim,
#                 "error": f"Failed to read image file: {str(err)}",
#             }

#         if img is None or img.size == 0:
#             return {
#                 "success": False,
#                 "face_count": face_count,
#                 "selected_face_index": face_index,
#                 "box": None,
#                 "embedding": None,
#                 "embedding_dim": self.embedding_dim,
#                 "error": f"Failed to decode image file: {image_path}",
#             }

#         # 4. Extract bounding box for selected face
#         target_face = detection["faces"][face_index]
#         box = target_face["box"]
#         x, y, w, h = box["x"], box["y"], box["w"], box["h"]

#         # Ensure bounding box is within image boundaries
#         img_h, img_w = img.shape[:2]
#         x1, y1 = max(0, x), max(0, y)
#         x2, y2 = min(img_w, x + w), min(img_h, y + h)

#         face_roi = img[y1:y2, x1:x2]
#         if face_roi.size == 0:
#             return {
#                 "success": False,
#                 "face_count": face_count,
#                 "selected_face_index": face_index,
#                 "box": box,
#                 "embedding": None,
#                 "embedding_dim": self.embedding_dim,
#                 "error": "Face region of interest (ROI) is empty.",
#             }

#         # 5. Perform SFace face alignment and crop preprocessing
#         aligned_face = cv2.resize(face_roi, self.target_size)

#         # 6. Extract SFace 128-dimensional embedding
#         raw_feature = self.recognizer.feature(aligned_face)

#         # 7. L2-normalize embedding vector
#         norm = np.linalg.norm(raw_feature)
#         if norm > 0:
#             normalized_feature = raw_feature / norm
#         else:
#             normalized_feature = raw_feature

#         embedding_list = [float(v) for v in normalized_feature.flatten()]

#         return {
#             "success": True,
#             "face_count": face_count,
#             "selected_face_index": face_index,
#             "box": box,
#             "embedding": embedding_list,
#             "embedding_dim": len(embedding_list),
#             "error": None,
#         }

"""Face Encoder module using OpenCV SFace.

Generates 128-dimensional L2-normalized facial embeddings using
the official OpenCV SFace model.

Optimized version:
- Keeps backward-compatible encode_face().
- Adds encode_face_from_image().
- Adds encode_faces_from_image().
- Allows callers to reuse an already-decoded image.
- Allows callers to reuse an already-computed face detection.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np

from core.face_detector import FaceDetector


class FaceEncoder:
    """OpenCV SFace encoder."""

    DEFAULT_MODEL_PATH = (
        Path(__file__).parent.parent
        / "models"
        / "face_recognition_sface_2021dec.onnx"
    )

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        detector: Optional[FaceDetector] = None,
    ) -> None:

        self.detector = detector or FaceDetector()

        self.model_path = (
            Path(model_path)
            if model_path
            else self.DEFAULT_MODEL_PATH
        )

        if not self.model_path.is_file():
            raise RuntimeError(
                f"SFace ONNX model file not found at "
                f"'{self.model_path}'. "
                "Please place "
                "'face_recognition_sface_2021dec.onnx' "
                "in the models directory."
            )

        self.recognizer = (
            cv2.FaceRecognizerSF.create(
                str(self.model_path),
                "",
            )
        )

        if self.recognizer is None:
            raise RuntimeError(
                "Failed to create OpenCV "
                "FaceRecognizerSF instance."
            )

        self.embedding_dim = 128
        self.target_size = (112, 112)

    # ------------------------------------------------------------------
    # BACKWARD-COMPATIBLE API
    # ------------------------------------------------------------------

    def encode_face(
        self,
        image_path: Union[str, Path],
        face_index: int = 0,
    ) -> Dict[str, Any]:
        """Detect and encode one face.

        This method remains compatible with the existing pipeline.
        """

        path = Path(image_path)

        if not path.is_file():
            return self._error_result(
                face_index,
                f"Image file not found: {image_path}",
            )

        # Detect once.
        detection = self.detector.detect_faces(path)

        if not detection.get("success", False):
            return self._error_result(
                face_index,
                detection.get(
                    "error",
                    "Face detection failed.",
                ),
            )

        face_count = int(
            detection.get("face_count", 0)
        )

        if face_count == 0:
            return self._error_result(
                face_index,
                f"No faces detected in image: {image_path}",
                face_count=0,
            )

        if face_index < 0 or face_index >= face_count:
            return self._error_result(
                face_index,
                (
                    f"Invalid face_index {face_index} "
                    f"for detected face_count {face_count}."
                ),
                face_count=face_count,
            )

        # Decode once.
        try:
            image_bytes = np.fromfile(
                str(path),
                dtype=np.uint8,
            )

            image = cv2.imdecode(
                image_bytes,
                cv2.IMREAD_COLOR,
            )

        except Exception as err:
            return self._error_result(
                face_index,
                f"Failed to read image file: {err}",
                face_count=face_count,
            )

        if image is None or image.size == 0:
            return self._error_result(
                face_index,
                f"Failed to decode image file: {image_path}",
                face_count=face_count,
            )

        # Reuse the detection we already calculated.
        return self.encode_face_from_image(
            image=image,
            face_index=face_index,
            detection=detection,
        )

    # ------------------------------------------------------------------
    # OPTIMIZED SINGLE-FACE API
    # ------------------------------------------------------------------

    def encode_face_from_image(
        self,
        image: np.ndarray,
        face_index: int = 0,
        detection: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Encode one face from an already-decoded image.

        If detection is provided, face detection is NOT repeated.
        """

        if image is None:
            return self._error_result(
                face_index,
                "Input image is None.",
            )

        if not isinstance(image, np.ndarray):
            return self._error_result(
                face_index,
                "Input image must be a NumPy array.",
            )

        if image.size == 0:
            return self._error_result(
                face_index,
                "Input image is empty.",
            )

        # Reuse detection whenever available.
        if detection is None:
            detection = (
                self.detector.detect_faces_from_image(
                    image
                )
            )

        if not detection.get("success", False):
            return self._error_result(
                face_index,
                detection.get(
                    "error",
                    "Face detection failed.",
                ),
            )

        face_count = int(
            detection.get("face_count", 0)
        )

        if face_count == 0:
            return self._error_result(
                face_index,
                "No faces detected in image.",
                face_count=0,
            )

        if face_index < 0 or face_index >= face_count:
            return self._error_result(
                face_index,
                (
                    f"Invalid face_index {face_index} "
                    f"for detected face_count {face_count}."
                ),
                face_count=face_count,
            )

        faces = detection.get("faces", [])

        if face_index >= len(faces):
            return self._error_result(
                face_index,
                "Detected face index is unavailable.",
                face_count=face_count,
            )

        box = faces[face_index].get("box")

        if not box:
            return self._error_result(
                face_index,
                "Detected face has no bounding box.",
                face_count=face_count,
            )

        return self._encode_box(
            image=image,
            box=box,
            face_index=face_index,
            face_count=face_count,
        )

    # ------------------------------------------------------------------
    # OPTIMIZED MULTI-FACE API
    # ------------------------------------------------------------------

    def encode_faces_from_image(
        self,
        image: np.ndarray,
        detection: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Encode all detected faces without repeating detection."""

        if image is None:
            return []

        if not isinstance(image, np.ndarray):
            return []

        if image.size == 0:
            return []

        if detection is None:
            detection = (
                self.detector.detect_faces_from_image(
                    image
                )
            )

        if not detection.get("success", False):
            return []

        faces = detection.get("faces", [])

        results: List[Dict[str, Any]] = []

        for face_index in range(len(faces)):

            result = self.encode_face_from_image(
                image=image,
                face_index=face_index,
                detection=detection,
            )

            results.append(result)

        return results

    # ------------------------------------------------------------------
    # INTERNAL ENCODING
    # ------------------------------------------------------------------

    def _encode_box(
        self,
        image: np.ndarray,
        box: Dict[str, Any],
        face_index: int,
        face_count: int,
    ) -> Dict[str, Any]:
        """Crop and encode a known face bounding box."""

        try:
            x = int(box["x"])
            y = int(box["y"])
            w = int(box["w"])
            h = int(box["h"])

        except Exception:
            return self._error_result(
                face_index,
                "Invalid face bounding box.",
                face_count=face_count,
                box=box,
            )

        if w <= 0 or h <= 0:
            return self._error_result(
                face_index,
                "Face bounding box has invalid dimensions.",
                face_count=face_count,
                box=box,
            )

        img_h, img_w = image.shape[:2]

        x1 = max(0, x)
        y1 = max(0, y)

        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)

        if x2 <= x1 or y2 <= y1:
            return self._error_result(
                face_index,
                "Face region of interest is empty.",
                face_count=face_count,
                box=box,
            )

        face_roi = image[
            y1:y2,
            x1:x2,
        ]

        if face_roi.size == 0:
            return self._error_result(
                face_index,
                "Face region of interest is empty.",
                face_count=face_count,
                box=box,
            )

        try:
            # Keep your original preprocessing behavior.
            aligned_face = cv2.resize(
                face_roi,
                self.target_size,
            )

            raw_feature = self.recognizer.feature(
                aligned_face
            )

        except Exception as err:
            return self._error_result(
                face_index,
                f"SFace feature extraction failed: {err}",
                face_count=face_count,
                box=box,
            )

        if raw_feature is None:
            return self._error_result(
                face_index,
                "SFace returned no feature vector.",
                face_count=face_count,
                box=box,
            )

        raw_feature = np.asarray(
            raw_feature,
            dtype=np.float32,
        )

        if raw_feature.size == 0:
            return self._error_result(
                face_index,
                "SFace returned an empty feature vector.",
                face_count=face_count,
                box=box,
            )

        # L2 normalization.
        norm = float(
            np.linalg.norm(raw_feature)
        )

        if norm > 0.0:
            normalized_feature = (
                raw_feature / norm
            )
        else:
            normalized_feature = raw_feature

        embedding_list = [
            float(value)
            for value in normalized_feature.flatten()
        ]

        return {
            "success": True,
            "face_count": face_count,
            "selected_face_index": face_index,
            "box": box,
            "embedding": embedding_list,
            "embedding_dim": len(
                embedding_list
            ),
            "error": None,
        }

    # ------------------------------------------------------------------
    # ERROR BUILDER
    # ------------------------------------------------------------------

    def _error_result(
        self,
        face_index: int,
        error: str,
        face_count: int = 0,
        box: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        return {
            "success": False,
            "face_count": face_count,
            "selected_face_index": face_index,
            "box": box,
            "embedding": None,
            "embedding_dim": self.embedding_dim,
            "error": str(error),
        }