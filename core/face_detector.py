# """Face Detector module using OpenCV.

# Provides local face detection capabilities for single and multi-face images with
# deterministic face box sorting, false positive filtering, and lighting compensation.
# """

# from pathlib import Path
# from typing import Any, Dict, List, Tuple, Union
# import logging
# import os
# import cv2
# import numpy as np

# logger = logging.getLogger(__name__)


# def nms_boxes(boxes: List[Tuple[int, int, int, int, float]], iou_threshold: float = 0.3) -> List[Tuple[int, int, int, int, float]]:
#     """Apply Non-Maximum Suppression to remove overlapping redundant bounding boxes."""
#     if not boxes:
#         return []

#     boxes = sorted(boxes, key=lambda b: (b[2] * b[3], b[4]), reverse=True)
#     kept = []

#     for b1 in boxes:
#         x1, y1, w1, h1, c1 = b1
#         overlap = False
#         for b2 in kept:
#             x2, y2, w2, h2, c2 = b2
#             # Calculate IoU
#             xx1 = max(x1, x2)
#             yy1 = max(y1, y2)
#             xx2 = min(x1 + w1, x2 + w2)
#             yy2 = min(y1 + h1, y2 + h2)

#             w_inter = max(0, xx2 - xx1)
#             h_inter = max(0, yy2 - yy1)
#             inter_area = w_inter * h_inter

#             area1 = w1 * h1
#             area2 = w2 * h2
#             union_area = area1 + area2 - inter_area

#             iou = inter_area / union_area if union_area > 0 else 0
#             if iou > iou_threshold:
#                 overlap = True
#                 break

#         if not overlap:
#             kept.append(b1)

#     return kept


# class FaceDetector:
#     """Robust local face detector utilizing OpenCV CascadeClassifier with multi-scale equalization."""

#     def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5, min_size: tuple[int, int] = (30, 30)) -> None:
#         """Initialize the face detector with cascade configuration parameters."""
#         self.scale_factor = scale_factor
#         self.min_neighbors = min_neighbors
#         self.min_size = min_size

#         # Find cascade file across standard paths
#         possible_paths = [
#             os.path.join(getattr(cv2.data, 'haarcascades', ''), "haarcascade_frontalface_default.xml"),
#             os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml"),
#             os.path.join(os.path.dirname(__file__), "..", "models", "haarcascade_frontalface_default.xml"),
#         ]

#         cascade_path = None
#         for p in possible_paths:
#             if os.path.exists(p):
#                 cascade_path = p
#                 break

#         if not cascade_path:
#             raise RuntimeError(f"Haar cascade XML file not found in {possible_paths}")

#         self.classifier = cv2.CascadeClassifier(cascade_path)
#         if self.classifier.empty():
#             raise RuntimeError("Failed to load OpenCV CascadeClassifier.")

#     def detect_faces(self, image_path: Union[str, Path]) -> Dict[str, Any]:
#         """Detect faces in an image file with multi-stage false-positive filtering."""
#         path = Path(image_path)

#         if not path.is_file():
#             return {
#                 "success": False,
#                 "face_count": 0,
#                 "faces": [],
#                 "image_size": {"width": 0, "height": 0},
#                 "error": f"Image file not found: {image_path}",
#             }

#         try:
#             image_bytes = np.fromfile(str(path), dtype=np.uint8)
#             img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
#         except Exception as err:
#             return {
#                 "success": False,
#                 "face_count": 0,
#                 "faces": [],
#                 "image_size": {"width": 0, "height": 0},
#                 "error": f"Failed to read image file: {str(err)}",
#             }

#         if img is None or img.size == 0:
#             return {
#                 "success": False,
#                 "face_count": 0,
#                 "faces": [],
#                 "image_size": {"width": 0, "height": 0},
#                 "error": f"Invalid or corrupt image format: {image_path}",
#             }

#         height, width = img.shape[:2]
#         center_x, center_y = width / 2.0, height / 2.0
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#         raw_candidates: List[Tuple[int, int, int, int, float]] = []

#         # Pass 1: Standard grayscale
#         rects1, weights1 = self._run_cascade(gray, self.scale_factor, self.min_neighbors)
#         for i, (x, y, w, h) in enumerate(rects1):
#             w_score = float(weights1[i]) if i < len(weights1) else 1.0
#             raw_candidates.append((int(x), int(y), int(w), int(h), w_score))

#         # Pass 2: Histogram equalized (for backlit or low-contrast faces)
#         if len(raw_candidates) == 0:
#             gray_eq = cv2.equalizeHist(gray)
#             rects2, weights2 = self._run_cascade(gray_eq, 1.1, max(3, self.min_neighbors - 1))
#             for i, (x, y, w, h) in enumerate(rects2):
#                 w_score = float(weights2[i]) if i < len(weights2) else 1.0
#                 raw_candidates.append((int(x), int(y), int(w), int(h), w_score))

#         # Pass 3: Sensitive fallback if still no face found
#         if len(raw_candidates) == 0:
#             rects3, weights3 = self._run_cascade(gray, 1.05, 3)
#             for i, (x, y, w, h) in enumerate(rects3):
#                 w_score = float(weights3[i]) if i < len(weights3) else 0.5
#                 raw_candidates.append((int(x), int(y), int(w), int(h), w_score))

#         # Filter candidates: Aspect ratio check (human face aspect ratio is approx 0.75 - 1.35)
#         filtered_candidates = []
#         for x, y, w, h, score in raw_candidates:
#             aspect_ratio = float(w) / float(h)
#             if 0.70 <= aspect_ratio <= 1.40 and w >= 25 and h >= 25:
#                 filtered_candidates.append((x, y, w, h, score))

#         # Apply Non-Maximum Suppression to remove duplicates
#         clean_candidates = nms_boxes(filtered_candidates, iou_threshold=0.3)

#         # Structure detected faces
#         detected_faces: List[Dict[str, Any]] = []
#         for x, y, w, h, score in clean_candidates:
#             confidence = round(min(1.0, max(0.0, score / 10.0)), 2) if score > 0 else 0.5
#             # Calculate distance to image center
#             fx_center = x + w / 2.0
#             fy_center = y + h / 2.0
#             dist_to_center = np.sqrt((fx_center - center_x) ** 2 + (fy_center - center_y) ** 2)
#             norm_dist = dist_to_center / (np.sqrt(width**2 + height**2) / 2.0 + 1e-5)

#             # Centrality score (closer to center = higher priority)
#             centrality_score = (w * h) * (1.0 - 0.4 * min(1.0, norm_dist))

#             detected_faces.append({
#                 "box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
#                 "confidence": confidence,
#                 "_rank_score": centrality_score
#             })

#         # Sort faces: Primary prominent face first
#         detected_faces.sort(key=lambda f: f["_rank_score"], reverse=True)
#         for f in detected_faces:
#             del f["_rank_score"]

#         logger.info("Detected %d face(s) in %s.", len(detected_faces), path.name)

#         return {
#             "success": True,
#             "face_count": len(detected_faces),
#             "faces": detected_faces,
#             "image_size": {"width": int(width), "height": int(height)},
#             "error": None,
#         }

#     def _run_cascade(self, gray_img: np.ndarray, scale_factor: float, min_neighbors: int) -> Tuple[Any, Any]:
#         """Run CascadeClassifier detectMultiScale3 with fallback."""
#         try:
#             rects, rejects, reject_weights = self.classifier.detectMultiScale3(
#                 gray_img,
#                 scaleFactor=scale_factor,
#                 minNeighbors=min_neighbors,
#                 minSize=self.min_size,
#                 outputRejectLevels=True,
#             )
#             return rects, reject_weights
#         except Exception:
#             rects = self.classifier.detectMultiScale(
#                 gray_img,
#                 scaleFactor=scale_factor,
#                 minNeighbors=min_neighbors,
#                 minSize=self.min_size,
#             )
#             return rects, [1.0] * len(rects)


"""Face Detector module using OpenCV.

Provides local face detection capabilities for single and multi-face images with
deterministic face box sorting, false-positive filtering, and lighting compensation.

Optimized version:
- Supports detection directly from an already-decoded OpenCV image.
- Avoids unnecessary image decoding when the caller already has the image.
- Keeps the original 3-stage detection fallback behavior.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import logging
import os

import cv2
import numpy as np


logger = logging.getLogger(__name__)


def nms_boxes(
    boxes: List[Tuple[int, int, int, int, float]],
    iou_threshold: float = 0.3,
) -> List[Tuple[int, int, int, int, float]]:
    """Apply Non-Maximum Suppression to remove overlapping boxes."""

    if not boxes:
        return []

    boxes = sorted(
        boxes,
        key=lambda b: (b[2] * b[3], b[4]),
        reverse=True,
    )

    kept = []

    for b1 in boxes:
        x1, y1, w1, h1, c1 = b1
        overlap = False

        for b2 in kept:
            x2, y2, w2, h2, c2 = b2

            xx1 = max(x1, x2)
            yy1 = max(y1, y2)
            xx2 = min(x1 + w1, x2 + w2)
            yy2 = min(y1 + h1, y2 + h2)

            w_inter = max(0, xx2 - xx1)
            h_inter = max(0, yy2 - yy1)

            inter_area = w_inter * h_inter

            area1 = w1 * h1
            area2 = w2 * h2

            union_area = area1 + area2 - inter_area

            iou = (
                inter_area / union_area
                if union_area > 0
                else 0.0
            )

            if iou > iou_threshold:
                overlap = True
                break

        if not overlap:
            kept.append(b1)

    return kept


class FaceDetector:
    """Robust local face detector using OpenCV Haar Cascade."""

    def __init__(
        self,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: tuple[int, int] = (30, 30),
    ) -> None:

        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

        possible_paths = [
            os.path.join(
                getattr(cv2.data, "haarcascades", ""),
                "haarcascade_frontalface_default.xml",
            ),
            os.path.join(
                os.path.dirname(__file__),
                "haarcascade_frontalface_default.xml",
            ),
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "models",
                "haarcascade_frontalface_default.xml",
            ),
        ]

        cascade_path = None

        for path in possible_paths:
            if os.path.exists(path):
                cascade_path = path
                break

        if not cascade_path:
            raise RuntimeError(
                "Haar cascade XML file not found in "
                f"{possible_paths}"
            )

        self.classifier = cv2.CascadeClassifier(cascade_path)

        if self.classifier.empty():
            raise RuntimeError(
                "Failed to load OpenCV CascadeClassifier."
            )

    # ------------------------------------------------------------------
    # PUBLIC PATH API
    # ------------------------------------------------------------------

    def detect_faces(
        self,
        image_path: Union[str, Path],
    ) -> Dict[str, Any]:
        """Detect faces in an image file."""

        path = Path(image_path)

        if not path.is_file():
            return self._error_result(
                f"Image file not found: {image_path}"
            )

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
                f"Failed to read image file: {str(err)}"
            )

        if image is None or image.size == 0:
            return self._error_result(
                f"Invalid or corrupt image format: {image_path}"
            )

        result = self.detect_faces_from_image(image)

        if result["success"]:
            logger.info(
                "Detected %d face(s) in %s.",
                result["face_count"],
                path.name,
            )

        return result

    # ------------------------------------------------------------------
    # OPTIMIZED IMAGE API
    # ------------------------------------------------------------------

    def detect_faces_from_image(
        self,
        image: np.ndarray,
    ) -> Dict[str, Any]:
        """Detect faces from an already-decoded OpenCV image.

        This avoids decoding the same image multiple times.
        """

        if image is None:
            return self._error_result(
                "Input image is None."
            )

        if not isinstance(image, np.ndarray):
            return self._error_result(
                "Input image must be a NumPy array."
            )

        if image.size == 0:
            return self._error_result(
                "Input image is empty."
            )

        if image.ndim == 2:
            gray = image
            height, width = image.shape[:2]

        elif image.ndim == 3:
            height, width = image.shape[:2]

            try:
                gray = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2GRAY,
                )
            except Exception as err:
                return self._error_result(
                    f"Failed to convert image to grayscale: {err}"
                )

        else:
            return self._error_result(
                "Unsupported image dimensions."
            )

        center_x = width / 2.0
        center_y = height / 2.0

        raw_candidates: List[
            Tuple[int, int, int, int, float]
        ] = []

        # --------------------------------------------------------------
        # PASS 1: Standard detection
        # --------------------------------------------------------------

        rects1, weights1 = self._run_cascade(
            gray,
            self.scale_factor,
            self.min_neighbors,
        )

        for i, (x, y, w, h) in enumerate(rects1):
            score = (
                float(weights1[i])
                if i < len(weights1)
                else 1.0
            )

            raw_candidates.append(
                (
                    int(x),
                    int(y),
                    int(w),
                    int(h),
                    score,
                )
            )

        # --------------------------------------------------------------
        # PASS 2: Histogram equalization fallback
        # --------------------------------------------------------------

        if not raw_candidates:
            gray_eq = cv2.equalizeHist(gray)

            rects2, weights2 = self._run_cascade(
                gray_eq,
                1.1,
                max(3, self.min_neighbors - 1),
            )

            for i, (x, y, w, h) in enumerate(rects2):
                score = (
                    float(weights2[i])
                    if i < len(weights2)
                    else 1.0
                )

                raw_candidates.append(
                    (
                        int(x),
                        int(y),
                        int(w),
                        int(h),
                        score,
                    )
                )

        # --------------------------------------------------------------
        # PASS 3: Sensitive fallback
        # --------------------------------------------------------------

        if not raw_candidates:
            rects3, weights3 = self._run_cascade(
                gray,
                1.05,
                3,
            )

            for i, (x, y, w, h) in enumerate(rects3):
                score = (
                    float(weights3[i])
                    if i < len(weights3)
                    else 0.5
                )

                raw_candidates.append(
                    (
                        int(x),
                        int(y),
                        int(w),
                        int(h),
                        score,
                    )
                )

        # --------------------------------------------------------------
        # FILTER
        # --------------------------------------------------------------

        filtered_candidates = []

        for x, y, w, h, score in raw_candidates:

            if h <= 0:
                continue

            aspect_ratio = float(w) / float(h)

            if (
                0.70 <= aspect_ratio <= 1.40
                and w >= 25
                and h >= 25
            ):
                filtered_candidates.append(
                    (
                        x,
                        y,
                        w,
                        h,
                        score,
                    )
                )

        # --------------------------------------------------------------
        # NMS
        # --------------------------------------------------------------

        clean_candidates = nms_boxes(
            filtered_candidates,
            iou_threshold=0.3,
        )

        # --------------------------------------------------------------
        # STRUCTURE RESULTS
        # --------------------------------------------------------------

        detected_faces: List[Dict[str, Any]] = []

        diagonal_half = (
            np.sqrt(width ** 2 + height ** 2) / 2.0
        )

        for x, y, w, h, score in clean_candidates:

            if score > 0:
                confidence = round(
                    min(
                        1.0,
                        max(0.0, score / 10.0),
                    ),
                    2,
                )
            else:
                confidence = 0.5

            fx_center = x + w / 2.0
            fy_center = y + h / 2.0

            distance = np.sqrt(
                (fx_center - center_x) ** 2
                + (fy_center - center_y) ** 2
            )

            norm_distance = (
                distance / (diagonal_half + 1e-5)
            )

            centrality_score = (
                (w * h)
                * (
                    1.0
                    - 0.4
                    * min(1.0, norm_distance)
                )
            )

            detected_faces.append(
                {
                    "box": {
                        "x": int(x),
                        "y": int(y),
                        "w": int(w),
                        "h": int(h),
                    },
                    "confidence": confidence,
                    "_rank_score": centrality_score,
                }
            )

        # Largest/most-central face first
        detected_faces.sort(
            key=lambda face: face["_rank_score"],
            reverse=True,
        )

        for face in detected_faces:
            del face["_rank_score"]

        logger.info(
            "Detected %d face(s).",
            len(detected_faces),
        )

        return {
            "success": True,
            "face_count": len(detected_faces),
            "faces": detected_faces,
            "image_size": {
                "width": int(width),
                "height": int(height),
            },
            "error": None,
        }

    # ------------------------------------------------------------------
    # CASCADE
    # ------------------------------------------------------------------

    def _run_cascade(
        self,
        gray_img: np.ndarray,
        scale_factor: float,
        min_neighbors: int,
    ) -> Tuple[Any, Any]:
        """Run CascadeClassifier detectMultiScale3 with fallback."""

        try:
            rects, rejects, reject_weights = (
                self.classifier.detectMultiScale3(
                    gray_img,
                    scaleFactor=scale_factor,
                    minNeighbors=min_neighbors,
                    minSize=self.min_size,
                    outputRejectLevels=True,
                )
            )

            return rects, reject_weights

        except Exception:
            rects = self.classifier.detectMultiScale(
                gray_img,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=self.min_size,
            )

            return rects, [1.0] * len(rects)

    # ------------------------------------------------------------------
    # ERROR
    # ------------------------------------------------------------------

    @staticmethod
    def _error_result(
        message: str,
    ) -> Dict[str, Any]:
        return {
            "success": False,
            "face_count": 0,
            "faces": [],
            "image_size": {
                "width": 0,
                "height": 0,
            },
            "error": str(message),
        }