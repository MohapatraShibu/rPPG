"""
face ROI extractor using MediaPipe FaceLandmarker (Tasks API, v0.10+)
falls back to OpenCV Haar cascade if the .task model file is missing
"""
import os
import cv2
import numpy as np
import mediapipe.tasks as mp_tasks

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "face_landmarker.task")

# forehead + cheek landmark indices (478-point FaceMesh topology)
_ROI_INDICES = [
    # forehead band
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    # left cheek
    117, 118, 119, 120, 121, 128, 245,
    # right cheek
    346, 347, 348, 349, 350, 357, 465,
]

class FaceROIExtractor:
    def __init__(self):
        model_path = os.path.abspath(_MODEL_PATH)
        if os.path.exists(model_path):
            self._use_landmarks = True
            VisionRunningMode = mp_tasks.vision.RunningMode
            opts = mp_tasks.vision.FaceLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
                running_mode=VisionRunningMode.IMAGE,
                num_faces=1,
            )
            self._landmarker = mp_tasks.vision.FaceLandmarker.create_from_options(opts)
        else:
            self._use_landmarks = False
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._detector = cv2.CascadeClassifier(cascade_path)

    def mean_rgb(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        # return mean [R, G, B] of face skin ROI, or None if no face found
        if self._use_landmarks:
            return self._landmarks_rgb(frame_bgr)
        return self._haar_rgb(frame_bgr)

    def _landmarks_rgb(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        h, w = frame_bgr.shape[:2]
        rgb_np = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        import mediapipe as mp
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_np)
        result = self._landmarker.detect(image)
        if not result.face_landmarks:
            return None
        lm = result.face_landmarks[0]
        pts = np.array(
            [[int(lm[i].x * w), int(lm[i].y * h)] for i in _ROI_INDICES],
            dtype=np.int32,
        )
        mask = np.zeros((h, w), dtype=np.uint8)
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 255)
        pixels = rgb_np[mask == 255]
        return pixels.mean(axis=0) if len(pixels) else None

    def _haar_rgb(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._detector.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces) == 0:
            return None
        x, y, fw, fh = faces[0]
        # use forehead (top 30%) + cheeks (middle 40%, left/right thirds)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        forehead = rgb[y: y + int(fh * 0.30), x + int(fw * 0.2): x + int(fw * 0.8)]
        mid_y = y + int(fh * 0.35)
        lc = rgb[mid_y: mid_y + int(fh * 0.25), x: x + int(fw * 0.25)]
        rc = rgb[mid_y: mid_y + int(fh * 0.25), x + int(fw * 0.75): x + fw]
        regions = [r for r in (forehead, lc, rc) if r.size > 0]
        if not regions:
            return None
        all_pixels = np.concatenate([r.reshape(-1, 3) for r in regions])
        return all_pixels.mean(axis=0)

    def close(self):
        if self._use_landmarks:
            self._landmarker.close()
