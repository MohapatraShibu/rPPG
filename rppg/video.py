# v reader that yields per-frame BGR arrays with metadata
import cv2
import numpy as np
from dataclasses import dataclass

@dataclass
class VideoMeta:
    fps: float
    total_frames: int
    width: int
    height: int
    duration_s: float


def open_video(path: str) -> tuple[cv2.VideoCapture, VideoMeta]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, VideoMeta(fps=fps, total_frames=total, width=w, height=h,
                          duration_s=total / fps)

def iter_chunks(cap: cv2.VideoCapture, meta: VideoMeta,
                chunk_sec: float = 5.0):
    # yield (chunk_index, rgb_trace) where rgb_trace is (N,3) float32
    from rppg.extractor import FaceROIExtractor
    extractor = FaceROIExtractor()
    chunk_frames = int(meta.fps * chunk_sec)
    chunk_idx = 0
    buf: list[np.ndarray] = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # downscale large frames to max 640px on longest side for speed
            h, w = frame.shape[:2]
            if max(h, w) > 640:
                scale = 640 / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            rgb = extractor.mean_rgb(frame)
            if rgb is not None:
                buf.append(rgb)

            if len(buf) >= chunk_frames:
                yield chunk_idx, np.array(buf[:chunk_frames], dtype=np.float32)
                chunk_idx += 1
                buf = buf[chunk_frames:]   # no overlap - strict 5s windows
    finally:
        extractor.close()

    # leftover frames (partial chunk - skip if < 2 s worth)
    if len(buf) >= int(meta.fps * 2):
        yield chunk_idx, np.array(buf, dtype=np.float32)
