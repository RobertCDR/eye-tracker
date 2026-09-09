import cv2
import numpy as np


def create_gaze_frame(width: int, height: int, x: int, y: int) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))

    cv2.circle(frame, (x, y), 20, (0, 0, 255), -1)

    return frame