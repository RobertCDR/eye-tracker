import cv2
import numpy as np

def draw_calibration_frame(width: int, height: int, point: tuple[int, int]) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    cv2.circle(frame, point, 15, (255, 255, 255), -1)

    return frame
