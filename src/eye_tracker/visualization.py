import cv2

from .eye_tracker import EyeTrackingResult


def draw_eye_tracking(
        frame,
        result: EyeTrackingResult,
) -> None:
    height, width, _ = frame.shape

    if result.valid:
        left_x = int(result.left_eye.iris_x * width)
        left_y = int(result.left_eye.iris_y * height)

        right_x = int(result.right_eye.iris_x * width)
        right_y = int(result.right_eye.iris_y * height)

        cv2.circle(frame, (left_x, left_y), 5, (0, 255, 0), -1)
        cv2.circle(frame, (right_x, right_y), 5, (0, 255, 0), -1)

    cv2.putText(
        frame,
        "EYE TRACKING",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2
    )

    status = "TRACKING" if result.valid else "NOT TRACKING"

    cv2.putText(
        frame,
        f"Status: {status}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )
