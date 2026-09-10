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
        0.8,
        (255, 255, 255),
        2
    )

    status = "TRACKING" if result.valid else "NOT TRACKING"

    cv2.putText(
        frame,
        f"Status: {status}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    if result.valid:
        cv2.putText(
            frame,
            "LEFT EYE",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"H: {result.left_eye.horizontal_position:.3f}",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        cv2.putText(
            frame,
            f"V: {result.left_eye.vertical_position:.3f}",
            (20, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        cv2.putText(
            frame,
            f"Open: {result.left_eye.openness:.3f}",
            (20, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        cv2.putText(
            frame,
            "RIGHT EYE",
            (20, 230),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"H: {result.right_eye.horizontal_position:.3f}",
            (20, 260),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        cv2.putText(
            frame,
            f"V: {result.right_eye.vertical_position:.3f}",
            (20, 285),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        cv2.putText(
            frame,
            f"Open: {result.right_eye.openness:.3f}",
            (20, 310),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )


def draw_gaze_estimation(
        frame,
        raw_horizontal: float,
        raw_vertical: float,
        smoothed_horizontal: float,
        smoothed_vertical: float,
        valid: bool,
) -> None:
    height, width, _ = frame.shape

    if valid:
        gaze_x = int(smoothed_horizontal * width)
        gaze_y = int(smoothed_vertical * height)

        cv2.circle(frame, (gaze_x, gaze_y), 5, (0, 0, 255), -1)

        cv2.putText(
            frame,
            f"Raw H: {raw_horizontal:.3f}  V: {raw_vertical:.3f}",
            (20, 350),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        cv2.putText(
            frame,
            f"Smooth H: {smoothed_horizontal:.3f}  V: {smoothed_vertical:.3f}",
            (20, 375),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
