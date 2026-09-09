import cv2

from eye_tracker.eye_tracker import EyeTracker
from eye_tracker.face_tracker import FaceTracker

from eye_tracker.gaze_estimator import GazeEstimator

from eye_tracker.visualization import draw_eye_tracking, draw_gaze_estimation

MODEL_PATH = "models/face_landmarker.task"


def main() -> None:
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Failed to open camera.")

    face_tracker = FaceTracker(MODEL_PATH)
    eye_tracker = EyeTracker()
    gaze_estimator = GazeEstimator()

    frame_count = 0
    print_every = 10

    try:
        while True:
            frame_count += 1
            ret, frame = camera.read()

            if not ret:
                print("Failed to read frame from camera.")
                break

            landmarks = face_tracker.process(frame)

            if landmarks:
                eye_tracking_result = eye_tracker.process(landmarks)

                gaze_result = gaze_estimator.process(eye_tracking_result)

                draw_eye_tracking(frame, eye_tracking_result)
                draw_gaze_estimation(
                    frame,
                    gaze_result.raw_horizontal_position,
                    gaze_result.raw_vertical_position,
                    gaze_result.horizontal_position,
                    gaze_result.vertical_position,
                    gaze_result.valid
                )

                if frame_count % print_every == 0:
                    left_eye = eye_tracking_result.left_eye
                    right_eye = eye_tracking_result.right_eye

                    print(
                        f"L: H={left_eye.horizontal_position:.3f} "
                        f"V={left_eye.vertical_position:.3f} "
                        f"Open={left_eye.openness:.3f} | "
                        f"R: H={right_eye.horizontal_position:.3f} "
                        f"V={right_eye.vertical_position:.3f} "
                        f"Open={right_eye.openness:.3f} | "
                        f"Gaze: H={gaze_result.horizontal_position:.3f} "
                        f"V={gaze_result.vertical_position:.3f} "
                        f"Valid={gaze_result.valid}"
                    )

                cv2.imshow("Eye Tracker", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        face_tracker.close()

if __name__ == "__main__":
    main()
