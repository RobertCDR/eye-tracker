import cv2

from eye_tracker.eye_tracker import EyeTracker
from eye_tracker.face_tracker import FaceTracker


MODEL_PATH = "models/face_landmarker.task"


def main() -> None:
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Failed to open camera.")

    face_tracker = FaceTracker(MODEL_PATH)
    eye_tracker = EyeTracker()

    try:
        while True:
            ret, frame = camera.read()

            if not ret:
                print("Failed to read frame from camera.")
                break

            landmarks = face_tracker.process(frame)

            if landmarks:
                eye_tracking_result = eye_tracker.process(landmarks)

                print(
                    f"Left: H={eye_tracking_result.left_eye.horizontal_position:.3f}, "
                    f"V={eye_tracking_result.left_eye.vertical_position:.3f}, "
                    f"O={eye_tracking_result.left_eye.openness:.3f} | "
                    f"Right: H={eye_tracking_result.right_eye.horizontal_position:.3f}, "
                    f"V={eye_tracking_result.right_eye.vertical_position:.3f}, "
                    f"O={eye_tracking_result.right_eye.openness:.3f}"
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
