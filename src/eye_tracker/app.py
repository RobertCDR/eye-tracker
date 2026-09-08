import cv2

from eye_tracker.eye_tracker import EyeTracker
from eye_tracker.face_tracker import FaceTracker

from eye_tracker.visualization import draw_eye_tracking

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

                draw_eye_tracking(frame, eye_tracking_result)

                cv2.imshow("Eye Tracker", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        face_tracker.close()

if __name__ == "__main__":
    main()
