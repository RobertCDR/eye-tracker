import cv2
import mediapipe as mp
from eye_tracker.eye_features import extract_eye_features
from eye_tracker.landmarks import (
    LEFT_EYE,
    RIGHT_EYE,
    LEFT_IRIS_CENTER,
    RIGHT_IRIS_CENTER
)
import time

MODEL_PATH = "models/face_landmarker.task"

def main() -> None:

    base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False
    )

    face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open camera.")

    try:
        start_time = time.time()

        while True:
            ret, frame = camera.read()

            if not ret:
                print("Failed to capture frame.")
                break

            # Convert the frame from BGR to RGB as MediaPipe uses RGB format
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp_ms = int((time.perf_counter() - start_time) * 1000)

            result = face_landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.face_landmarks:

                face = result.face_landmarks[0]

                left_eye_features = extract_eye_features(
                    face,
                    LEFT_EYE,
                    LEFT_IRIS_CENTER
                )

                right_eye_features = extract_eye_features(
                    face,
                    RIGHT_EYE,
                    RIGHT_IRIS_CENTER
                )

                print(
                    f"Left: H={left_eye_features.horizontal_position:.3f}, V={left_eye_features.vertical_position:.3f}, O={left_eye_features.openness:.3f} | "
                    f"Right: H={right_eye_features.horizontal_position:.3f}, V={right_eye_features.vertical_position:.3f}, O={right_eye_features.openness:.3f}"
                )

                for face_landmarks in result.face_landmarks:

                    height, width, _ = frame.shape

                    for landmark in face_landmarks:
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)

                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            cv2.imshow("Face Tracker", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        camera.release()
        face_landmarker.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()