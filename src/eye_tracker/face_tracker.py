import time
import cv2

import mediapipe as mp


class FaceTracker:
    def __init__(self, model_path: str) -> None:
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )

        self._face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

        self._start_time = time.perf_counter()

    def process(self, frame) -> list:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp_ms = int((time.perf_counter() - self._start_time) * 1000)

            result = self._face_landmarker.detect_for_video(image, timestamp_ms)

            if not result.face_landmarks:
                return []

            return result.face_landmarks[0]

    def close(self) -> None:
            self._face_landmarker.close()