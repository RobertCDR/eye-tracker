import time

import cv2

from eye_tracker.eye_tracker import EyeTracker
from eye_tracker.face_tracker import FaceTracker

from eye_tracker.gaze_estimator import GazeEstimator
from eye_tracker.gaze_mapper import GazeMapper

from eye_tracker.visualization import draw_eye_tracking, draw_gaze_estimation
from eye_tracker.calibration_visualization import draw_calibration_frame
from eye_tracker.gaze_visualization import create_gaze_frame

from eye_tracker.screen import Screen

from eye_tracker.calibration import Calibration, generate_calibration_points


MODEL_PATH = "models/face_landmarker.task"


def main() -> None:
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Failed to open camera.")

    face_tracker = FaceTracker(MODEL_PATH)
    eye_tracker = EyeTracker()

    gaze_estimator = GazeEstimator()

    screen = Screen.primary()

    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    calibration = Calibration()
    gaze_mapper = GazeMapper()

    points = generate_calibration_points(screen.width, screen.height)
    calibration.start(points)

    calibration_samples = []
    calibration_phase = "settling"  # can be "settling" or "collecting"
    calibration_phase_start = time.perf_counter()
    settling_duration = 1  # seconds
    samples_per_point = 45
    calibration_finished = False

    try:
        while True:
            ret, frame = camera.read()

            if not ret:
                print("Failed to read frame from camera.")
                break

            landmarks = face_tracker.process(frame)

            if landmarks:
                eye_tracking_result = eye_tracker.process(landmarks)

                gaze_result = gaze_estimator.process(eye_tracking_result)

                if not calibration.complete:
                    point = calibration.current_point

                    if point is not None:
                        calibration_frame = draw_calibration_frame(screen.width, screen.height, point)
                        cv2.imshow("Calibration", calibration_frame)

                    elapsed_time = time.perf_counter() - calibration_phase_start

                    if calibration_phase == "settling":
                        if elapsed_time >= settling_duration:
                            calibration_phase = "collecting"

                    elif calibration_phase == "collecting":
                        if gaze_result.valid:
                            calibration_samples.append((gaze_result.horizontal_position, gaze_result.vertical_position))

                        if len(calibration_samples) >= samples_per_point:
                            point = calibration.current_point

                            if point is not None:
                                calibration.add_samples(calibration_samples, point[0], point[1])

                                calibration_samples.clear()
                                calibration.next_point()

                                calibration_phase = "settling"
                                calibration_phase_start = time.perf_counter()
                            else:
                                print("Failed to get current calibration point.")

                if calibration.complete and not calibration_finished:
                    calibration_finished = True

                    gaze_mapper.fit(calibration)

                    # calibration_error = gaze_mapper.calibration_error(calibration)
                    # print(f"Calibration complete. Calibration error: {calibration_error:.2f} pixels")

                    cv2.destroyWindow("Calibration")

                    cv2.namedWindow("Gaze", cv2.WINDOW_NORMAL)
                    cv2.setWindowProperty("Gaze", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

                    print("Calibration complete.")

                if calibration_finished and gaze_result.valid:
                    screen_position = gaze_mapper.map(gaze_result.horizontal_position, gaze_result.vertical_position)

                    gaze_frame = create_gaze_frame(screen.width, screen.height, screen_position.x, screen_position.y)

                    cv2.imshow("Gaze", gaze_frame)

                draw_eye_tracking(frame, eye_tracking_result)
                draw_gaze_estimation(
                    frame,
                    gaze_result.raw_horizontal_position,
                    gaze_result.raw_vertical_position,
                    gaze_result.horizontal_position,
                    gaze_result.vertical_position,
                    gaze_result.valid
                )

                # cv2.imshow("Eye Tracker", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        face_tracker.close()

if __name__ == "__main__":
    main()
