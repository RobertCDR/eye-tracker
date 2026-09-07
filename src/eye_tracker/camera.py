import time

import cv2

def main() -> None:
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open camera.")

    # Request preferred camera settings
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    camera.set(cv2.CAP_PROP_FPS, 30)

    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = camera.get(cv2.CAP_PROP_FPS)

    print(f"Actual camera settings: {actual_width}x{actual_height} @ {actual_fps} FPS")

    frame_count = 0
    fps_start_time = time.perf_counter()

    try:
        while True:
            frame_start = time.perf_counter()

            ret, frame = camera.read()

            if not ret:
                print("Failed to capture frame.")
                break

            frame_count += 1

            # Calculate and display FPS every second
            elapsed_time = time.perf_counter() - fps_start_time

            if elapsed_time >= 1.0:
                measured_fps = frame_count / elapsed_time
                print(f"Measured FPS: {measured_fps:.2f}")

                frame_count = 0
                fps_start_time = time.perf_counter()

            # Show the captured frame in a window
            cv2.imshow("Camera Feed", frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_time = time.perf_counter() - frame_start

            _ = frame_time  # Placeholder for any additional processing time calculations

    finally:
        camera.release()
        cv2.destroyAllWindows()
            
if __name__ == "__main__":
    main()
