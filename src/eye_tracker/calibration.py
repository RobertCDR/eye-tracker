from dataclasses import dataclass


@dataclass
class CalibrationSample:
    gaze_x: float
    gaze_y: float
    screen_x: int
    screen_y: int


class Calibration:
    def __init__(self):
        self.samples: list[CalibrationSample] = []
        self.points: list[tuple[int, int]] = []
        self.current_point_index: int = 0

    def start(self, points: list[tuple[int, int]]) -> None:
        if not points:
            raise ValueError("No calibration points provided.")

        self.points = points
        self.current_point_index = 0

    @property
    def current_point(self) -> tuple[int, int] | None:
        if self.current_point_index >= len(self.points):
            return None
        
        return self.points[self.current_point_index]

    @property
    def complete(self) -> bool:
        return self.current_point_index >= len(self.points)

    def next_point(self) -> None:
        self.current_point_index += 1

    def add_sample(
            self,
            gaze_x: float,
            gaze_y: float,
            screen_x: int,
            screen_y: int
    ) -> None:
        self.samples.append(CalibrationSample(
            screen_x=screen_x,
            screen_y=screen_y,
            gaze_x=gaze_x,
            gaze_y=gaze_y
            )
        )

    def add_samples(
            self,
            gaze_samples: list[tuple[float, float]],
            screen_x: int,
            screen_y: int
    ) -> None:
        if not gaze_samples:
            raise ValueError("No gaze samples provided.")

        # Calculate the average gaze position from the provided samples
        average_gaze_x = sum(sample[0] for sample in gaze_samples) / len(gaze_samples)
        average_gaze_y = sum(sample[1] for sample in gaze_samples) / len(gaze_samples)

        self.add_sample(
            gaze_x=average_gaze_x,
            gaze_y=average_gaze_y,
            screen_x=screen_x,
            screen_y=screen_y
        )


def generate_calibration_points(
        screen_width: int,
        screen_height: int,
) -> list[tuple[int, int]]:
    positions = [
        (0.1, 0.1),  # Top-left
        (0.5, 0.1),  # Top-center
        (0.9, 0.1),  # Top-right
        (0.1, 0.5),  # Middle-left
        (0.5, 0.5),  # Center
        (0.9, 0.5),  # Middle-right
        (0.1, 0.9),  # Bottom-left
        (0.5, 0.9),  # Bottom-center
        (0.9, 0.9)   # Bottom-right
    ]

    return [(int(x * screen_width), int(y * screen_height)) for x, y in positions]
