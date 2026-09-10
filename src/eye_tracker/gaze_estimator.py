from dataclasses import dataclass
from .eye_tracker import EyeTrackingResult

@dataclass
class GazeResult:
    horizontal_position: float
    vertical_position: float

    raw_horizontal_position: float
    raw_vertical_position: float

    valid: bool

class GazeEstimator:
    def __init__(
            self,
            smoothing: float = 0.2
    ) -> None:
        self.smoothing = smoothing
        self._horizontal = 0.5
        self._vertical = 0.5

    def process(
            self,
            eye_tracker_result: EyeTrackingResult
    ) -> GazeResult:
        left_eye = eye_tracker_result.left_eye
        right_eye = eye_tracker_result.right_eye

        if not left_eye.valid or not right_eye.valid:
            return GazeResult(
                horizontal_position=self._horizontal,
                vertical_position=self._vertical,
                raw_horizontal_position=self._horizontal,
                raw_vertical_position=self._vertical,
                valid=False
            )

        left_horizontal = 1.0 - left_eye.horizontal_position
        right_horizontal = right_eye.horizontal_position
        horizontal_position = (left_horizontal + right_horizontal) / 2

        left_vertical = 1.0 - left_eye.vertical_position
        right_vertical = right_eye.vertical_position
        vertical_position = (left_vertical + right_vertical) / 2

        self._horizontal += self.smoothing * (horizontal_position - self._horizontal)
        self._vertical += self.smoothing * (vertical_position - self._vertical)

        return GazeResult(
            horizontal_position=self._horizontal,
            raw_horizontal_position=horizontal_position,
            vertical_position=self._vertical,
            raw_vertical_position=vertical_position,
            valid=True
        )
