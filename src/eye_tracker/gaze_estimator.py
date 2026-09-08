from dataclasses import dataclass
from .eye_tracker import EyeTrackingResult

@dataclass
class GazeResult:
    horizontal_position: float
    vertical_position: float
    valid: bool

class GazeEstimator:
    def process(
            self,
            eye_tracker_result: EyeTrackingResult
    ) -> GazeResult:
        left_eye_features = eye_tracker_result.left_eye
        right_eye_features = eye_tracker_result.right_eye

        if not left_eye_features.valid or not right_eye_features.valid:
            return GazeResult(horizontal_position=0.0, vertical_position=0.0, valid=False)

        horizontal_position = (left_eye_features.horizontal_position + right_eye_features.horizontal_position) / 2
        vertical_position = (left_eye_features.vertical_position + right_eye_features.vertical_position) / 2

        return GazeResult(horizontal_position=horizontal_position, vertical_position=vertical_position, valid=True)
