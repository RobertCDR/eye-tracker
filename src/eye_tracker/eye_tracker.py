from dataclasses import dataclass

from mediapipe.tasks.python.components.containers import NormalizedLandmark

from .eye_features import EyeFeatures, extract_eye_features
from .landmarks import (
    LEFT_EYE,
    RIGHT_EYE,
    LEFT_IRIS_CENTER,
    RIGHT_IRIS_CENTER
)


@dataclass
class EyeTrackingResult:
    left_eye: EyeFeatures
    right_eye: EyeFeatures

    @property
    def valid(self) -> bool:
        return self.left_eye.valid and self.right_eye.valid


class EyeTracker:
    def process(
            self,
            landmarks: list[NormalizedLandmark] # type: ignore
    ) -> EyeTrackingResult:
        left_eye_features = extract_eye_features(
            landmarks,
            LEFT_EYE,
            LEFT_IRIS_CENTER
        )

        right_eye_features = extract_eye_features(
            landmarks,
            RIGHT_EYE,
            RIGHT_IRIS_CENTER
        )

        return EyeTrackingResult(
            left_eye=left_eye_features,
            right_eye=right_eye_features
        )