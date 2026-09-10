from dataclasses import dataclass

import numpy as np

from .calibration import Calibration


@dataclass
class ScreenPosition:
    x: int
    y: int
    valid: bool


class GazeMapper:
    def __init__(self) -> None:
        self._coefficients_x: np.ndarray | None = None
        self._coefficients_y: np.ndarray | None = None

    @property
    def calibrated(self) -> bool:
        return self._coefficients_x is not None and self._coefficients_y is not None

    def fit(self, calibration: Calibration) -> None:
        if len(calibration.samples) < 3:
            raise ValueError("At least 3 calibration samples are required for fitting.")

        gaze = np.array([[sample.gaze_x, sample.gaze_y, 1.0] for sample in calibration.samples])

        screen_x = np.array([sample.screen_x for sample in calibration.samples])
        screen_y = np.array([sample.screen_y for sample in calibration.samples])

        # Use least squares to find the best-fitting coefficients for mapping gaze to screen coordinates
        self._coefficients_x = np.linalg.lstsq(gaze, screen_x, rcond=None)[0]
        self._coefficients_y = np.linalg.lstsq(gaze, screen_y, rcond=None)[0]

    def map(self, gaze_x: float, gaze_y: float) -> ScreenPosition:
        if not self.calibrated:
            return ScreenPosition(x=0, y=0, valid=False)

        gaze = np.array([gaze_x, gaze_y, 1.0])

        screen_x = int(gaze @ self._coefficients_x)
        screen_y = int(gaze @ self._coefficients_y)

        return ScreenPosition(x=screen_x, y=screen_y, valid=True)


    def calibration_error(self, calibration: Calibration) -> float:
        if not self.calibrated:
            raise RuntimeError("GazeMapper is not calibrated.")

        total_error = 0.0

        for sample in calibration.samples:
            predicted_position = self.map(sample.gaze_x, sample.gaze_y)

            error_x = predicted_position.x - sample.screen_x
            error_y = predicted_position.y - sample.screen_y

            error = (error_x ** 2 + error_y ** 2) ** 0.5
            total_error += error

        return total_error / len(calibration.samples)
    