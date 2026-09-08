from dataclasses import dataclass
from mediapipe.tasks.python.components.containers import NormalizedLandmark

@dataclass
class EyeFeatures:
    horizontal_position: float
    vertical_position: float
    openness: float
    valid: bool


def _distance(first: NormalizedLandmark, second: NormalizedLandmark) -> float: # type: ignore
    dx = first.x - second.x
    dy = first.y - second.y

    return (dx ** 2 + dy ** 2) ** 0.5


def _normalized_horizontal_position(
        iris: NormalizedLandmark, # type: ignore
        outer_corner: NormalizedLandmark, # type: ignore
        inner_corner: NormalizedLandmark # type: ignore
) -> float:

    eye_x = inner_corner.x - outer_corner.x
    eye_y = inner_corner.y - outer_corner.y

    iris_x = iris.x - outer_corner.x
    iris_y = iris.y - outer_corner.y

    eye_length_squared = eye_x ** 2 + eye_y ** 2

    if eye_length_squared == 0:
        raise ValueError("Eye length is zero, cannot compute horizontal position.")

    return (iris_x * eye_x + iris_y * eye_y) / eye_length_squared


def _normalized_vertical_position(
        iris: NormalizedLandmark, # type: ignore
        top: NormalizedLandmark, # type: ignore
        bottom: NormalizedLandmark # type: ignore
) -> float:

    eye_x = bottom.x - top.x
    eye_y = bottom.y - top.y

    iris_x = iris.x - top.x
    iris_y = iris.y - top.y

    eye_length_squared = eye_x ** 2 + eye_y ** 2

    if eye_length_squared == 0:
        raise ValueError("Eye length is zero, cannot compute vertical position.")

    return (iris_x * eye_x + iris_y * eye_y) / eye_length_squared


def extract_eye_features(
        landmarks: list[NormalizedLandmark], # type: ignore
        eye_definitions: dict[str, int],
        iris_center_index: int
) -> EyeFeatures:

    required_indices = [
        eye_definitions["outer_corner"],
        eye_definitions["inner_corner"],
        eye_definitions["top"],
        eye_definitions["bottom"],
        iris_center_index
    ]

    if (
        not landmarks
        or any(index < 0 or index >= len(landmarks) for index in required_indices)
    ):
        return EyeFeatures(
            horizontal_position=0.0,
            vertical_position=0.0,
            openness=0.0,
            valid=False
        )

    outer_corner = landmarks[eye_definitions["outer_corner"]]
    inner_corner = landmarks[eye_definitions["inner_corner"]]
    top = landmarks[eye_definitions["top"]]
    bottom = landmarks[eye_definitions["bottom"]]
    iris_center = landmarks[iris_center_index]

    eye_width = _distance(outer_corner, inner_corner)

    if eye_width == 0:
        raise ValueError("Eye width is zero, cannot compute features.")

    horizontal_position = _normalized_horizontal_position(iris_center, outer_corner, inner_corner)
    vertical_position = _normalized_vertical_position(iris_center, top, bottom)

    eye_height = _distance(top, bottom)

    openness = eye_height / eye_width

    return EyeFeatures(
        horizontal_position=horizontal_position,
        vertical_position=vertical_position,
        openness=openness,
        valid=openness > 0.25
    )