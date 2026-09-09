from dataclasses import dataclass
from mediapipe.tasks.python.components.containers import NormalizedLandmark

@dataclass
class EyeFeatures:
    iris_x: float
    iris_y: float
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
        raise ValueError("Eye corners must not overlap.")

    return (iris_x * eye_x + iris_y * eye_y) / eye_length_squared


def _normalized_vertical_position(
        iris: NormalizedLandmark, # type: ignore
        outer_corner: NormalizedLandmark, # type: ignore
        inner_corner: NormalizedLandmark # type: ignore
) -> float:

    eye_x = inner_corner.x - outer_corner.x
    eye_y = inner_corner.y - outer_corner.y

    eye_width = (eye_x ** 2 + eye_y ** 2) ** 0.5

    if eye_width == 0:
        raise ValueError("Eye corners must not overlap.")

    center_x = (outer_corner.x + inner_corner.x) / 2
    center_y = (outer_corner.y + inner_corner.y) / 2

    iris_x = iris.x - center_x
    iris_y = iris.y - center_y

    vertical_x = -eye_y / eye_width
    vertical_y = eye_x / eye_width

    return 0.5 + (iris_x * vertical_x + iris_y * vertical_y) / eye_width


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
            iris_x=0.0,
            iris_y=0.0,
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
    vertical_position = _normalized_vertical_position(iris_center, outer_corner, inner_corner)

    eye_height = _distance(top, bottom)

    openness = eye_height / eye_width

    return EyeFeatures(
        iris_x=iris_center.x,
        iris_y=iris_center.y,
        horizontal_position=horizontal_position,
        vertical_position=vertical_position,
        openness=openness,
        valid=openness > 0.10
    )