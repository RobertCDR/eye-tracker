from dataclasses import dataclass

from screeninfo import get_monitors


@dataclass
class Screen:
    width: int
    height: int

    @classmethod
    def primary(cls) -> "Screen":
        monitor = get_monitors()[0]
        return cls(width=monitor.width, height=monitor.height)

    def to_pixels(
            self,
            horizontal_position: float,
            vertical_position: float
    ) -> tuple[int, int]:
        x = int(horizontal_position * self.width)
        y = int(vertical_position * self.height)
        return x, y