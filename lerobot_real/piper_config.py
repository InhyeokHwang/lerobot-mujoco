from dataclasses import dataclass, field
from lerobot.robots import RobotConfig
from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from typing import Tuple

gripper_range_mm: Tuple[float, float] = (0.0, 60.0)  # (close_mm, open_mm) 예시
gripper_effort_n: float = 2.0  # 0~5 권장 범위 안


@RobotConfig.register_subclass("piper") # CLI
@dataclass
class PiperRobotConfig(RobotConfig):
    port: str    
    interface: str = "can"
    can_channel: str = "can0"
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "cam_1": OpenCVCameraConfig(
                index_or_path=4,
                fps=30,
                width=640,
                height=480,
            )
        }
    )