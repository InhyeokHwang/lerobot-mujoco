from dataclasses import dataclass, field
from lerobot.robots import RobotConfig
from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig

@RobotConfig.register_subclass("piper") # CLI
@dataclass
class PiperRobotConfig(RobotConfig):
    port: str    
    interface: str = "can"
    can_channel: str = "can0"
    cameras: dict[str, CameraConfig] = field(
        default_factory={
            "cam_1": OpenCVCameraConfig(
                index_or_path=2,
                fps=30,
                width=480,
                height=640,
            )
        }
    )