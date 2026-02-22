import numpy as np
from dataclasses import dataclass

@dataclass
class ControllerPose:
    pos: np.ndarray      # (3,)
    quat: np.ndarray     # (4,) xyzw

@dataclass
class ControllerState:
    trigger: float
    squeeze: float
    touchpad: bool
    thumbstick: bool
    button0: bool
    button1: bool

    trigger_value: float
    squeeze_value: float
    touchpad_xy: np.ndarray   # (2,)
    thumbstick_xy: np.ndarray # (2,)

@dataclass
class Quest3InputFrame:
    left_pose: ControllerPose
    right_pose: ControllerPose
    left_state: ControllerState
    right_state: ControllerState