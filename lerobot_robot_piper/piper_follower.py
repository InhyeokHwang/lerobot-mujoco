# piper_robot.py
from __future__ import annotations

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorNormMode
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from lerobot.robots.robot import Robot
from .piper_config import PiperRobotConfig
from .piper_bus import PiperMotorsBus

logger = logging.getLogger(__name__)


class PiperFollower(Robot):
    """
    Piper follower robot.
    - Piper SDK가 제공하는 "절대각(deg)"을 그대로 사용
    - LeRobot의 MotorCalibration(half-turn homing / range recording) 절차는 사용하지 않음
    """

    config_class = PiperRobotConfig
    name = "piper_follower"

    def __init__(self, config: PiperRobotConfig):
        super().__init__(config)
        self.config = config

        motors: dict[str, Motor] = {
            "joint_1": Motor(1, "piper_joint", MotorNormMode.DEGREES),
            "joint_2": Motor(2, "piper_joint", MotorNormMode.DEGREES),
            "joint_3": Motor(3, "piper_joint", MotorNormMode.DEGREES),
            "joint_4": Motor(4, "piper_joint", MotorNormMode.DEGREES),
            "joint_5": Motor(5, "piper_joint", MotorNormMode.DEGREES),
            "joint_6": Motor(6, "piper_joint", MotorNormMode.DEGREES),
        }

        self.bus = PiperMotorsBus(port=self.config.port, motors=motors)
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        """
        Observation motor features.
        - 기본: pos
        - 가능하면 vel/torque도 포함 (bus가 지원할 때)
        """
        features: dict[str, type] = {}
        for motor in self.bus.motors:
            features[f"{motor}.pos"] = float

            if getattr(self.bus, "supports_velocity", False): # TODO: 이거 확인하는거 검증
                features[f"{motor}.vel"] = float

            if getattr(self.bus, "supports_torque", False): # TODO: 이거 확인하는거 검증
                features[f"{motor}.torque"] = float
        return features
    
    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Camera features for observation space."""
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Combined observation features from motors and cameras."""
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Action features."""
        return {f"{motor}.pos": float for motor in self.bus.motors}
    
    @property
    def is_connected(self) -> bool:
        """Check if robot is connected."""
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    def calibrate(self) -> None:
        """
        LeRobot 추상 메서드 구현용.
        Piper는 SDK가 절대각을 제공하므로 별도 캘리브레이션 절차가 필요 없다고 가정.
        """
        self._calibrated = True

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        self.bus.enable_torque()

        if calibrate:
            self.calibrate()

        logger.info(f"{self} connected.")

    def configure(self) -> None:
        self.bus.configure_motors()

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        start = time.perf_counter()

        states = self.bus.sync_read_all_states()
        obs: dict[str, Any] = {}

        for m in self.bus.motors.keys():
            st = states.get(m, {})
            obs[f"{m}.pos"] = float(st.get("position", 0.0))
            # vel/torque는 bus가 지원할 때만 들어오도록 (bus가 0.0으로 채워도 OK)
            obs[f"{m}.vel"] = float(st.get("velocity", 0.0))
            obs[f"{m}.torque"] = float(st.get("torque", 0.0))

        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.async_read()

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} get_observation: {dt_ms:.1f}ms")
        return obs
    
    def _gripper_norm_to_mm(self, g: float) -> float:
        g = max(0.0, min(1.0, float(g)))

        close_mm, open_mm = self.config.gripper_range_mm

        # open=1, close=0
        return close_mm + g * (open_mm - close_mm)
    
    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        goal_pos = {k.removesuffix(".pos"): float(v) for k, v in action.items() if k.endswith(".pos")}

        joint_limits = getattr(self.config, "joint_limits", None)
        if joint_limits:
            for m, val in list(goal_pos.items()):
                if m in joint_limits:
                    lo, hi = joint_limits[m]
                    goal_pos[m] = max(float(lo), min(float(hi), float(val)))

        gripper_norm = goal_pos.pop("gripper", None)

        # joints 전송
        if goal_pos:
            self.bus.sync_write("Goal_Position", goal_pos)

        # gripper 전송 (0~1 -> mm)
        if gripper_norm is not None:
            target_mm = self._gripper_norm_to_mm(gripper_norm)
            self.bus._sdk.send_gripper_mm(target_mm, effort_n=self.config.gripper_effort_n, enable=True)

        # 반환은 “학습/로깅용”으로 원래 스케일(0~1)을 유지
        out = {f"{m}.pos": v for m, v in goal_pos.items()}
        if gripper_norm is not None:
            out["gripper.pos"] = float(max(0.0, min(1.0, gripper_norm)))
        return out

    @check_if_not_connected
    def disconnect(self) -> None:
        self.bus.disconnect(getattr(self.config, "disable_torque_on_disconnect", True))
        for cam in self.cameras.values():
            cam.disconnect()
        logger.info(f"{self} disconnected.")
