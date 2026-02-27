# piper_bus.py
from __future__ import annotations

import logging
import time

from .piper_sdk_adapter import PiperSdkAdapter

from lerobot.motors.motors_bus import MotorsBusBase, Motor, MotorCalibration, NameOrID, Value
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

logger = logging.getLogger(__name__)

class PiperMotorsBus(MotorsBusBase):
    supports_velocity = True
    supports_torque = True

    default_timeout = 0.01

    def __init__(
        self,
        port: str,
        motors: dict[str, Motor],
    ):
        super().__init__(port, motors)
        self.port = port
        self._is_connected = False
        self._sdk = PiperSdkAdapter(port)

        # last-known state cache
        self._last_known: dict[str, dict[str, float]] = {
            name: {"position": 0.0, "velocity": 0.0, "torque": 0.0} for name in self.motors
        }

    @property
    def is_connected(self) -> bool:
        return self._is_connected and self._sdk.is_connected

    # ---- helpers ----
    def _motor_ids(self, motors: str | list[str] | None) -> list[int]:
        if motors is None:
            names = list(self.motors.keys())
        elif isinstance(motors, str):
            names = [motors]
        else:
            names = motors
        return [self.motors[n].id for n in names]

    def _name_list(self, motors: str | list[str] | None) -> list[str]:
        if motors is None:
            return list(self.motors.keys())
        if isinstance(motors, str):
            return [motors]
        return motors

    # ---- connection ----
    def connect(self, handshake: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self.__class__.__name__}('{self.port}') is already connected.")
        try:
            self._sdk.connect()
            self._is_connected = True

            if handshake:
                _ = self.sync_read("Present_Position")
            logger.info(f"{self.__class__.__name__} connected on {self.port}")
        except Exception as e:
            self._is_connected = False
            raise ConnectionError(f"Failed to connect Piper bus: {e}") from e

    def disconnect(self, disable_torque: bool = True) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(...)

        self._sdk.disconnect(disable_torque=disable_torque)
        self._is_connected = False

    # ---- torque ----
    def enable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        ids = self._motor_ids(motors)
        for attempt in range(num_retry + 1):
            try:
                self._sdk.enable_torque(ids)
                return
            except Exception:
                if attempt == num_retry:
                    raise
                time.sleep(0.01)

    def disable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        ids = self._motor_ids(motors)
        for attempt in range(num_retry + 1):
            try:
                self._sdk.disable_torque(ids)
                return
            except Exception:
                if attempt == num_retry:
                    raise
                time.sleep(0.01)


    def configure_motors(self) -> None:
        """Write implementation-specific recommended settings to every motor.

        Typical changes include shortening the return delay, increasing
        acceleration limits or disabling safety locks.
        """
        pass

    # ---- calibration (LeRobot interface requirement) ----
    def read_calibration(self) -> dict[str, MotorCalibration]:
        return {}

    def write_calibration(self, calibration_dict: dict[str, MotorCalibration], cache: bool = True) -> None:
        if cache:
            self.calibration.update(calibration_dict)

    # ---- read API (LeRobot 표준 key) ----
    def read(self, data_name: str, motor: str) -> Value:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if data_name != "Present_Position":
            raise ValueError(f"PiperMotorsBus.read supports only Present_Position (got {data_name})")

        out = self.sync_read(data_name, [motor])
        return out[motor]

    def sync_read(self, data_name: str, motors: str | list[str] | None = None) -> dict[str, Value]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        names = self._name_list(motors)
        ids = [self.motors[n].id for n in names]

        if data_name == "Present_Position":
            raw = self._sdk.read_joint_positions_deg(ids)  # {id: deg}
            result: dict[str, float] = {}
            for n in names:
                jid = self.motors[n].id
                raw_deg = float(raw.get(jid, self._last_known[n]["position"]))
                self._last_known[n]["position"] = raw_deg
                result[n] = raw_deg
            return result

        raise ValueError(f"Unsupported data_name: {data_name}")

    def sync_read_all_states(self, motors: str | list[str] | None = None) -> dict[str, dict[str, float]]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        names = self._name_list(motors)
        ids = [self.motors[n].id for n in names]

        raw_pos = self._sdk.read_joint_positions_deg(ids)          # {id: deg}
        raw_vel = self._sdk.read_joint_velocities_deg_s(ids)       # {id: deg/s} (없으면 {})
        raw_tau = self._sdk.read_joint_torques(ids)                # {id: tau}   (없으면 {})

        out: dict[str, dict[str, float]] = {}
        for n in names:
            jid = self.motors[n].id

            rp = float(raw_pos.get(jid, self._last_known[n]["position"]))
            rv = float(raw_vel.get(jid, self._last_known[n]["velocity"]))
            rt = float(raw_tau.get(jid, self._last_known[n]["torque"]))

            self._last_known[n]["position"] = rp
            self._last_known[n]["velocity"] = rv
            self._last_known[n]["torque"] = rt

            out[n] = {"position": rp, "velocity": rv, "torque": rt}

        return out

    # ---- write API ----
    def write(self, data_name: str, motor: str, value: Value) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if data_name != "Goal_Position":
            raise ValueError(f"PiperMotorsBus.write supports only Goal_Position (got {data_name})")

        self.sync_write("Goal_Position", {motor: value})

    def sync_write(self, data_name: str, values: Value | dict[str, Value]) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if data_name != "Goal_Position":
            raise ValueError(f"Unsupported write data_name: {data_name}")
        if not isinstance(values, dict):
            raise TypeError("sync_write expects dict[str, Value] for Piper")

        targets: dict[int, float] = {}
        gripper_norm: float | None = None

        for name, v in values.items():
            if name == "gripper":
                gripper_norm = float(v)
            else:
                targets[self.motors[name].id] = float(v)

        if targets:
            self._sdk.send_joint_positions_deg(targets)

        if gripper_norm is not None:
            # 0~1 클램프
            g = max(0.0, min(1.0, gripper_norm))

            # config가 bus엔 없으니: bus에서 mm 범위를 들고 있게 하거나,
            # follower가 mm로 변환해서 bus에 넘기는 방식 중 하나를 택해야 함.
            # => 추천: follower가 변환하고 bus는 mm를 받기
            raise RuntimeError("PiperMotorsBus expects gripper in mm; convert in PiperFollower")
