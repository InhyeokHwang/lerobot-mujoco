from __future__ import annotations

import abc
import math
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union, Literal
from lerobot.motors.motors_bus import Motor, MotorsBusBase, MotorCalibration

from piper_sdk import C_PiperInterface_V2  

# Value 타입은 너 프로젝트에서 이미 정의되어 있을 가능성이 큼.
# 여기서는 편의상 Union으로 둠.
Value = Union[int, float, bool]


class PiperMotorsBus(MotorsBusBase):
    default_timeout_sec = 0.2
    state_poll_hz = 100  # piper_sdk 내부 poll/스레드가 사실상 담당 :contentReference[oaicite:3]{index=3}

    # 자주 쓰는 data_name alias들 (너 프로젝트 네이밍에 맞게 수정)
    _DATA_ALIASES = {
        # position
        "pos": "present_position",
        "position": "present_position",
        "present_pos": "present_position",
        "present_position": "present_position",
        "goal_pos": "goal_position",
        "target_position": "goal_position",
        "goal_position": "goal_position",

        # gripper
        "gripper": "present_gripper",
        "present_gripper": "present_gripper",
        "goal_gripper": "goal_gripper",
        "target_gripper": "goal_gripper",
        "goal_gripper": "goal_gripper",
    }

    def __init__(
        self,
        port: str,  # "can0"
        motors: dict[str, Motor],
        calibration: dict[str, MotorCalibration] | None = None,
        use_can_fd: bool = True,
        can_interface: str = "auto",
        piper_interface: type[C_PiperInterface_V2] = C_PiperInterface_V2,
        use_radians: bool = True,
        bitrate: int = 1_000_000,
        data_bitrate: int | None = 5_000_000,  # CAN-FD라면 여길 쓰고, can_activate.sh도 fd 지원이면 확장 가능
        can_activate_script: str = "can_activate.sh", 
        usb_address: str | None = None,  # 필요하면 can_activate.sh 3번째 인자로 넘김
    ):
        super().__init__(port, motors, calibration)
        self.port = port
        self.can_interface = can_interface
        self.use_can_fd = use_can_fd
        self.bitrate = bitrate
        self.data_bitrate = data_bitrate
        self.use_radians = use_radians
        self.can_activate_script = can_activate_script
        self.usb_address = usb_address

        self._piper_cls = piper_interface
        self._piper: C_PiperInterface_V2 | None = None
        self._is_connected = False

    # ----------------------------
    # helpers
    # ----------------------------
    def _canon_data(self, data_name: str) -> str:
        key = data_name.strip()
        if key in self._DATA_ALIASES:
            return self._DATA_ALIASES[key]
        # 이미 canonical인 4개는 허용
        if key in {"present_position", "present_gripper", "goal_position", "goal_gripper"}:
            return key
        raise ValueError(
            f"Unknown data_name '{data_name}'. "
            "Supported: present_position, present_gripper, goal_position, goal_gripper "
            f"(and aliases: {sorted(self._DATA_ALIASES.keys())})"
        )
    
    def _ensure_connected(self) -> None:
        if not self._is_connected or self._piper is None:
            raise RuntimeError("PiperMotorsBus is not connected. Call connect() first.")

    def _motor_to_joint_idx(self, motor_name: str) -> int:
        """
        motor_name: "joint_1".."joint_6"
        return: 1..6
        """
        if motor_name not in self.motors:
            raise KeyError(f"Unknown motor name: {motor_name}. Known: {list(self.motors.keys())}")
        # 가장 흔한 네이밍 joint_1..joint_6 가정
        if motor_name.startswith("joint_"):
            idx = int(motor_name.split("_", 1)[1])
            if 1 <= idx <= 6:
                return idx
        # fallback: Motor.id를 그대로 쓰는 경우
        mid = int(self.motors[motor_name].id)
        if 1 <= mid <= 6:
            return mid
        raise ValueError(f"Cannot map motor '{motor_name}' to joint index. Expected joint_1..joint_6 or id 1..6.")

    def _deg001_to_rad(self, deg001: int) -> float:
        # 0.001 degree -> rad
        return (deg001 / 1000.0) * (math.pi / 180.0)

    def _rad_to_deg001(self, rad: float) -> int:
        # rad -> 0.001 degree (int)
        return int(round((rad * 180.0 / math.pi) * 1000.0))

    def _mm001_to_m(self, mm001: int) -> float:
        # 0.001 mm -> m
        return mm001 * 1e-6

    def _m_to_mm001(self, m: float) -> int:
        # m -> 0.001 mm
        return int(round(m / 1e-6))

    def _read_joint_feedback_deg001(self) -> list[int]:
        """
        piper_sdk: GetArmJointMsgs() returns angles in 0.001 degrees. :contentReference[oaicite:4]{index=4}
        """
        self._ensure_connected()
        j = self._piper.GetArmJointMsgs().joint_state
        return [j.joint_1, j.joint_2, j.joint_3, j.joint_4, j.joint_5, j.joint_6]

    def _read_gripper_feedback_mm001(self) -> int:
        self._ensure_connected()
        g = self._piper.GetArmGripperMsgs().gripper_state
        return g.grippers_angle  # 0.001 mm :contentReference[oaicite:5]{index=5}

    def _send_joint_targets_deg001(self, targets_deg001: list[int]) -> None:
        """
        piper_sdk JointCtrl expects 0.001 degrees for each joint. :contentReference[oaicite:6]{index=6}
        """
        self._ensure_connected()
        self._piper.JointCtrl(
            targets_deg001[0],
            targets_deg001[1],
            targets_deg001[2],
            targets_deg001[3],
            targets_deg001[4],
            targets_deg001[5],
        )

    # ----------------------------
    # abstract interface impl
    # ----------------------------
    def connect(self, handshake: bool = True) -> None:
        if self._is_connected:
            return

        # 1) can_activate.sh
        script_path = self.can_activate_script
        if not os.path.isabs(script_path):
            here = os.path.dirname(os.path.abspath(__file__))
            script_path = os.path.join(here, script_path)

        cmd = ["bash", script_path, self.port, str(self.bitrate)]
        if self.usb_address:
            cmd.append(self.usb_address)

        subprocess.run(cmd, check=True)

        # 2) piper interface creation + port connection
        # can_auto_init=True면 내부에서 C_STD_CAN 만들고 Init/Read 스레드까지 관리. 
        self._piper = self._piper_cls(can_name=self.port, judge_flag=True, can_auto_init=True)
        # ConnectPort - CAN수신 쓰레드 시작
        self._piper.ConnectPort(can_init=False, piper_init=True, start_thread=True)  
        self._is_connected = True

        if handshake:
            # torque enable
            self.enable_torque(motors=None, num_retry=3)

    def disconnect(self, disable_torque: bool = True) -> None:
        if not self._is_connected:
            return
        if self._piper is None:
            self._is_connected = False
            return

        if disable_torque:
            try:
                self.disable_torque(motors=None, num_retry=1)
            except Exception:
                pass

        # SDK DisconnectPort가 CAN close까지 처리 
        try:
            self._piper.DisconnectPort(thread_timeout=0.2)
        finally:
            self._piper = None
            self._is_connected = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    # data_name for read, write, sync_read, sync_write
    # 1. "present_position"
    # 2. "present_gripper"
    # 3. "goal_position"
    # 4. "goal_gripper"
    def read(self, data_name: str, motor: str) -> Value:
        data = self._canon_data(data_name)
        jidx = self._motor_to_joint_idx(motor)

        if data == "present_position":
            deg001_all = self._read_joint_feedback_deg001()
            deg001 = deg001_all[jidx - 1]
            return self._deg001_to_rad(deg001) if self.use_radians else (deg001 / 1000.0)

        if data == "present_gripper":
            mm001 = self._read_gripper_feedback_mm001()
            return self._mm001_to_m(mm001)

        raise KeyError(f"Unsupported data_name for read(): {data_name} (canon={data})")

    def write(self, data_name: str, motor: str, value: Value) -> None:
        data = self._canon_data(data_name)
        jidx = self._motor_to_joint_idx(motor)

        if data == "goal_position":
            # 단일 관절 write라도 JointCtrl은 6축을 한 번에 받으니까
            # 현재 피드백을 기본으로 깔고 해당 축만 업데이트해서 전송
            cur = self._read_joint_feedback_deg001()
            if self.use_radians:
                cur[jidx - 1] = self._rad_to_deg001(float(value))
            else:
                # degrees(float) -> 0.001 degrees
                cur[jidx - 1] = int(round(float(value) * 1000.0))
            self._send_joint_targets_deg001(cur)
            return

        if data == "goal_gripper":
            # value: meters로 받는다고 가정 -> 0.001mm로 변환
            mm001 = self._m_to_mm001(float(value))
            # gripper_code 0x01: enable (또는 0x03 enable+clear error)
            self._ensure_connected()
            self._piper.GripperCtrl(gripper_angle=mm001, gripper_effort=0, gripper_code=0x01, set_zero=0x00)  # :contentReference[oaicite:11]{index=11}
            return

        raise KeyError(f"Unsupported data_name for write(): {data_name} (canon={data})")

    def sync_read(self, data_name: str, motors: str | list[str] | None = None) -> dict[str, Value]:
        data = self._canon_data(data_name)

        if motors is None:
            motor_list = list(self.motors.keys())
        elif isinstance(motors, str):
            motor_list = [motors]
        else:
            motor_list = list(motors)

        out: dict[str, Value] = {}

        if data == "present_position":
            deg001_all = self._read_joint_feedback_deg001()
            for m in motor_list:
                jidx = self._motor_to_joint_idx(m)
                deg001 = deg001_all[jidx - 1]
                out[m] = self._deg001_to_rad(deg001) if self.use_radians else (deg001 / 1000.0)
            return out

        if data == "present_gripper":
            mm001 = self._read_gripper_feedback_mm001()
            val = self._mm001_to_m(mm001)
            for m in motor_list:
                out[m] = val
            return out

        raise KeyError(f"Unsupported data_name for sync_read(): {data_name} (canon={data})")

    def sync_write(self, data_name: str, values: Value | dict[str, Value]) -> None:
        data = self._canon_data(data_name)

        if data == "goal_position":
            # values가 dict면 해당 모터들만 업데이트, Value 하나면 전체에 동일 적용(비추천이지만 지원)
            if isinstance(values, dict):
                cur = self._read_joint_feedback_deg001()
                for m, v in values.items():
                    jidx = self._motor_to_joint_idx(m)
                    if self.use_radians:
                        cur[jidx - 1] = self._rad_to_deg001(float(v))
                    else:
                        cur[jidx - 1] = int(round(float(v) * 1000.0))
                self._send_joint_targets_deg001(cur)
            else:
                # 전체 축 동일 값으로 맞추는 경우
                if self.use_radians:
                    deg001 = self._rad_to_deg001(float(values))
                else:
                    deg001 = int(round(float(values) * 1000.0))
                self._send_joint_targets_deg001([deg001] * 6)
            return

        if data == "goal_gripper":
            # dict면 첫 값만 사용(그리퍼는 사실상 1개)
            v = next(iter(values.values())) if isinstance(values, dict) else values
            mm001 = self._m_to_mm001(float(v))
            self._ensure_connected()
            self._piper.GripperCtrl(gripper_angle=mm001, gripper_effort=0, gripper_code=0x01, set_zero=0x00)  # :contentReference[oaicite:12]{index=12}
            return

        raise KeyError(f"Unsupported data_name for sync_write(): {data_name} (canon={data})")

    def enable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        self._ensure_connected()
        # EnablePiper -> EnableArm(7) 
        last_err: Optional[Exception] = None
        for _ in range(max(1, num_retry + 1)):
            try:
                self._piper.EnablePiper()
                return
            except Exception as e:
                last_err = e
                time.sleep(0.05)
        if last_err:
            raise last_err

    def disable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        self._ensure_connected()

        last_err: Optional[Exception] = None
        for _ in range(max(1, num_retry + 1)):
            try:
                self._piper.DisablePiper()
                return
            except Exception as e:
                last_err = e
                time.sleep(0.05)
        if last_err:
            raise last_err

    def read_calibration(self) -> dict[str, MotorCalibration]:
        # Piper 쪽은 보통 “모터 EEPROM 캘리브레이션” 같은 걸 직접 제공하지 않으니
        # LeRobot 레벨에서 쓰는 calibration cache만 반환하는 형태로 시작하는 게 안전.
        return dict(self.calibration)

    def write_calibration(self, calibration_dict: dict[str, MotorCalibration], cache: bool = True) -> None:
        # 실제 장치에 쓰는 API가 없다면(대부분), cache만 업데이트
        if cache:
            self.calibration.update(calibration_dict)
