import logging
import math
import subprocess
import time
from typing import Optional
from pathlib import Path

from piper_sdk import C_PiperInterface_V2
from lerobot.utils.errors import DeviceNotConnectedError

logger = logging.getLogger(__name__)


class PiperSdkAdapter:
    """
    Notes about piper_sdk units (from SDK docstrings):
      - Joint feedback (GetArmJointMsgs): 0.001 degrees :contentReference[oaicite:5]{index=5}
      - Joint command (JointCtrl): inputs are 0.001 degrees :contentReference[oaicite:6]{index=6}
      - High-speed feedback (GetArmHighSpdInfoMsgs):
          motor_speed: 0.001 rad/s
          effort: 0.001 N·m (doc says N/m but gripper doc says N·m; treat as torque-like) :contentReference[oaicite:7]{index=7}
    """

    def __init__(self, port: str):
        # port is CAN channel name like "can0"
        self.port = port
        self.can_setup_script = str(Path(__file__).resolve().parent / "can_activate.sh")
        self._connected = False

        # Pass can_name explicitly (C_PiperInterface_V2 uses can_name internally) :contentReference[oaicite:8]{index=8}
        self.interface = C_PiperInterface_V2(can_name=port)

        # internal cache: last joint targets (deg), useful if you only command subset
        self._last_targets_deg: dict[int, float] = {i: 0.0 for i in range(1, 7)}

    def _ensure_can_ready(self):
        if self.can_setup_script:
            subprocess.run(["bash", self.can_setup_script], check=True)

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        if self._connected:
            return

        self._ensure_can_ready()

        self.interface.ConnectPort(can_init=False, piper_init=True, start_thread=True)

        msg = self.interface.GetArmJointMsgs()
        if msg is None:
            raise ConnectionError("GetArmJointMsgs() returned None after ConnectPort()")

        self._connected = True

        try:
            self.set_motion_mode(ctrl_mode=0x01, move_mode=0x01, speed=50, is_mit_mode=0x00)
        except Exception as e:
            logger.warning(f"[PIPER] MotionCtrl_2 failed: {e}")

        # enable 재시도 + enabled 확인
        for k in range(20):
            try:
                self.interface.EnableArm(7)
            except Exception as e:
                logger.warning(f"[PIPER] EnableArm failed (try {k}): {e}")

            time.sleep(0.1)

            if self.is_enabled():
                logger.info("[PIPER] Arm enabled.")
                break
        else:
            logger.warning("[PIPER] Arm not enabled after retries.")

    def disconnect(self, disable_torque: bool = True) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self.__class__.__name__}('{self.port}') is not connected.")

        if disable_torque:
            try:
                self.disable_torque()
            except Exception as e:
                logger.warning(f"Failed to disable torque during disconnect: {e}")

        # Disable + stop threads/close CAN
        self.interface.DisablePiper()  # :contentReference[oaicite:11]{index=11}
        self.interface.DisconnectPort()

        self._connected = False

    def set_motion_mode(
        self,
        ctrl_mode: int = 0x01,
        move_mode: int = 0x01,
        speed: int = 50,
        is_mit_mode: int = 0x00,
        residence_time: int = 0,
        installation_pos: int = 0x00,
    ) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError("Device not connected")

        self.interface.MotionCtrl_2(
            ctrl_mode=ctrl_mode,
            move_mode=move_mode,
            move_spd_rate_ctrl=speed,
            is_mit_mode=is_mit_mode,
            residence_time=residence_time,
            installation_pos=installation_pos,
        )

    def is_enabled(self) -> bool:
        if not self.is_connected:
            return False
        info = self.interface.GetArmLowSpdInfoMsgs()
        if info is None:
            return False
        try:
            return (
                info.motor_1.foc_status.driver_enable_status and
                info.motor_2.foc_status.driver_enable_status and
                info.motor_3.foc_status.driver_enable_status and
                info.motor_4.foc_status.driver_enable_status and
                info.motor_5.foc_status.driver_enable_status and
                info.motor_6.foc_status.driver_enable_status
            )
        except AttributeError:
            return False

    # ---- torque(enable/disable) ----
    def enable_torque(self, joint_ids: Optional[list[int]] = None) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError("Device not connected")

        # SDK: EnableArm motor_num 1..7, 7=all motors :contentReference[oaicite:12]{index=12}
        if joint_ids is None:
            self.interface.EnableArm(7)
            return

        for jid in joint_ids:
            if jid == 7:
                self.interface.EnableArm(7)
            else:
                self.interface.EnableArm(int(jid))

    def disable_torque(self, joint_ids: Optional[list[int]] = None) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError("Device not connected")

        # SDK: DisableArm motor_num 1..7, 7=all motors :contentReference[oaicite:13]{index=13}
        if joint_ids is None:
            self.interface.DisableArm(7)
            return

        for jid in joint_ids:
            if jid == 7:
                self.interface.DisableArm(7)
            else:
                self.interface.DisableArm(int(jid))

    # ---- read ----
    def read_joint_positions_deg(self, joint_ids: list[int]) -> dict[int, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError("Device not connected")

        msg = self.interface.GetArmJointMsgs()
        if msg is None:
            raise RuntimeError("GetArmJointMsgs() returned None")

        js = msg.joint_state  # ArmMsgFeedBackJointStates :contentReference[oaicite:14]{index=14}
        all_deg = {
            1: js.joint_1 * 0.001,
            2: js.joint_2 * 0.001,
            3: js.joint_3 * 0.001,
            4: js.joint_4 * 0.001,
            5: js.joint_5 * 0.001,
            6: js.joint_6 * 0.001,
        }
        return {jid: all_deg[jid] for jid in joint_ids}

    def read_joint_velocities_deg_s(self, joint_ids: list[int]) -> dict[int, float]:
        """
        Uses high-speed motor feedback:
          motor_speed is 0.001 rad/s -> deg/s = motor_speed*0.001*180/pi :contentReference[oaicite:15]{index=15}
        If unavailable, returns 0.0 for those joints.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError("Device not connected")

        info = self.interface.GetArmHighSpdInfoMsgs()
        if info is None:
            return {jid: 0.0 for jid in joint_ids}

        # info.motor_1 ... info.motor_6
        motor = {
            1: info.motor_1,
            2: info.motor_2,
            3: info.motor_3,
            4: info.motor_4,
            5: info.motor_5,
            6: info.motor_6,
        }

        out: dict[int, float] = {}
        for jid in joint_ids:
            ms = getattr(motor[jid], "motor_speed", None)
            if ms is None:
                out[jid] = 0.0
            else:
                out[jid] = (ms * 0.001) * (180.0 / math.pi)
        return out

    def read_joint_torques(self, joint_ids: list[int]) -> dict[int, float]:
        """
        Uses high-speed motor feedback:
          effort is 0.001 (torque-like) -> Nm approx = effort*0.001 :contentReference[oaicite:16]{index=16}
        If unavailable, returns 0.0.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError("Device not connected")

        info = self.interface.GetArmHighSpdInfoMsgs()
        if info is None:
            return {jid: 0.0 for jid in joint_ids}

        motor = {
            1: info.motor_1,
            2: info.motor_2,
            3: info.motor_3,
            4: info.motor_4,
            5: info.motor_5,
            6: info.motor_6,
        }

        out: dict[int, float] = {}
        for jid in joint_ids:
            eff = getattr(motor[jid], "effort", None)
            out[jid] = 0.0 if eff is None else (eff * 0.001)
        return out

    # ---- write ----
    def send_joint_positions_deg(self, targets: dict[int, float]) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError("Device not connected")

        # update cache
        for jid, deg in targets.items():
            if jid < 1 or jid > 6:
                raise ValueError(f"joint_id must be 1..6, got {jid}")
            self._last_targets_deg[jid] = float(deg)

        def to_mdeg(deg: float) -> int:
            # 0.001 degree unit (milli-degree)
            return int(round(deg * 1000.0))

        j1 = to_mdeg(self._last_targets_deg[1])
        j2 = to_mdeg(self._last_targets_deg[2])
        j3 = to_mdeg(self._last_targets_deg[3])
        j4 = to_mdeg(self._last_targets_deg[4])
        j5 = to_mdeg(self._last_targets_deg[5])
        j6 = to_mdeg(self._last_targets_deg[6])

        self.interface.JointCtrl(j1, j2, j3, j4, j5, j6)  # :contentReference[oaicite:18]{index=18}


    # ---- gripper ----
    def enable_gripper(self, clear_error: bool = False) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError("Device not connected")
        code = 0x03 if clear_error else 0x01
        # angle/effort은 현재값 유지 목적이면 0으로 보내도 SDK가 limit 처리함
        self.interface.GripperCtrl(gripper_angle=0, gripper_effort=0, gripper_code=code, set_zero=0x00)

    def disable_gripper(self, clear_error: bool = False) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError("Device not connected")
        code = 0x02 if clear_error else 0x00
        self.interface.GripperCtrl(gripper_angle=0, gripper_effort=0, gripper_code=code, set_zero=0x00)

    def send_gripper_mm(self, target_mm: float, effort_n: float = 2.0, enable: bool = True) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError("Device not connected")

        mm_milli = int(round(target_mm * 1000.0))  # 0.001mm
        eff_milli = int(round(max(0.0, min(5.0, effort_n)) * 1000.0))  # 0.001N/m

        code = 0x01 if enable else 0x00
        self.interface.GripperCtrl(
            gripper_angle=mm_milli,
            gripper_effort=eff_milli,
            gripper_code=code,
            set_zero=0x00,
        )