from __future__ import annotations
from typing import Any

import numpy as np
from lerobot.robots import Robot
from lerobot.motors import Motor, MotorNormMode
from lerobot.cameras import make_cameras_from_configs
import time
import threading
from lerobot.motors.motors_bus import MotorCalibration
from .piper_config import PiperRobotConfig
from .piper_bus import PiperMotorsBus

class PiperRobot(Robot):
    config_class = PiperRobotConfig
    name = "piper"

    def __init__(self, config: PiperRobotConfig):
        super().__init__(config)
        self.bus = PiperMotorsBus(
            port=self.config.port,
            # id: 조인트 id
            # model: 그냥 라벨링
            # norm_mode : RANGE_0_100, RANGE_M100_100, DEGREES
            # 0~100 정규화인지, -100~100 정규화인지, Degrees 값인지
            motors={ 
                "joint_1": Motor(id=1, model="piper_joint", norm_mode=MotorNormMode.DEGREES),
                "joint_2": Motor(id=2, model="piper_joint", norm_mode=MotorNormMode.DEGREES),
                "joint_3": Motor(id=3, model="piper_joint", norm_mode=MotorNormMode.DEGREES),
                "joint_4": Motor(id=4, model="piper_joint", norm_mode=MotorNormMode.DEGREES),
                "joint_5": Motor(id=5, model="piper_joint", norm_mode=MotorNormMode.DEGREES),
                "joint_6": Motor(id=6, model="piper_joint", norm_mode=MotorNormMode.DEGREES),
            }
        )
    
    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            "joint_1.pos": float,
            "joint_2.pos": float,
            "joint_3.pos": float,
            "joint_4.pos": float,
            "joint_5.pos": float,
            "joint_6.pos": float
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.cameras[cam].height, self.cameras[cam].width, 3) for cam in self.cameras
        }

    @property
    def observation_features(self) -> dict:
        return {**self._motors_ft, **self._cameras_ft}
    
    @property
    def action_features(self) -> dict:
        return self._motors_ft
    
    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        if not self.is_calibrated and calibrate:
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()

    def disconnect(self) -> None:
        self.bus.disconnect()
        for cam in self.cameras.values():
            cam.disconnect()

    @property
    def is_calibrated(self) -> bool:
        # 1) 파일이 이미 저장되어 로드되었거나, 2) calibration dict가 motors와 일치하면 calibrated로 간주
        if getattr(self, "calibration", None) is None:
            return False
        return set(self.calibration.keys()) == set(self.bus.motors.keys())

    def calibrate(self) -> None:
        """
        Piper용 소프트웨어 캘리브레이션:
        - 홈(중립) 자세에서 현재 각도를 homing_offset으로 저장
        - 사용자가 관절을 끝까지 움직이는 동안 min/max 범위 기록
        """
        self.bus.connect(handshake=False)

        # 안전: 토크 끄고 손으로 움직이게
        try:
            self.bus.disable_torque(num_retry=1)
        except Exception:
            pass

        # 1) Home pose offsets
        input(
            "\n[CALIB] 로봇을 '중립(Home) 자세'로 맞춘 뒤 ENTER를 누르세요.\n"
            " - 이 자세가 소프트웨어 상의 0 기준이 됩니다.\n"
        )
        home_deg = self._read_joint_deg()

        # 여기서는 "present = raw + homing_offset" 같은 Feetech 정의가 아니라,
        # LeRobot 레벨에서 사용할 offset을 그냥 저장하는 용도라고 보면 됨.
        # homing_offset을 -home로 잡아두면 (raw + offset)에서 home이 0이 됨.
        homing_offsets = {m: -home_deg[m] for m in self.bus.motors.keys()}

        # 2) Range recording
        print(
            "\n[CALIB] 이제 각 관절을 천천히 끝까지 움직여 보세요.\n"
            " - 충분히 왕복하면 됩니다.\n"
            " - 멈추려면 ENTER를 누르세요.\n"
        )

        stop_flag = {"stop": False}

        def _wait_enter():
            input()
            stop_flag["stop"] = True

        t = threading.Thread(target=_wait_enter, daemon=True)
        t.start()

        # 초기값을 home 기준으로 세팅 (offset 적용 후 값 기준으로 range를 잡고 싶으면 아래처럼)
        # offset 적용 기준 range:
        #   val_cal = raw + homing_offset
        mins = {m: (home_deg[m] + homing_offsets[m]) for m in self.bus.motors.keys()}
        maxs = {m: (home_deg[m] + homing_offsets[m]) for m in self.bus.motors.keys()}

        hz = 30.0
        dt = 1.0 / hz
        while not stop_flag["stop"]:
            cur_deg = self._read_joint_deg()
            for m in self.bus.motors.keys():
                val_cal = cur_deg[m] + homing_offsets[m]  # home이 0이 되는 좌표
                if val_cal < mins[m]:
                    mins[m] = val_cal
                if val_cal > maxs[m]:
                    maxs[m] = val_cal
            time.sleep(dt)

        # 3) Build MotorCalibration dict
        calibration: dict[str, MotorCalibration] = {}
        for motor_name, motor in self.bus.motors.items():
            calibration[motor_name] = MotorCalibration(
                id=motor.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor_name],
                range_min=mins[motor_name],
                range_max=maxs[motor_name],
            )

        # 4) Save / cache
        self.calibration = calibration
        self.bus.write_calibration(calibration, cache=True)

        # Robot base 쪽에 _save_calibration()이 있다면 저장
        if hasattr(self, "_save_calibration"):
            self._save_calibration()

        print("\n[CALIB] 완료!")
        for m in self.bus.motors.keys():
            print(
                f" - {m}: offset={homing_offsets[m]:.3f} deg, "
                f"range=[{mins[m]:.3f}, {maxs[m]:.3f}] deg (home=0 기준)"
            )

        # 토크는 사용 패턴에 따라 켜도 되고, 그대로 둬도 됨.
        # 보통은 connect() 후 configure()에서 enable_torque 하니까 여기서는 안 켜도 됨.