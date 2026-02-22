# quest3_teleop.py
from .quest3_server import Quest3NetworkServer
from .quest3_preprocessor import Quest3Preprocessor
from .quest3_input import *
from pathlib import Path
_here = Path(__file__).resolve().parent

class Quest3Teleop:
    def __init__(self):
        _here = Path(__file__).resolve().parent
        self.server = Quest3NetworkServer(cert=str(_here / "cert.pem"),key=str(_here / "key.pem"))
        self.pre = Quest3Preprocessor()

    def read(self) -> Quest3InputFrame:
        Tl = self.server.get_left_matrix()
        Tr = self.server.get_right_matrix()

        Tl, Tr = self.pre.process(Tl, Tr)

        lp, lq = self.pre.mat_to_pose(Tl)
        rp, rq = self.pre.mat_to_pose(Tr)

        ls = self._parse_state(self.server.get_left_state())
        rs = self._parse_state(self.server.get_right_state())

        return Quest3InputFrame(
            left_pose = ControllerPose(lp, lq),
            right_pose = ControllerPose(rp, rq),
            left_state = ls,
            right_state = rs,
        )

    @staticmethod
    def _parse_state(arr):
        return ControllerState(
            trigger        = arr[0],
            squeeze        = arr[1],
            touchpad       = bool(arr[2]),
            thumbstick     = bool(arr[3]),
            button0        = bool(arr[4]),
            button1        = bool(arr[5]),
            trigger_value  = arr[0],
            squeeze_value  = arr[1],
            touchpad_xy    = arr[6:8],
            thumbstick_xy  = arr[8:10],
        )