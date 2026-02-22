# quest3_server.py
from multiprocessing import Array, Process
import numpy as np
import asyncio
from vuer import Vuer
from vuer.schemas import DefaultScene, MotionControllers


class Quest3NetworkServer:
    def __init__(self, cert=None, key=None, host="0.0.0.0", port=8012):
        # Vuer server
        self.app = Vuer(host=host, port=port, cert=cert, key=key, queries=dict(grid=False), queue_len=3)

        # shared buffers
        self.right_T = Array("d", 16, lock=True)
        self.left_T  = Array("d", 16, lock=True)

        self.right_state = Array("d", 14, lock=True)
        self.left_state  = Array("d", 14, lock=True)

        # event handler
        self.app.add_handler("CONTROLLER_MOVE")(self._on_controller)

        # IMPORTANT: spawn a per-session coroutine to enable controller streaming
        self.app.spawn(start=False)(self._session_main)

        # run server in another process
        self.proc = Process(target=self._run, daemon=True)
        self.proc.start()

    def _run(self):
        # run the Vuer server (blocking)
        self.app.run()

    async def _session_main(self, session, fps=60):
        """
        Called once per connected client session.
        This is where we must enable MotionControllers streaming,
        otherwise CONTROLLER_MOVE won't arrive.
        """
        session.set @ DefaultScene(grid=False, frameloop="always")

        # This line is the KEY
        session.upsert @ MotionControllers(
            stream=True,
            left=True,
            right=True,
            key="motion-controller",
        )

        # keep the session alive
        while True:
            await asyncio.sleep(1.0)

    async def _on_controller(self, event, *_):
        d = event.value

        if "right" in d and isinstance(d["right"], (list, tuple)) and len(d["right"]) == 16:
            self.right_T[:] = d["right"]

        if "left" in d and isinstance(d["left"], (list, tuple)) and len(d["left"]) == 16:
            self.left_T[:] = d["left"]

        self._write_state(self.right_state, d.get("rightState", {}), right=True)
        self._write_state(self.left_state,  d.get("leftState",  {}), right=False)

    @staticmethod
    def _write_state(buf, st, right=True):
        # NOTE: st is dict coming from vuer client
        tp = st.get("touchpadValue", [0.0, 0.0]) or [0.0, 0.0]
        ts = st.get("thumbstickValue", [0.0, 0.0]) or [0.0, 0.0]

        # right: a/b, left: x/y (your earlier code treated both as a/b; keep minimal here)
        if right:
            b0 = st.get("aButton", False)
            b1 = st.get("bButton", False)
        else:
            b0 = st.get("aButton", False)
            b1 = st.get("bButton", False)

        # store as floats (safer with Array('d', ...))
        buf[:] = [
            float(st.get("triggerValue", 0.0) or 0.0),   # 0
            float(st.get("squeezeValue", 0.0) or 0.0),   # 1
            float(bool(st.get("touchpad", False))),      # 2
            float(bool(st.get("thumbstick", False))),    # 3
            float(bool(b0)),                             # 4
            float(bool(b1)),                             # 5
            float(tp[0] if len(tp) > 0 else 0.0),        # 6
            float(tp[1] if len(tp) > 1 else 0.0),        # 7
            float(ts[0] if len(ts) > 0 else 0.0),        # 8
            float(ts[1] if len(ts) > 1 else 0.0),        # 9
            0.0, 0.0, 0.0, 0.0                           # padding 10~13
        ]

    # raw getters
    def get_left_matrix(self) -> np.ndarray:
        return np.array(self.left_T[:]).reshape(4, 4, order="F")

    def get_right_matrix(self) -> np.ndarray:
        return np.array(self.right_T[:]).reshape(4, 4, order="F")

    def get_left_state(self) -> np.ndarray:
        return np.array(self.left_state[:], dtype=float)

    def get_right_state(self) -> np.ndarray:
        return np.array(self.right_state[:], dtype=float)