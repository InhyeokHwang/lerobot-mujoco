import mujoco
import numpy as np

CTRL2EE_RIGHT = np.eye(4, dtype=np.float64)

CTRL2EE_LEFT = np.eye(4, dtype=np.float64)
CTRL2EE_LEFT[:3, :3] = np.array([
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0,-1.0],
], dtype=np.float64) 


def _as_int(i) -> int:
    return int(np.asarray(i).reshape(-1)[0])


def _as_vec(x, n: int) -> np.ndarray:
    # (n,), (1,n), (n,1) 등 뭐가 와도 (n,)로 정리
    return np.asarray(x, dtype=np.float64).reshape(-1)[:n]


def T_from_pos_quat_xyzw(pos: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    """pos(3,), quat_xyzw(4,) -> 4x4. MuJoCo quat is wxyz."""
    pos = _as_vec(pos, 3)
    quat_xyzw = _as_vec(quat_xyzw, 4)

    q_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)

    # MuJoCo Python binding이 요구하는 shape에 안전하게 맞춤: (9,1) and (4,1)
    R = np.empty((9, 1), dtype=np.float64)
    mujoco.mju_quat2Mat(R, q_wxyz.reshape(4, 1))

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.reshape(3, 3)
    T[:3, 3] = pos
    return T


def set_mocap_from_T(data: mujoco.MjData, mocap_id: int, T: np.ndarray) -> None:
    """4x4 -> data.mocap_pos/quat (quat wxyz)"""
    mocap_id = _as_int(mocap_id)
    T = np.asarray(T, dtype=np.float64)

    data.mocap_pos[mocap_id] = _as_vec(T[:3, 3], 3)

    q = np.empty((4, 1), dtype=np.float64)
    R_flat = np.asarray(T[:3, :3], dtype=np.float64).reshape(9, 1)
    mujoco.mju_mat2Quat(q, R_flat)

    # data.mocap_quat[mocap_id]가 (1,4)로 잡히는 환경도 있어서 (4,)로 넣어줌
    data.mocap_quat[mocap_id] = q.reshape(4,)


def T_from_mocap(model: mujoco.MjModel, data: mujoco.MjData, mocap_id: int) -> np.ndarray:
    mocap_id = _as_int(mocap_id)

    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = _as_vec(data.mocap_pos[mocap_id], 3)

    # 여기서 핵심: mocap_quat이 (1,4)로 나와도 (4,1)로 정리해서 전달
    q = _as_vec(data.mocap_quat[mocap_id], 4).reshape(4, 1)

    R = np.empty((9, 1), dtype=np.float64)
    mujoco.mju_quat2Mat(R, q)

    T[:3, :3] = R.reshape(3, 3)
    return T


class Controller:
    def __init__(
        self,
        use_rotation: bool = True,
        pos_scale: float = 1.0,
        R_fix: np.ndarray | None = None,   
    ):
        self.use_rotation = use_rotation
        self.pos_scale = float(pos_scale)

        self.R_fix = np.eye(3) if R_fix is None else np.asarray(R_fix, dtype=np.float64).reshape(3, 3)

        self._active = False
        self._p_ctrl0 = None    # (3,)
        self._R_ctrl0 = None    # (3,3)
        self._T_m0 = None       # (4,4)

    def update(self, squeeze: float, T_ctrl: np.ndarray, T_mocap_now: np.ndarray):
        on = squeeze > 0.5

        # rising edge
        if on and not self._active:
            self._active = True
            self._p_ctrl0 = T_ctrl[:3, 3].copy()

            self._R_ctrl0 = (self.R_fix @ T_ctrl[:3, :3]).copy()
            self._T_m0 = T_mocap_now.copy()
            return False, None

        # falling edge
        if (not on) and self._active:
            self._active = False
            self._p_ctrl0 = None
            self._R_ctrl0 = None
            self._T_m0 = None
            return False, None

        # hold
        if self._active and (self._p_ctrl0 is not None) and (self._T_m0 is not None) and (self._R_ctrl0 is not None):
            dp = (T_ctrl[:3, 3] - self._p_ctrl0) * self.pos_scale

            T_des = self._T_m0.copy()
            T_des[:3, 3] = self._T_m0[:3, 3] + dp

            if self.use_rotation:
                R_ctrl_now = self.R_fix @ T_ctrl[:3, :3]
                # 컨트롤러 상대 회전: ΔR = R0^T * Rnow
                dR = self._R_ctrl0.T @ R_ctrl_now
                # mocap 기준 자세에 상대 회전 적용
                T_des[:3, :3] = self._T_m0[:3, :3] @ dR

            return True, T_des

        return False, None