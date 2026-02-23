from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import time
import numpy as np
import mujoco
import mink
from loop_rate_limiters import RateLimiter

from quest3.quest3_teleop import Quest3Teleop
from mink_ik.quest3_utils import (
    Controller,
    T_from_pos_quat_xyzw,
    set_mocap_from_T,
    T_from_mocap,
)

from piper_config import PiperRobotConfig
from piper_follower import PiperFollower


# =========================
# Config
# =========================
_HERE = Path(__file__).parent

# MuJoCo model is used as: FK + IK internal representation
_XML = Path(__file__).parent.parent / "description" / "agilex_piper" / "scene.xml"

SOLVER = "daqp"

# IK convergence (strictness)
POS_THRESHOLD = 1e-4
ORI_THRESHOLD = 1e-4

MAX_ITERS = 20
DAMPING = 1e-3

# Control loop (hardware)
CTRL_HZ = 50.0
# Dataset record rate
REC_HZ = 20.0
REC_DT = 1.0 / REC_HZ

# Safety: per-step max delta (rad) clamp (tune!)
MAX_DQ_PER_STEP = 0.06  # rad per control tick (~3.4deg @50Hz)

# Optional: low-pass smoothing on command
LPF_ALPHA = 0.35  # 0..1 (higher = less smoothing)


# =========================
# Dataset
# =========================
class EpisodeDataset:
    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._frames: List[Dict] = []
        self._episode_idx = 0

    def add_frame(self, frame: Dict, task: str):
        fr = dict(frame)
        fr["task"] = task
        self._frames.append(fr)

    def clear_episode_buffer(self):
        self._frames.clear()

    def save_episode(self):
        if len(self._frames) == 0:
            print("[DATASET] skip save (empty episode)")
            return
        ep_path = self.out_dir / f"episode_{self._episode_idx:04d}.npz"
        np.savez_compressed(ep_path, frames=np.array(self._frames, dtype=object))
        print(f"[DATASET] saved: {ep_path} (frames={len(self._frames)})")
        self._episode_idx += 1
        self._frames.clear()


# =========================
# MuJoCo helpers (same style as your sim code)
# =========================
TARGET_RADIUS = 0.05
TARGET_RGBA = [1.0, 0.1, 0.1, 0.9]


def load_model_with_big_target(xml_path: Path, target_body_name: str = "target") -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(xml_path.as_posix())

    try:
        body = spec.body(target_body_name)
    except Exception:
        body = None

    if body is None:
        body = spec.worldbody.add_body(name=target_body_name, mocap=True)

    r = float(TARGET_RADIUS)
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[r, r, r],
        rgba=TARGET_RGBA,
        contype=0,
        conaffinity=0,
    )
    return spec.compile()


def reset_to_home_if_exists(model: mujoco.MjModel, data: mujoco.MjData, key_name: str = "home") -> None:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)


def pick_site_name(model: mujoco.MjModel) -> str:
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper") != -1:
        return "gripper"
    if model.nsite > 0:
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, 0)
    raise RuntimeError("No site exists in this model. Cannot run site-based FrameTask.")


def site_pose(model: mujoco.MjModel, data: mujoco.MjData, site_id: int) -> tuple[np.ndarray, np.ndarray]:
    # world position + world quat(wxyz)
    pos = data.site_xpos[site_id].copy()
    xmat = data.site_xmat[site_id].reshape(3, 3)
    quat = np.empty(4, dtype=np.float64)  # wxyz
    mujoco.mju_mat2Quat(quat, xmat.reshape(-1))
    return pos, quat


def get_obs_state(model: mujoco.MjModel, data: mujoco.MjData, site_id: int) -> np.ndarray:
    pos, quat = site_pose(model, data, site_id)
    return np.concatenate([pos, quat], axis=0)  # (7,)


def _vec1(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def converge_ik(
    configuration: mink.Configuration,
    end_effector_task: mink.FrameTask,
    posture_task: mink.PostureTask,
    dt: float,
    solver: str,
    pos_threshold: float,
    ori_threshold: float,
    max_iters: int,
) -> bool:
    tasks = [end_effector_task, posture_task]
    subdt = dt / max(1, max_iters)

    for _ in range(max_iters):
        vel = mink.solve_ik(configuration, tasks, subdt, solver, damping=DAMPING)
        configuration.integrate_inplace(vel, subdt)

        err = end_effector_task.compute_error(configuration)
        if np.linalg.norm(err[:3]) <= pos_threshold and np.linalg.norm(err[3:]) <= ori_threshold:
            return True
    return False


# =========================
# Hardware IO adapters (YOU may need to adjust these 2 functions)
# =========================
def extract_qpos_from_obs(obs: Dict) -> np.ndarray:
    """
    PiperFollower.get_observation() 결과에서 관절각(qpos) 6개를 rad로 꺼내기.

    너가 예시로 쓰던 포맷이:
      obs["joint_1.pos"] ... obs["joint_6.pos"]

    만약 단위가 degree/0.001deg면 여기서 rad로 변환해야 함.
    """
    q = np.array([obs[f"joint_{i}.pos"] for i in range(1, 7)], dtype=np.float64)
    # TODO: 필요시 변환 (예: deg -> rad)
    # q = np.deg2rad(q)
    return q


def build_action_from_qpos(q_cmd: np.ndarray) -> Dict:
    """
    PiperFollower.send_action() 입력 딕셔너리 생성.

    예시:
      { "joint_1.pos": q1, ..., "joint_6.pos": q6 }
    """
    return {f"joint_{i}.pos": float(q_cmd[i - 1]) for i in range(1, 7)}


# =========================
# Safety utilities
# =========================
def clamp_step(q_meas: np.ndarray, q_cmd: np.ndarray, max_dq: float) -> np.ndarray:
    dq = np.clip(q_cmd - q_meas, -max_dq, max_dq)
    return q_meas + dq


def lowpass(prev: Optional[np.ndarray], new: np.ndarray, alpha: float) -> np.ndarray:
    if prev is None:
        return new.copy()
    return (1.0 - alpha) * prev + alpha * new


# =========================
# Main
# =========================
def main():
    TASK_NAME = "piper_singlearm_quest3_real"
    NUM_DEMO = 50

    dataset = EpisodeDataset(out_dir=_HERE / "dataset_out_real_single")
    teleop = Quest3Teleop()

    # Build MuJoCo model for FK/IK
    model = load_model_with_big_target(_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    configuration = mink.Configuration(model)

    ee_site = pick_site_name(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site)
    if site_id == -1:
        raise RuntimeError(f"EE site not found: {ee_site}")
    print("[INFO] Using EE site:", ee_site)

    # IK tasks
    end_effector_task = mink.FrameTask(
        frame_name=ee_site,
        frame_type="site",
        position_cost=1.0,
        orientation_cost=0.1,
        lm_damping=1.0,
    )
    posture_task = mink.PostureTask(model=model, cost=1e-2)

    # Mocap target id
    mocap_id = model.body("target").mocapid

    # Right controller only
    follow = Controller(use_rotation=True, pos_scale=1.0, R_fix=np.eye(3))

    # Connect robot
    cfg = PiperRobotConfig(port="can0")
    robot = PiperFollower(cfg)
    robot.connect(calibrate=False)

    rate = RateLimiter(frequency=CTRL_HZ, warn=False)

    episode_id = 0
    record_flag = False
    rec_accum = 0.0

    prev_reset = False
    prev_done = False

    last_q_cmd: Optional[np.ndarray] = None

    def hard_reset():
        nonlocal record_flag, follow, last_q_cmd
        dataset.clear_episode_buffer()
        record_flag = False
        follow = Controller(use_rotation=True, pos_scale=1.0, R_fix=follow.R_fix)
        last_q_cmd = None

        # "reset" in hardware: hold current pose (safe)
        obs = robot.get_observation()
        q_meas = extract_qpos_from_obs(obs)
        robot.send_action(build_action_from_qpos(q_meas))
        print("[RESET] episode buffer cleared + hold current pose")

        # Also reset mocap target to current EE pose (based on measured q)
        data.qpos[: len(q_meas)] = q_meas
        mujoco.mj_forward(model, data)
        mink.move_mocap_to_frame(model, data, "target", ee_site, "site")
        mujoco.mj_forward(model, data)

        # posture target anchored to current pose
        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)

    try:
        print("[INFO] Connected. Right controller only.")
        print("  - Hold squeeze: move target")
        print("  - Button B (button1): reset episode (hold + target reset)")
        print("  - Button A (button0): save episode (if recording)")

        hard_reset()

        while episode_id < NUM_DEMO:
            dt = rate.dt

            # -----------------------------
            # 1) Read hardware observation
            # -----------------------------
            obs = robot.get_observation()
            q_meas = extract_qpos_from_obs(obs)

            # Sync MuJoCo state for FK/IK
            data.qpos[: len(q_meas)] = q_meas
            mujoco.mj_forward(model, data)
            configuration.update(data.qpos)

            # -----------------------------
            # 2) Read Quest3 frame
            # -----------------------------
            frame = teleop.read()

            # reset: button B rising edge
            reset_now = bool(frame.right_state.button1)
            reset = reset_now and (not prev_reset)
            prev_reset = reset_now
            if reset:
                hard_reset()
                rate.sleep()
                continue

            # done: button A rising edge
            done_now = bool(frame.right_state.button0)
            done = done_now and (not prev_done)
            prev_done = done_now
            if done:
                print(f"[DONE] pressed. record_flag={record_flag}, frames={len(dataset._frames)}")
                if record_flag:
                    dataset.save_episode()
                    episode_id += 1
                    print(f"[DATASET] Episode done. episode_id={episode_id}/{NUM_DEMO}")
                hard_reset()
                continue

            # controller pose -> 4x4
            T_ctrl = _T_from_pos_quat_xyzw(frame.right_pose.pos, frame.right_pose.quat)

            # current mocap pose -> 4x4
            T_moc_now = _T_from_mocap(model, data, mocap_id)

            # squeeze-gated teleop
            ok, T_des = follow.update(frame.right_state.squeeze, T_ctrl, T_moc_now)
            if ok and T_des is not None:
                _set_mocap_from_T(data, mocap_id, T_des)

            # start recording when target starts being updated
            if (not record_flag) and ok:
                record_flag = True
                print("[DATASET] Start recording")

            # -----------------------------
            # 3) Mocap -> IK target, solve IK
            # -----------------------------
            T_wt = mink.SE3.from_mocap_name(model, data, "target")
            end_effector_task.set_target(T_wt)

            posture_task.set_target_from_configuration(configuration)

            converge_ik(
                configuration,
                end_effector_task=end_effector_task,
                posture_task=posture_task,
                dt=dt,
                solver=SOLVER,
                pos_threshold=POS_THRESHOLD,
                ori_threshold=ORI_THRESHOLD,
                max_iters=MAX_ITERS,
            )

            # command from IK
            q_cmd = np.array(configuration.q, dtype=np.float64).copy()
            q_cmd = q_cmd[:6]  # piper 6dof

            # -----------------------------
            # 4) Safety shaping: clamp + lpf
            # -----------------------------
            q_cmd = clamp_step(q_meas, q_cmd, MAX_DQ_PER_STEP)
            q_cmd = lowpass(last_q_cmd, q_cmd, LPF_ALPHA)
            last_q_cmd = q_cmd.copy()

            # -----------------------------
            # 5) Send to hardware
            # -----------------------------
            robot.send_action(build_action_from_qpos(q_cmd))

            # -----------------------------
            # 6) Record at 20Hz
            # -----------------------------
            rec_accum += dt
            if rec_accum >= REC_DT:
                rec_accum -= REC_DT

                # observation.state: EE pose from measured q (FK)
                obs_state = get_obs_state(model, data, site_id)

                # observation.target: mocap pose
                target_pos = _vec1(data.mocap_pos[mocap_id])[:3]
                target_quat = _vec1(data.mocap_quat[mocap_id])[:4]
                target_state = np.concatenate([target_pos, target_quat], axis=0)  # (7,)

                if record_flag:
                    dataset.add_frame(
                        {
                            "observation.state": obs_state,
                            "observation.target": target_state,
                            "observation.qpos_meas": q_meas.copy(),
                            "action.qpos_cmd": q_cmd.copy(),
                        },
                        task=TASK_NAME,
                    )

            rate.sleep()

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt")
    finally:
        # If you want: save partial episode on exit
        if record_flag and len(dataset._frames) > 0:
            print("[INFO] Saving partial episode before exit.")
            dataset.save_episode()
        robot.disconnect()


if __name__ == "__main__":
    main()