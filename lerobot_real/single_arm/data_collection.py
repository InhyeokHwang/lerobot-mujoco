from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import time
import numpy as np
import mujoco
import mink
from loop_rate_limiters import RateLimiter

from quest3.quest3_teleop import Quest3Teleop

from mink_ik.single_arm_mink_ik import (
    pick_ee_site,
    site_pose,
    check_reached_single,
    initialize_model,
)

from mink_ik.quest3_utils import (
    Controller,
    T_from_pos_quat_xyzw,
    set_mocap_from_T,
    T_from_mocap,
)

from lerobot_real.hardware_config.piper.piper_config import PiperRobotConfig
from lerobot_real.hardware_config.piper.piper_follower import PiperFollower


_HERE = Path(__file__).parent

SOLVER = "daqp"

# IK
POSTURE_COST = 1e-3
MAX_ITERS_PER_CYCLE = 20
DAMPING = 1e-3

# Convergence thresholds
POS_THRESHOLD = 1e-4
ORI_THRESHOLD = 1e-4

# Control loop rate
RATE_HZ = 100.0

# Data collection rate
REC_HZ = 100.0
REC_DT = 1.0 / REC_HZ

# Safety: per-step max delta clamp (deg per control tick)
MAX_DQ_PER_STEP_DEG = 3.5 

HOME_REACH_TOL_DEG = 1.5  # tolerance degree    
HOME_TIMEOUT_SEC = 12.0   # timeout
HOME_SETTLE_SEC = 0.3     # settle

# roll <-> yaw
R_SWAP_XZ = np.array([
    [0.0, 0.0, 1.0],
    [0.0,-1.0, 0.0],
    [1.0, 0.0, 0.0],
], dtype=np.float64) 

# 부호 교정
R_FLIP_RP = np.array([
    [-1.0,  0.0,  0.0],
    [ 0.0, 1.0,  0.0],
    [ 0.0,  0.0,  1.0],
], dtype=np.float64)  # = Rz(pi), det=+1

class EpisodeDataset:
    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # frame = episode buffer
        self._frames: List[Dict] = [] # observation.state, observation.target, action.qpos, task
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
        # npz compress
        np.savez_compressed(ep_path, frames=np.array(self._frames, dtype=object))
        print(f"[DATASET] saved: {ep_path} (frames={len(self._frames)})")

        self._episode_idx += 1
        self._frames.clear()


def get_obs_state(model: mujoco.MjModel, data: mujoco.MjData, site_id: int) -> np.ndarray:
    pos, quat = site_pose(model, data, site_id)  # (3,), (4,wxyz)
    return np.concatenate([pos, quat], axis=0)   # (7,)

def _vec1(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)

def clamp_step_deg(q_meas_deg: np.ndarray, q_cmd_deg: np.ndarray, max_dq_deg: float) -> np.ndarray:
    dq = np.clip(q_cmd_deg - q_meas_deg, -max_dq_deg, max_dq_deg)
    return q_meas_deg + dq

def extract_qpos_deg_from_obs(obs: Dict) -> np.ndarray:
    # PiperFollower.get_observation() keys: "joint_1.pos" ... "joint_6.pos"
    return np.array([obs[f"joint_{i}.pos"] for i in range(1, 7)], dtype=np.float64)

def build_action_from_qpos_deg(q_cmd_deg: np.ndarray) -> Dict:
    # PiperFollower.send_action expects keys ending with ".pos"
    return {f"joint_{i}.pos": float(q_cmd_deg[i - 1]) for i in range(1, 7)}

def get_home_qpos_deg(model: mujoco.MjModel, key_id: int) -> np.ndarray:
    # key_qpos can be (nkey*nq,) OR (nkey, nq) depending on mujoco/python binding
    kq = np.asarray(model.key_qpos)

    if kq.ndim == 2:
        q_home = kq[key_id]                 
    else:
        q_home = kq[key_id * model.nq : (key_id + 1) * model.nq]  

    q_home = np.asarray(q_home, dtype=np.float64).reshape(-1)    
    return np.rad2deg(q_home[:6]).reshape(6,)                     

def pre_move_to_keyframe_home(
    *,
    model: mujoco.MjModel,
    key_id: int,
    robot: PiperFollower,
    teleop: Quest3Teleop,
    rate: RateLimiter,
    tol_deg: float,
    timeout_sec: float,
    allow_abort: bool = True
) -> bool:
    q_goal_deg = get_home_qpos_deg(model, key_id)
    print(f"[HOME] one-shot goal_deg = {np.round(q_goal_deg, 2)}")

    robot.send_action(build_action_from_qpos_deg(q_goal_deg))

    t_start = time.perf_counter()
    prev_abort = False

    while True:
        # abort (B)
        frame = teleop.read()
        if allow_abort:
            abort_now = bool(frame.right_state.button1)
            abort = abort_now and (not prev_abort)
            prev_abort = abort_now
            if abort:
                obs = robot.get_observation()
                q_meas_deg = extract_qpos_deg_from_obs(obs)
                robot.send_action(build_action_from_qpos_deg(q_meas_deg))
                print("[HOME] aborted by button B. Holding current pose.")
                return False

        obs = robot.get_observation()
        q_meas_deg = extract_qpos_deg_from_obs(obs)
        err = np.abs(q_goal_deg - q_meas_deg)

        if np.all(err <= tol_deg):
            print(f"[HOME] reached (max_err={float(err.max()):.2f} deg).")
            return True

        if (time.perf_counter() - t_start) > timeout_sec:
            print(f"[HOME] timeout after {timeout_sec:.1f}s (max_err={float(err.max()):.2f} deg). Holding.")
            robot.send_action(build_action_from_qpos_deg(q_meas_deg))
            return False

        rate.sleep()

def main():
    # dataset setup
    TASK_NAME = "piper_single_arm_quest3_real"
    NUM_DEMO = 50
    dataset = EpisodeDataset(out_dir=_HERE / "dataset_out")

    # Quest3
    teleop = Quest3Teleop()

    # MuJoCo model for FK/IK internal representation
    model, data, configuration = initialize_model()

    # initial pose
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    configuration.update(data.qpos)

    # EE site
    ee_site = pick_ee_site(model)
    print(f"[INFO] EE site: {ee_site}")

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site)
    if site_id < 0:
        raise RuntimeError(f"EE site not found: {ee_site}")

    # IK tasks
    ee_task = mink.FrameTask(
        frame_name=ee_site,
        frame_type="site",
        position_cost=1.0,
        orientation_cost=0.5,
        lm_damping=1.0,
    )
    posture_task = mink.PostureTask(model=model, cost=POSTURE_COST)
    tasks = [ee_task, posture_task]

    # Mocap target
    mocap_id = model.body("target").mocapid

    # Right controller only
    follow = Controller(use_rotation=True, pos_scale=1.0, R_fix=np.eye(3))

    # Connect hardware
    piper_cfg = PiperRobotConfig(port="can0")
    robot = PiperFollower(piper_cfg)
    robot.connect(calibrate=False)

    # timing
    rate = RateLimiter(frequency=RATE_HZ, warn=False)
    rec_accum = 0.0

    # episode logic
    episode_id = 0
    record_flag = False

    prev_reset = False
    prev_done = False
    last_q_cmd_deg: Optional[np.ndarray] = None

    def hard_reset():
        nonlocal record_flag, follow, last_q_cmd_deg
        dataset.clear_episode_buffer()
        record_flag = False
        follow = Controller(use_rotation=True, pos_scale=1.0, R_fix=follow.R_fix)
        last_q_cmd_deg = None

        # Hold current hardware pose
        obs = robot.get_observation()
        q_meas_deg = extract_qpos_deg_from_obs(obs)
        robot.send_action(build_action_from_qpos_deg(q_meas_deg))
        print("[RESET] episode buffer cleared + hold current pose")

        # Sync MuJoCo to measured q (FK)
        data.qpos[:6] = np.deg2rad(q_meas_deg)  # MuJoCo model uses radians for joints
        mujoco.mj_forward(model, data)
        configuration.update(data.qpos)

        # Reset mocap target to current EE pose (in MuJoCo)
        mink.move_mocap_to_frame(model, data, "target", ee_site, "site")
        mujoco.mj_forward(model, data)

        # Anchor posture to current pose
        posture_task.set_target_from_configuration(configuration)

    try:
        print("[INFO] Connected. Right controller only.")
        print("  - Hold squeeze: move target")
        print("  - Button B (button1): reset episode (hold + target reset)")
        print("  - Button A (button0): save episode (if recording)")
        if key_id != -1:
            print("  - BEFORE START: moving hardware to keyframe 'home' (press B to abort)")

            pre_move_to_keyframe_home(
                model=model,
                key_id=key_id,
                robot=robot,
                teleop=teleop,
                rate=rate,
                tol_deg=HOME_REACH_TOL_DEG,
                timeout_sec=HOME_TIMEOUT_SEC,
            )
            # small settle
            time.sleep(HOME_SETTLE_SEC)

            # After homing attempt, sync Mujoco + anchor from measured pose
            hard_reset()

        else:
            hard_reset()

        while episode_id < NUM_DEMO:
            frame_dt = rate.dt
            ik_dt = frame_dt / float(max(1, MAX_ITERS_PER_CYCLE))

            obs = robot.get_observation()
            q_meas_deg = extract_qpos_deg_from_obs(obs)

            data.qpos[:6] = np.deg2rad(q_meas_deg)
            mujoco.mj_forward(model, data)
            configuration.update(data.qpos)

            frame = teleop.read()

            reset_now = bool(frame.right_state.button1)
            reset = reset_now and (not prev_reset)
            prev_reset = reset_now
            if reset:
                dataset.clear_episode_buffer()
                record_flag = False
                print("[RESET] recording aborted + buffer cleared")
                pre_move_to_keyframe_home(
                    model=model,
                    key_id=key_id,
                    robot=robot,
                    teleop=teleop,
                    rate=rate,
                    tol_deg=HOME_REACH_TOL_DEG,
                    timeout_sec=HOME_TIMEOUT_SEC,
                )
                time.sleep(HOME_SETTLE_SEC)
                hard_reset()
                rate.sleep()
                continue

            done_now = bool(frame.right_state.button0)
            done = done_now and (not prev_done)
            prev_done = done_now
            if done:
                print(f"[DONE] button A pressed. record_flag={record_flag} frames={len(dataset._frames)}")
                if record_flag:
                    dataset.save_episode()
                    episode_id += 1
                    print(f"[DATASET] Episode done. episode_id={episode_id}/{NUM_DEMO}")
                hard_reset()
                continue

            T_ctrl = T_from_pos_quat_xyzw(frame.right_pose.pos, frame.right_pose.quat)
            T_ctrl[:3,:3] = T_ctrl[:3,:3] @ (R_FLIP_RP @ R_SWAP_XZ)

            T_moc_now = T_from_mocap(model, data, mocap_id)

            ok, T_des = follow.update(frame.right_state.squeeze, T_ctrl, T_moc_now)
            if ok and T_des is not None:
                set_mocap_from_T(data, mocap_id, T_des)

            if (not record_flag) and ok:
                record_flag = True
                print("[DATASET] Start recording")

            ee_task.set_target(mink.SE3.from_mocap_name(model, data, "target"))

            target_pos = data.mocap_pos[mocap_id].copy()
            target_quat = data.mocap_quat[mocap_id].copy()

            reached = False
            for _ in range(MAX_ITERS_PER_CYCLE):
                vel = mink.solve_ik(configuration, tasks, ik_dt, SOLVER, damping=DAMPING)
                configuration.integrate_inplace(vel, ik_dt)

                data.qpos[:] = configuration.q
                mujoco.mj_forward(model, data)

                reached = check_reached_single(
                    model, data,
                    site_id,
                    target_pos, target_quat,
                    POS_THRESHOLD, ORI_THRESHOLD,
                )
                if reached:
                    break

            q_cmd_rad = np.array(configuration.q, dtype=np.float64)[:6].copy()
            q_cmd_deg = np.rad2deg(q_cmd_rad)

            q_cmd_deg = clamp_step_deg(q_meas_deg, q_cmd_deg, MAX_DQ_PER_STEP_DEG)
            last_q_cmd_deg = q_cmd_deg.copy()

            robot.send_action(build_action_from_qpos_deg(q_cmd_deg))

            rec_accum += frame_dt
            if rec_accum >= REC_DT:
                rec_accum -= REC_DT

                obs_state = get_obs_state(model, data, site_id)

                tpos = _vec1(data.mocap_pos[mocap_id])[:3]
                tquat = _vec1(data.mocap_quat[mocap_id])[:4]
                target_state = np.concatenate([tpos, tquat], axis=0)

                if record_flag:
                    dataset.add_frame(
                        {
                            "observation.state": obs_state,
                            "observation.target": target_state,
                            "observation.qpos_meas_deg": q_meas_deg.copy(),
                            "action.qpos_cmd_deg": q_cmd_deg.copy(),
                            "info.reached": bool(reached),
                        },
                        task=TASK_NAME,
                    )

            rate.sleep()

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt")
    finally:
        if record_flag and len(dataset._frames) > 0:
            print("[INFO] Saving partial episode before exit.")
            dataset.save_episode()
        robot.disconnect()

if __name__ == "__main__":
    main()