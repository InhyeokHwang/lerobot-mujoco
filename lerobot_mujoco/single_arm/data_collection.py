from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter
import mink

from quest3.quest3_teleop import Quest3Teleop

from mink_ik.single_arm_mink_ik import (
    pick_ee_site,                     
    site_pose,                       
    check_reached_single,             
    initialize_model,                 
    build_ctrl_map_for_joints,       
    apply_configuration,             
)

from mink_ik.quest3_utils import (
    Controller,
    T_from_pos_quat_xyzw,
    set_mocap_from_T,
    T_from_mocap,
)

_HERE = Path(__file__).parent

SOLVER = "daqp"

# IK
POSTURE_COST = 1e-3
MAX_ITERS_PER_CYCLE = 20
DAMPING = 1e-3

# Convergence thresholds
POS_THRESHOLD = 1e-4
ORI_THRESHOLD = 1e-4

# Viewer loop rate
RATE_HZ = 100.0

# Data collection rate
REC_HZ = 20.0
REC_DT = 1.0 / REC_HZ

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
        frame = dict(frame)
        frame["task"] = task
        self._frames.append(frame)

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
    pos, quat = site_pose(model, data, site_id) # 3 4
    return np.concatenate([pos, quat], axis=0)

def _vec1(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)

def main():
    # dataset setup 
    TASK_NAME = "piper_single_arm_teleop"
    NUM_DEMO = 50
    dataset = EpisodeDataset(out_dir=_HERE / "dataset_out")

    # Quest3 input
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

    ee_site = pick_ee_site(model)
    print(f"[INFO] EE site: {ee_site}")

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site)
    if site_id < 0:
        raise RuntimeError(f"EE site not found in model: {ee_site}")

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
    posture_task.set_target_from_configuration(configuration)

    joint2act = build_ctrl_map_for_joints(model)

    # init mocap target to current EE
    mink.move_mocap_to_frame(model, data, "target", ee_site, "site")
    mujoco.mj_forward(model, data)

    # timing
    rate = RateLimiter(frequency=RATE_HZ, warn=False)
    rec_accum = 0.0

    # right controller only
    follow = Controller(use_rotation=True, pos_scale=1.0, R_fix=np.eye(3))

    # episode logic
    episode_id = 0
    record_flag = False

    def hard_reset():
        nonlocal record_flag, follow
        if key_id != -1:
            mujoco.mj_resetDataKeyframe(model, data, key_id)
        else:
            mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        configuration.update(data.qpos)

        posture_task.set_target_from_configuration(configuration)
        mink.move_mocap_to_frame(model, data, "target", ee_site, "site")
        mujoco.mj_forward(model, data)

        dataset.clear_episode_buffer()
        record_flag = False
        follow = Controller(use_rotation=True, pos_scale=1.0, R_fix=np.eye(3))
        print("[RESET] env + episode buffer cleared")

    prev_reset = False
    prev_done = False

    # loop
    with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        while viewer.is_running() and episode_id < NUM_DEMO:
            frame_dt = rate.dt # 0.01s
            ik_dt = frame_dt / MAX_ITERS_PER_CYCLE # 0.0005s

            # controller input
            frame = teleop.read()

            # controller 4x4 homogeneous transform
            T_ctrl = T_from_pos_quat_xyzw(frame.right_pose.pos, frame.right_pose.quat)
            T_ctrl[:3,:3] = T_ctrl[:3,:3] @ (R_FLIP_RP @ R_SWAP_XZ)
            
            # mocap
            mocap_id = model.body("target").mocapid

            # mocap 4x4 homogeneous transform
            T_moc_now = T_from_mocap(model, data, mocap_id)

            # controller delta
            ok, T_des = follow.update(frame.right_state.squeeze, T_ctrl, T_moc_now)
            if ok and T_des is not None:
                set_mocap_from_T(data, mocap_id, T_des)

            # record start condition: mocap being updated
            if (not record_flag) and ok:
                record_flag = True
                print("[DATASET] Start recording")

            # reset: right controller button B
            reset_now = bool(frame.right_state.button1)
            reset = reset_now and (not prev_reset)
            prev_reset = reset_now
            if reset:
                hard_reset()
                viewer.sync()
                rate.sleep()
                continue

            # done: right controller button A
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

            # set mocap to IK task target
            ee_task.set_target(mink.SE3.from_mocap_name(model, data, "target"))

            # copy current targets (to check reached)
            target_pos = data.mocap_pos[mocap_id].copy()
            target_quat = data.mocap_quat[mocap_id].copy()

            # IK sub-iterations
            reached = False
            for _ in range(MAX_ITERS_PER_CYCLE):
                vel = mink.solve_ik(configuration, tasks, ik_dt, SOLVER, DAMPING)
                configuration.integrate_inplace(vel, ik_dt)

                apply_configuration(model, data, configuration, joint2act=joint2act)
                mujoco.mj_step(model, data)

                reached = check_reached_single(
                    model, data,
                    site_id,
                    target_pos, target_quat,
                    POS_THRESHOLD, ORI_THRESHOLD,
                )
                if reached:
                    break

            # --------------- data collection ---------------
            rec_accum += frame_dt
            if rec_accum >= REC_DT:
                rec_accum -= REC_DT

                # observation
                obs_state = get_obs_state(model, data, site_id)  # (7,)

                # action label: current qpos
                action_qpos = data.qpos.copy()

                # target (mocap) pose
                tpos = _vec1(data.mocap_pos[mocap_id])[:3]
                tquat = _vec1(data.mocap_quat[mocap_id])[:4]
                target_state = np.concatenate([tpos, tquat], axis=0)  # (7,)

                if record_flag:
                    dataset.add_frame(
                        {
                            "observation.state": obs_state,
                            "observation.target": target_state,
                            "action.qpos": action_qpos,
                        },
                        task=TASK_NAME,
                    )
            # ----------------------------------------------------

            viewer.sync()
            rate.sleep()


if __name__ == "__main__":
    main()