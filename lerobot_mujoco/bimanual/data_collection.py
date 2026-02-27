from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter
import mink

from quest3.quest3_teleop import Quest3Teleop
from mink_ik.bimanual_mink_ik import (
    pick_two_ee_sites,
    site_pose,
    check_reached,
    initialize_model,
    build_ctrl_map_for_joints,
    apply_configuration,
)

from mink_ik.quest3_utils import (
    Controller,
    T_from_pos_quat_xyzw,
    set_mocap_from_T,
    T_from_mocap,
    CTRL2EE_LEFT, CTRL2EE_RIGHT
)

_HERE = Path(__file__).parent

SOLVER = "daqp"

# IK
POSTURE_COST = 1e-3
MAX_ITERS_PER_CYCLE = 20
DAMPING = 5e-4

# Convergence thresholds
POS_THRESHOLD = 1e-3
ORI_THRESHOLD = 1e-2

# Viewer loop rate
RATE_HZ = 100.0

# Data collection rate 
REC_HZ = 20.0
REC_DT = 1.0 / REC_HZ

class EpisodeDataset:
    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir) # data directory
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


def get_obs_state(model: mujoco.MjModel, data: mujoco.MjData, site_left_id: int, site_right_id: int) -> np.ndarray:
    lpos, lquat = site_pose(model, data, site_left_id) # 3 4
    rpos, rquat = site_pose(model, data, site_right_id) # 3 4 
    return np.concatenate([lpos, lquat, rpos, rquat], axis=0)

def _vec1(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)

def main():
    # dataset setup
    TASK_NAME = "dual_arm_teleop"
    NUM_DEMO = 50
    dataset = EpisodeDataset(out_dir=_HERE / "dataset_out")

    model, data, configuration = initialize_model()

    # initial pose
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    configuration.update(data.qpos)

    ee_left, ee_right = pick_two_ee_sites(model)
    print(f"[INFO] EE sites: {ee_left}, {ee_right}")

    site_left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_left)
    site_right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_right)

    # tasks
    left_task = mink.FrameTask(
        frame_name=ee_left, frame_type="site",
        position_cost=1.0, orientation_cost=1.0,
        lm_damping=1.0,
    )
    right_task = mink.FrameTask(
        frame_name=ee_right, frame_type="site",
        position_cost=1.0, orientation_cost=1.0,
        lm_damping=1.0,
    )
    posture_task = mink.PostureTask(model=model, cost=POSTURE_COST)
    tasks = [left_task, right_task, posture_task]
    posture_task.set_target_from_configuration(configuration)

    joint2act = build_ctrl_map_for_joints(model)

    # init mocap targets to current EE
    mink.move_mocap_to_frame(model, data, "target_left", ee_left, "site")
    mink.move_mocap_to_frame(model, data, "target_right", ee_right, "site")
    mujoco.mj_forward(model, data)

    # timing
    rate = RateLimiter(frequency=RATE_HZ, warn=False)
    rec_accum = 0.0

    # Quest3 input
    teleop = Quest3Teleop()

    follow_left  = Controller(use_rotation=True, pos_scale=1.0, R_fix=np.eye(3))
    follow_right = Controller(use_rotation=True, pos_scale=1.0, R_fix=np.eye(3))

    # episode logic
    episode_id = 0
    record_flag = False

    # reset
    def hard_reset():
        nonlocal record_flag, follow_left, follow_right
        if key_id != -1:
            mujoco.mj_resetDataKeyframe(model, data, key_id)
        else:
            mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        configuration.update(data.qpos)
        mink.move_mocap_to_frame(model, data, "target_left", ee_left, "site")
        mink.move_mocap_to_frame(model, data, "target_right", ee_right, "site")
        mujoco.mj_forward(model, data)
        dataset.clear_episode_buffer()
        record_flag = False
        follow_left  = Controller(use_rotation=True, pos_scale=1.0, R_fix=np.eye(3))
        follow_right = Controller(use_rotation=True, pos_scale=1.0, R_fix=np.eye(3))
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
            T_ctrl_L = T_from_pos_quat_xyzw(frame.left_pose.pos, frame.left_pose.quat)
            T_ctrl_R = T_from_pos_quat_xyzw(frame.right_pose.pos, frame.right_pose.quat)

            # left controller axis compensation
            T_ctrl_L = T_ctrl_L @ CTRL2EE_LEFT
            T_ctrl_R = T_ctrl_R @ CTRL2EE_RIGHT

            # mocap
            mocap_l = model.body("target_left").mocapid
            mocap_r = model.body("target_right").mocapid

            # mocap 4x4 homogeneous transform
            T_moc_L_now = T_from_mocap(model, data, mocap_l)
            T_moc_R_now = T_from_mocap(model, data, mocap_r)

            # controller delta -> mocap
            left_ok, T_L_des = follow_left.update(frame.left_state.squeeze, T_ctrl_L, T_moc_L_now)
            right_ok, T_R_des = follow_right.update(frame.right_state.squeeze, T_ctrl_R, T_moc_R_now)

            if left_ok and T_L_des is not None:
                set_mocap_from_T(data, mocap_l, T_L_des)
            if right_ok and T_R_des is not None:
                set_mocap_from_T(data, mocap_r, T_R_des)

            # record start condition: mocap being updated
            if (not record_flag) and (left_ok or right_ok):
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
            left_task.set_target(mink.SE3.from_mocap_name(model, data, "target_left"))
            right_task.set_target(mink.SE3.from_mocap_name(model, data, "target_right"))

            # copy current targets (to check reached)
            left_target_pos = data.mocap_pos[mocap_l].copy()
            right_target_pos = data.mocap_pos[mocap_r].copy()
            left_target_quat = data.mocap_quat[mocap_l].copy()
            right_target_quat = data.mocap_quat[mocap_r].copy()

            # IK sub-iterations
            reached = False
            for _ in range(MAX_ITERS_PER_CYCLE):
                vel = mink.solve_ik(configuration, tasks, ik_dt, SOLVER, DAMPING)
                configuration.integrate_inplace(vel, ik_dt)

                apply_configuration(model, data, configuration, joint2act=joint2act)
                mujoco.mj_step(model, data)

                reached = check_reached(
                    model, data,
                    site_left_id, site_right_id,
                    left_target_pos, left_target_quat,
                    right_target_pos, right_target_quat,
                    POS_THRESHOLD, ORI_THRESHOLD,
                )
                if reached:
                    break

            # --------------- data collection ---------------
            rec_accum += frame_dt
            if rec_accum >= REC_DT:
                rec_accum -= REC_DT

                # observation/state
                obs_state = get_obs_state(model, data, site_left_id, site_right_id)

                # action label: current qpos
                action_qpos = data.qpos.copy()
                
                # mocap pose
                left_target_pos  = _vec1(data.mocap_pos[mocap_l])[:3]
                right_target_pos = _vec1(data.mocap_pos[mocap_r])[:3]
                left_target_quat  = _vec1(data.mocap_quat[mocap_l])[:4]
                right_target_quat = _vec1(data.mocap_quat[mocap_r])[:4]

                target_state = np.concatenate(
                    [left_target_pos, left_target_quat, right_target_pos, right_target_quat],
                    axis=0
                )

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