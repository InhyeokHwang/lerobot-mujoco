from __future__ import annotations

from pathlib import Path
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter
import mink

from quest3.quest3_teleop import Quest3Teleop
from .quest3_utils import (
    Controller,
    _T_from_pos_quat_xyzw,
    _set_mocap_from_T,
    _T_from_mocap,
)

# _XML = Path(__file__).parent.parent / "description" / "franka_emika_panda" / "mjx_scene.xml"
_XML = Path(__file__).parent.parent / "description" / "agilex_piper" / "scene.xml"

SOLVER = "daqp"

# Convergence (how strict "reached" is)
POS_THRESHOLD = 1e-4
ORI_THRESHOLD = 1e-4

# IK iterations per viewer frame
MAX_ITERS = 20  
DAMPING = 1e-3

# Viewer loop rate
RATE_HZ = 200.0

# target(red sphere) config
TARGET_RADIUS = 0.05
TARGET_RGBA = [1.0, 0.1, 0.1, 0.9]


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
    # solve_ik에는 "list[task]"를 주면 됨
    tasks = [end_effector_task, posture_task]

    for _ in range(max_iters):
        vel = mink.solve_ik(configuration, tasks, dt, solver, damping=DAMPING)
        configuration.integrate_inplace(vel, dt)

        err = end_effector_task.compute_error(configuration)
        if np.linalg.norm(err[:3]) <= pos_threshold and np.linalg.norm(err[3:]) <= ori_threshold:
            return True
    return False


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
    # 모델마다 ee site 이름이 다름: 우선 "gripper" 있으면 그걸 사용
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper") != -1:
        return "gripper"
    # 아니면 첫 번째 site
    if model.nsite > 0:
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, 0)
    raise RuntimeError("No site exists in this model. Cannot run site-based FrameTask.")


def _actuator_joint_id(model: mujoco.MjModel, act_id: int) -> Optional[int]:
    try:
        trnid = model.actuator_trnid[act_id]
        j_id = int(trnid[0])
        if 0 <= j_id < model.njnt:
            return j_id
    except Exception:
        pass
    return None


def build_ctrl_map_for_joints(model: mujoco.MjModel) -> dict[int, int]:
    """
    joint id -> actuator id (ctrl index) 맵.
    actuator 기반 모델이면 qpos를 직접 바꾸지 말고 ctrl을 쓰는 게 안전.
    """
    m: dict[int, int] = {}
    if model.nu <= 0:
        return m
    for a in range(model.nu):
        j = _actuator_joint_id(model, a)
        if j is None:
            continue
        if j not in m:
            m[j] = a
    return m


def apply_configuration(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    configuration: mink.Configuration,
    joint2act: dict[int, int],
) -> None:
    if model.nu <= 0 or not joint2act:
        data.qpos[:] = configuration.q
        return

    for j_id, a_id in joint2act.items():
        qadr = int(model.jnt_qposadr[j_id])
        jtype = int(model.jnt_type[j_id])
        if jtype in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL):
            continue
        data.ctrl[a_id] = float(configuration.q[qadr])


def main():
    model = load_model_with_big_target(_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    configuration = mink.Configuration(model)

    ee_site = pick_site_name(model)
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

    # ctrl map (actuator 모델이면 ctrl로 q를 넣어야 함)
    joint2act = build_ctrl_map_for_joints(model)

    # Quest3 teleop
    teleop = Quest3Teleop()

    # right controller
    follow = Controller(use_rotation=True, pos_scale=1.0, R_fix=np.eye(3))

    with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # reset -> posture 기준
        reset_to_home_if_exists(model, data, "home")
        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)
        mujoco.mj_forward(model, data)

        # target(=mocap)을 EE 위치로 초기화
        mink.move_mocap_to_frame(model, data, "target", ee_site, "site")
        mujoco.mj_forward(model, data)

        rate = RateLimiter(frequency=RATE_HZ, warn=False)

        prev_reset = False

        while viewer.is_running():
            dt = rate.dt

            frame = teleop.read()

            # right controller pose -> 4x4
            T_ctrl = _T_from_pos_quat_xyzw(frame.right_pose.pos, frame.right_pose.quat)

            # current mocap pose -> 4x4
            mocap_id = model.body("target").mocapid
            T_moc_now = _T_from_mocap(model, data, mocap_id)

            # reset: button1 rising edge (원하면 left/right 바꿔도 됨)
            reset_now = bool(frame.right_state.button1)
            reset = reset_now and (not prev_reset)
            prev_reset = reset_now
            if reset:
                reset_to_home_if_exists(model, data, "home")
                configuration.update(data.qpos)
                posture_task.set_target_from_configuration(configuration)
                mujoco.mj_forward(model, data)
                mink.move_mocap_to_frame(model, data, "target", ee_site, "site")
                mujoco.mj_forward(model, data)
                follow = Controller(use_rotation=True, pos_scale=1.0, R_fix=follow.R_fix)
                print("[RESET] home + target reset")
                viewer.sync()
                rate.sleep()
                continue

            # while squeeze pressed 
            ok, T_des = follow.update(frame.right_state.squeeze, T_ctrl, T_moc_now)
            if ok and T_des is not None:
                _set_mocap_from_T(data, mocap_id, T_des)

            # mocap -> task target
            T_wt = mink.SE3.from_mocap_name(model, data, "target")
            end_effector_task.set_target(T_wt)

            # IK
            converge_ik(
                configuration,
                end_effector_task=end_effector_task,
                posture_task=posture_task,
                dt=dt / max(1, MAX_ITERS),  # substep 느낌
                solver=SOLVER,
                pos_threshold=POS_THRESHOLD,
                ori_threshold=ORI_THRESHOLD,
                max_iters=MAX_ITERS,
            )

            # apply q to sim
            apply_configuration(model, data, configuration, joint2act)

            mujoco.mj_step(model, data)

            viewer.sync()
            rate.sleep()


if __name__ == "__main__":
    main()