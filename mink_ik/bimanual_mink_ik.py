from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter
import mink

from quest3.quest3_teleop import Quest3Teleop
from .quest3_utils import Controller, _T_from_pos_quat_xyzw, _set_mocap_from_T, _T_from_mocap, R_Y_PI

_HERE = Path(__file__).parent
_XML = _HERE.parent / "description" / "dual_arm" / "scene.xml"

SOLVER = "daqp"

# IK
POSTURE_COST = 1e-4
MAX_ITERS_PER_CYCLE = 20
DAMPING = 5e-4

# Convergence thresholds
POS_THRESHOLD = 1e-3
ORI_THRESHOLD = 1e-2

# Viewer loop rate
RATE_HZ = 100.0

# Mocap target
TARGET_RADIUS = 0.03
TARGET_RGBA_LEFT = [0.1, 0.9, 0.1, 0.9]
TARGET_RGBA_RIGHT = [0.1, 0.1, 0.9, 0.9]


def pick_two_ee_sites(model: mujoco.MjModel) -> Tuple[str, str]:
    common_pairs = [
        ("gripper_left", "gripper_right"),
    ]
    for a, b in common_pairs:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, a) != -1 and \
           mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, b) != -1:
            return a, b
    raise RuntimeError("EE sites not found: expected gripper_left/right")


def _quaternion_error(q_current: np.ndarray, q_target: np.ndarray) -> float:
    return float(min(np.linalg.norm(q_current - q_target), np.linalg.norm(q_current + q_target)))

# used for IK convergence
def site_pose(model: mujoco.MjModel, data: mujoco.MjData, site_id: int) -> Tuple[np.ndarray, np.ndarray]:
    pos = data.site_xpos[site_id].copy()
    quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, data.site_xmat[site_id])
    return pos, quat


# check if EE is reached to mocap target
def check_reached(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_left_id: int,
    site_right_id: int,
    left_target_pos: np.ndarray,
    left_target_quat: np.ndarray,
    right_target_pos: np.ndarray,
    right_target_quat: np.ndarray,
    pos_threshold: float,
    ori_threshold: float,
) -> bool:
    meas_l_pos, meas_l_quat = site_pose(model, data, site_left_id)
    meas_r_pos, meas_r_quat = site_pose(model, data, site_right_id)

    err_pos_l = np.linalg.norm(meas_l_pos - left_target_pos)
    err_ori_l = _quaternion_error(meas_l_quat, left_target_quat)
    err_pos_r = np.linalg.norm(meas_r_pos - right_target_pos)
    err_ori_r = _quaternion_error(meas_r_quat, right_target_quat)

    return (
        err_pos_l <= pos_threshold
        and err_ori_l <= ori_threshold
        and err_pos_r <= pos_threshold
        and err_ori_r <= ori_threshold
    )

# mocap creation
def _ensure_mocap_target(spec: mujoco.MjSpec, name: str, rgba: List[float]) -> None:
    try:
        body = spec.body(name)
    except Exception:
        body = None

    if body is None:
        body = spec.worldbody.add_body(name=name, mocap=True)

    r = float(TARGET_RADIUS)
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[r, r, r],
        rgba=rgba,
        contype=0,
        conaffinity=0,
    )


def _load_model(xml_path: Path) -> mujoco.MjModel:
    try:
        spec = mujoco.MjSpec.from_file(xml_path.as_posix())
        _ensure_mocap_target(spec, "target_left", TARGET_RGBA_LEFT)
        _ensure_mocap_target(spec, "target_right", TARGET_RGBA_RIGHT)
        return spec.compile()
    except Exception as e:
        print(f"[WARN] MjSpec injection failed ({type(e).__name__}: {e}). "
              f"Falling back to from_xml_path; assuming targets already exist in XML.")
        return mujoco.MjModel.from_xml_path(xml_path.as_posix())


def initialize_model() -> Tuple[mujoco.MjModel, mujoco.MjData, mink.Configuration]:
    model = _load_model(_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    configuration = mink.Configuration(model)
    return model, data, configuration


def _actuator_joint_id(model: mujoco.MjModel, act_id: int) -> Optional[int]:
    try:
        trnid = model.actuator_trnid[act_id]
        j_id = int(trnid[0])
        if 0 <= j_id < model.njnt:
            return j_id
    except Exception:
        pass
    return None

# joint -> actuator (control index) mapping
def build_ctrl_map_for_joints(model: mujoco.MjModel) -> Dict[int, int]:
    m: Dict[int, int] = {}
    if model.nu <= 0:
        return m
    for a in range(model.nu):
        j = _actuator_joint_id(model, a)
        if j is None:
            continue
        if j not in m:
            m[j] = a
    return m

# if actuator exists -> data.ctrl
# else -> data.qpos
def apply_configuration(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    configuration: mink.Configuration,
    joint2act: Dict[int, int],
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

# used for robot initial state
def initialize_mocap_targets_to_sites(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_left_name: str,
    site_right_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    site_left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_left_name)
    site_right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_right_name)
    if site_left_id < 0 or site_right_id < 0:
        raise RuntimeError("EE sites not found (check your site names).")

    mocap_l = model.body("target_left").mocapid
    mocap_r = model.body("target_right").mocapid

    data.mocap_pos[mocap_l] = data.site_xpos[site_left_id].copy()
    data.mocap_pos[mocap_r] = data.site_xpos[site_right_id].copy()

    ql = np.empty(4, dtype=np.float64)
    qr = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(ql, data.site_xmat[site_left_id])
    mujoco.mju_mat2Quat(qr, data.site_xmat[site_right_id])
    data.mocap_quat[mocap_l] = ql
    data.mocap_quat[mocap_r] = qr

def converge_ik(
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    configuration: mink.Configuration,
    tasks: list,
    joint2act: Dict[int, int],
    frame_dt: float,
    solver: str,
    damping: float,
    max_iters: int,
    site_left_id: int,
    site_right_id: int,
    left_target_pos: np.ndarray,
    left_target_quat: np.ndarray,
    right_target_pos: np.ndarray,
    right_target_quat: np.ndarray,
    pos_threshold: float,
    ori_threshold: float,
) -> bool:
    if max_iters <= 0:
        return False

    ik_dt = frame_dt / float(max_iters)
    reached = False

    for _ in range(max_iters):
        vel = mink.solve_ik(configuration, tasks, ik_dt, solver, damping)
        configuration.integrate_inplace(vel, ik_dt)

        apply_configuration(model, data, configuration, joint2act=joint2act)
        mujoco.mj_step(model, data)

        reached = check_reached(
            model,
            data,
            site_left_id,
            site_right_id,
            left_target_pos,
            left_target_quat,
            right_target_pos,
            right_target_quat,
            pos_threshold,
            ori_threshold,
        )
        if reached:
            break

    return reached


def main():
    model, data, configuration = initialize_model()

    # initial pose
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    configuration.update(data.qpos)

    # EE sites
    ee_left, ee_right = pick_two_ee_sites(model)
    print(f"[INFO] EE sites: {ee_left}, {ee_right}")

    site_left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_left)
    site_right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_right)

    # tasks
    left_task = mink.FrameTask(
        frame_name=ee_left, frame_type="site",
        position_cost=1.0, orientation_cost=0.5,
        lm_damping=1.0,
    )
    right_task = mink.FrameTask(
        frame_name=ee_right, frame_type="site",
        position_cost=1.0, orientation_cost=0.5,
        lm_damping=1.0,
    )
    posture_task = mink.PostureTask(model=model, cost=POSTURE_COST)
    tasks = [left_task, right_task, posture_task]
    posture_task.set_target_from_configuration(configuration)

    # ctrl map
    joint2act = build_ctrl_map_for_joints(model)

    # init mocap targets to current EE
    initialize_mocap_targets_to_sites(model, data, ee_left, ee_right)
    mujoco.mj_forward(model, data)

    rate = RateLimiter(frequency=RATE_HZ, warn=False)

    # Quest3 input
    teleop = Quest3Teleop()

    follow_left  = Controller(use_rotation=True, pos_scale=1.0, R_fix=R_Y_PI)
    follow_right = Controller(use_rotation=True, pos_scale=1.0, R_fix=np.eye(3))

    def hard_reset() -> None:
        nonlocal follow_left, follow_right
        if key_id != -1:
            mujoco.mj_resetDataKeyframe(model, data, key_id)
        else:
            mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)
        initialize_mocap_targets_to_sites(model, data, ee_left, ee_right)
        mujoco.mj_forward(model, data)
        follow_left  = Controller(use_rotation=True, pos_scale=1.0, R_fix=R_Y_PI)
        follow_right = Controller(use_rotation=True, pos_scale=1.0, R_fix=np.eye(3))
        print("[RESET] home + mocap + followers reset")


    with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        prev_reset = False

        while viewer.is_running():
            frame_dt = rate.dt
            frame = teleop.read()

            # reset right button B
            reset_now = bool(frame.right_state.button1)
            reset = reset_now and (not prev_reset)
            prev_reset = reset_now
            if reset:
                hard_reset()
                viewer.sync()
                rate.sleep()
                continue

            # controller pose -> 4x4
            T_ctrl_L = _T_from_pos_quat_xyzw(frame.left_pose.pos, frame.left_pose.quat)
            T_ctrl_R = _T_from_pos_quat_xyzw(frame.right_pose.pos, frame.right_pose.quat)

            # mocap pose pose -> 4x4
            mocap_l = model.body("target_left").mocapid
            mocap_r = model.body("target_right").mocapid
            T_moc_L_now = _T_from_mocap(model, data, mocap_l)
            T_moc_R_now = _T_from_mocap(model, data, mocap_r)

            okL, T_L_des = follow_left.update(frame.left_state.squeeze, T_ctrl_L, T_moc_L_now)
            okR, T_R_des = follow_right.update(frame.right_state.squeeze, T_ctrl_R, T_moc_R_now)

            if okL and T_L_des is not None:
                _set_mocap_from_T(data, mocap_l, T_L_des)

            if okR and T_R_des is not None:
                _set_mocap_from_T(data, mocap_r, T_R_des)

            # mocap -> task target
            T_wt_left = mink.SE3.from_mocap_name(model, data, "target_left")
            T_wt_right = mink.SE3.from_mocap_name(model, data, "target_right")
            left_task.set_target(T_wt_left)
            right_task.set_target(T_wt_right)

            mocap_l = model.body("target_left").mocapid
            mocap_r = model.body("target_right").mocapid
            left_target_pos = data.mocap_pos[mocap_l].copy()
            right_target_pos = data.mocap_pos[mocap_r].copy()
            left_target_quat = data.mocap_quat[mocap_l].copy()
            right_target_quat = data.mocap_quat[mocap_r].copy()

            #IK
            converge_ik(
                model=model,
                data=data,
                configuration=configuration,
                tasks=tasks,
                joint2act=joint2act,
                frame_dt=frame_dt,
                solver=SOLVER,
                damping=DAMPING,
                max_iters=MAX_ITERS_PER_CYCLE,
                site_left_id=site_left_id,
                site_right_id=site_right_id,
                left_target_pos=left_target_pos,
                left_target_quat=left_target_quat,
                right_target_pos=right_target_pos,
                right_target_quat=right_target_quat,
                pos_threshold=POS_THRESHOLD,
                ori_threshold=ORI_THRESHOLD,
            )

            # render/sleep exactly once per frame
            viewer.sync()
            rate.sleep()


if __name__ == "__main__":
    main()