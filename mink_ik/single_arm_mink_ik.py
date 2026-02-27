from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter
import mink

from quest3.quest3_teleop import Quest3Teleop
from .quest3_utils import Controller, T_from_pos_quat_xyzw, set_mocap_from_T, T_from_mocap

_HERE = Path(__file__).parent
_XML = _HERE.parent / "description" / "agilex_piper" / "scene.xml"

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

# Mocap target
TARGET_RADIUS = 0.03
TARGET_RGBA = [1.0, 0.1, 0.1, 0.9]

def pick_ee_site(model: mujoco.MjModel) -> str:
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper") != -1:
        return "gripper"
    if model.nsite > 0:
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, 0)
    raise RuntimeError("EE site not found: expected at least one site (e.g., 'gripper').")

def _quaternion_error(q_current: np.ndarray, q_target: np.ndarray) -> float:
    return float(min(np.linalg.norm(q_current - q_target), np.linalg.norm(q_current + q_target)))

def site_pose(model: mujoco.MjModel, data: mujoco.MjData, site_id: int) -> Tuple[np.ndarray, np.ndarray]:
    pos = data.site_xpos[site_id].copy()
    quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, data.site_xmat[site_id])
    return pos, quat

def check_reached_single(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    pos_threshold: float,
    ori_threshold: float,
) -> bool:
    meas_pos, meas_quat = site_pose(model, data, site_id)

    err_pos = np.linalg.norm(meas_pos - target_pos)
    err_ori = _quaternion_error(meas_quat, target_quat)

    return (err_pos <= pos_threshold) and (err_ori <= ori_threshold)

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
        _ensure_mocap_target(spec, "target", TARGET_RGBA)
        return spec.compile()
    except Exception as e:
        print(
            f"[WARN] MjSpec injection failed ({type(e).__name__}: {e}). "
            f"Falling back to from_xml_path; assuming 'target' exists in XML."
        )
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



def main():
    model, data, configuration = initialize_model()

    # initial pose
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    configuration.update(data.qpos) # fk

    # EE site
    ee_site = pick_ee_site(model)
    print(f"[INFO] EE site: {ee_site}")
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site)

    # tasks
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

    # ctrl map
    joint2act = build_ctrl_map_for_joints(model)

    # init mocap target to current EE
    mink.move_mocap_to_frame(model, data, "target", ee_site, "site")
    mujoco.mj_forward(model, data)

    rate = RateLimiter(frequency=RATE_HZ, warn=False)

    # Quest3 input
    teleop = Quest3Teleop()

    # right controller only
    follow = Controller(use_rotation=True, pos_scale=1.0, R_fix=np.eye(3))

    def hard_reset() -> None:
        nonlocal follow
        if key_id != -1:
            mujoco.mj_resetDataKeyframe(model, data, key_id)
        else:
            mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)
        mink.move_mocap_to_frame(model, data, "target", ee_site, "site")
        mujoco.mj_forward(model, data)
        follow = Controller(use_rotation=True, pos_scale=1.0, R_fix=follow.R_fix)
        print("[RESET] home + mocap + follower reset")

    with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        prev_reset = False

        while viewer.is_running():
            frame_dt = rate.dt
            frame = teleop.read()

            # reset: right button B
            reset_now = bool(frame.right_state.button1)
            reset = reset_now and (not prev_reset)
            prev_reset = reset_now
            if reset:
                hard_reset()
                viewer.sync()
                rate.sleep()
                continue

            # controller pose -> 4x4
            T_ctrl = T_from_pos_quat_xyzw(frame.right_pose.pos, frame.right_pose.quat)

            # mocap pose -> 4x4
            mocap_id = model.body("target").mocapid
            T_moc_now = T_from_mocap(model, data, mocap_id)

            # squeeze gating: controller delta -> mocap target
            ok, T_des = follow.update(frame.right_state.squeeze, T_ctrl, T_moc_now)
            if ok and T_des is not None:
                set_mocap_from_T(data, mocap_id, T_des)

            # mocap -> task target
            T_wt = mink.SE3.from_mocap_name(model, data, "target")
            ee_task.set_target(T_wt)

            # current target (for convergence check)
            target_pos = data.mocap_pos[mocap_id].copy()
            target_quat = data.mocap_quat[mocap_id].copy()

            # IK
            ik_dt = frame_dt / float(MAX_ITERS_PER_CYCLE)
            reached = False
            for _ in range(MAX_ITERS_PER_CYCLE):
                vel = mink.solve_ik(configuration, tasks, ik_dt, SOLVER, DAMPING)
                configuration.integrate_inplace(vel, ik_dt)

                apply_configuration(model, data, configuration, joint2act=joint2act)
                mujoco.mj_step(model, data)

                reached = check_reached_single(
                    model,
                    data,
                    site_id,
                    target_pos,
                    target_quat,
                    POS_THRESHOLD,
                    ORI_THRESHOLD,
                )
                if reached:
                    break
                
            viewer.sync()
            rate.sleep()


if __name__ == "__main__":
    main()