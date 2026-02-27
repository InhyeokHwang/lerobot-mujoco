from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List

import mujoco
import mujoco.viewer
import numpy as np

import mink

from loop_rate_limiters import RateLimiter

from mink_ik.bimanual_mink_ik import (
    pick_two_ee_sites,
    initialize_model,
)

def _vec1(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)

def load_episode_npz(ep_path: Path) -> List[Dict[str, Any]]:
    """episode_XXXX.npz -> frames(list[dict])"""
    data = np.load(ep_path, allow_pickle=True)
    frames_obj = data["frames"]  
    frames = list(frames_obj.tolist())
    if not isinstance(frames, list) or (len(frames) > 0 and not isinstance(frames[0], dict)):
        raise ValueError(f"Invalid frames format in {ep_path}")
    return frames

def set_mocap_targets_from_target_state(model: mujoco.MjModel, data: mujoco.MjData, target_state: np.ndarray):
    target_state = _vec1(target_state) # target_state: [L_pos(3), L_quat_wxyz(4), R_pos(3), R_quat_wxyz(4)] => (14,)
    if target_state.size < 14:
        raise ValueError(f"target_state expected size 14, got {target_state.size}")

    lpos = target_state[0:3]
    lquat = target_state[3:7]
    rpos = target_state[7:10]
    rquat = target_state[10:14]

    mocap_l = model.body("target_left").mocapid
    mocap_r = model.body("target_right").mocapid

    data.mocap_pos[mocap_l] = lpos.reshape(3,)
    data.mocap_quat[mocap_l] = lquat.reshape(4,)
    data.mocap_pos[mocap_r] = rpos.reshape(3,)
    data.mocap_quat[mocap_r] = rquat.reshape(4,)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dataset_out", help="directory containing episode_XXXX.npz")
    parser.add_argument("--episode", type=int, default=0, help="episode index (e.g., 0 -> episode_0000.npz)")
    parser.add_argument("--hz", type=float, default=20.0, help="replay rate (Hz)")
    parser.add_argument("--loop", action="store_true", help="loop episode forever")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    # dataset path
    data_dir = (here / args.data_dir).resolve()
    ep_path = data_dir / f"episode_{args.episode:04d}.npz"
    if not ep_path.exists():
        raise FileNotFoundError(f"Episode file not found: {ep_path}")

    frames = load_episode_npz(ep_path)
    if len(frames) == 0:
        raise RuntimeError(f"Empty episode: {ep_path}")

    print(f"[VIS] Loaded {len(frames)} frames from {ep_path}")

    # MuJoCo model/data
    model, data, configuration = initialize_model()

    # keyframe (initial pose)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # EE sites, mocap target init
    ee_left, ee_right = pick_two_ee_sites(model)
    mink.move_mocap_to_frame(model, data, "target_left", ee_left, "site")
    mink.move_mocap_to_frame(model, data, "target_right", ee_right, "site")
    mujoco.mj_forward(model, data)

    rate = RateLimiter(frequency=float(args.hz), warn=False)

    with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        idx = 0
        while viewer.is_running():
            frame = frames[idx]
            # robot replay
            if "action.qpos" in frame:
                qpos = _vec1(frame["action.qpos"])
                if qpos.size != model.nq:
                    raise ValueError(f"action.qpos size mismatch: got {qpos.size}, expected model.nq={model.nq}")
                data.qpos[:] = qpos
                if data.qvel.size > 0:
                    data.qvel[:] = 0.0
    
            # target(mocap) replay
            if "observation.target" in frame:
                try:
                    set_mocap_targets_from_target_state(model, data, frame["observation.target"])
                except Exception as e:
                    print(f"[VIS][WARN] target update failed at idx={idx}: {type(e).__name__}: {e}")

            mujoco.mj_forward(model, data)

            viewer.sync()
            rate.sleep()

            idx += 1
            if idx >= len(frames):
                if args.loop:
                    idx = 0
                else:
                    print("[VIS] Done.")
                    break

if __name__ == "__main__":
    main()