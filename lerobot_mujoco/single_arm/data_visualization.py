from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

from mink_ik.single_arm_mink_ik import(
    pick_ee_site,
    initialize_model,
    initialize_mocap_target_to_site
)

def _vec1(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)

def load_episode_npz(ep_path: Path) -> List[Dict[str, Any]]:
    """episode_XXXX.npz -> frames(list[dict])"""
    data = np.load(ep_path, allow_pickle=True)
    frames_obj = data["frames"]  # dtype=object array
    frames = list(frames_obj.tolist())
    if not isinstance(frames, list) or (len(frames) > 0 and not isinstance(frames[0], dict)):
        raise ValueError(f"Invalid frames format in {ep_path}")
    return frames

def set_mocap_target_from_target_state(model: mujoco.MjModel, data: mujoco.MjData, target_state: np.ndarray):
    target_state = _vec1(target_state) # target_state (single arm): [pos(3), quat_wxyz(4)] => (7,)
    if target_state.size < 7:
        raise ValueError(f"target_state expected size 7, got {target_state.size}")

    pos = target_state[0:3]
    quat = target_state[3:7]

    mocap_id = model.body("target").mocapid

    data.mocap_pos[mocap_id] = pos.reshape(3,)
    data.mocap_quat[mocap_id] = quat.reshape(4,)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dataset_out_piper_mujoco", help="dir containing episode_XXXX.npz")
    parser.add_argument("--episode", type=int, default=0, help="episode index (0 -> episode_0000.npz)")
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
    ee_left, ee_right = pick_ee_site(model)
    initialize_mocap_target_to_site(model, data, ee_left, ee_right)
    mujoco.mj_forward(model, data)

    rate = RateLimiter(frequency=float(args.hz), warn=False)

    with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        idx = 0
        while viewer.is_running():
            fr = frames[idx]
            # robot replay
            if "action.qpos" in fr:
                qpos = _vec1(fr["action.qpos"])
                if qpos.size != model.nq:
                    raise ValueError(
                        f"action.qpos size mismatch at idx={idx}: got {qpos.size}, expected model.nq={model.nq}"
                    )
                data.qpos[:] = qpos
                if data.qvel.size > 0:
                    data.qvel[:] = 0.0

            # target(mocap) replay
            if "observation.target" in fr:
                try:
                    set_mocap_target_from_target_state(model, data, fr["observation.target"])
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