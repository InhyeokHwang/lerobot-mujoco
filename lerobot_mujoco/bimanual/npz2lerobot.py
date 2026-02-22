from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

HERE = Path(__file__).resolve().parent
NPZ_DIR = (HERE / "dataset_out").resolve()

OUT_PARENT = (HERE / "lerobot_data").resolve()
REPO_ID = "dual_arm_teleop_npz"
FPS = 20

TASK_NAME = "dual arm teleop (npz)"


def _as_frame_dict(x) -> Dict[str, Any]:
    if hasattr(x, "item"):
        x = x.item()
    if not isinstance(x, dict):
        raise TypeError(f"Frame must be dict, got {type(x)}")
    return x


def load_npz_episode(ep_path: Path) -> List[Dict[str, Any]]:
    d = np.load(ep_path, allow_pickle=True)
    frames = d["frames"]
    return [_as_frame_dict(frames[i]) for i in range(len(frames))]


def main():
    npz_files = sorted(NPZ_DIR.glob("episode_*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No episodes found in: {NPZ_DIR}")

    first_frames = load_npz_episode(npz_files[0])
    if not first_frames:
        raise RuntimeError(f"First episode is empty: {npz_files[0].name}")

    a0 = np.asarray(first_frames[0]["action.qpos"], dtype=np.float32).reshape(-1)
    action_dim = int(a0.shape[0])

    # ✅ task는 features에 넣지 않는다 (이 구현에서 task는 "special case")
    features = {
        "observation.environment_state": {
            "dtype": "float32",
            "shape": (14,),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": ["action"],
        },
    }

    run = datetime.now().strftime("%Y%m%d_%H%M%S")
    ds_root = (OUT_PARENT / run).resolve()

    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        root=ds_root,
        fps=FPS,
        features=features,
        use_videos=False,
        # robot_type="piper",  # 필요하면 넣어도 됨
    )

    for ep_i, ep_path in enumerate(npz_files):
        frames = load_npz_episode(ep_path)
        if not frames:
            print(f"[NPZ2LERO] skip empty episode: {ep_path.name}")
            continue

        for f in frames:
            obs = np.asarray(f["observation.state"], dtype=np.float32).reshape(14,)       # ✅ (14,)
            act = np.asarray(f["action.qpos"], dtype=np.float32).reshape(action_dim,)     # ✅ (action_dim,)

            dataset.add_frame({
                "observation.environment_state": obs,
                "action": act,
                "task": TASK_NAME,   # ✅ 핵심: frame 안에 task 문자열
            })

        dataset.save_episode()
        print(f"[NPZ2LERO] saved ep={ep_i} from {ep_path.name} frames={len(frames)}")

    dataset.finalize()
    print(f"[NPZ2LERO] DONE -> {ds_root / REPO_ID}")


if __name__ == "__main__":
    main()