from __future__ import annotations

import argparse
from pathlib import Path

import torch

# ✅ 너 환경 코드 기준 import (lerobot.datasets.*)
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.datasets.factory import resolve_delta_timestamps

from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, required=True, help="Dataset root folder (contains meta/, data/)")
    p.add_argument("--repo_id", type=str, default="dual_arm_teleop_npz", help="Dataset repo_id used in create()")
    p.add_argument("--out", type=str, default="./ckpt/act_dual_arm", help="Output checkpoint dir")
    p.add_argument("--chunk_size", type=int, default=10)
    p.add_argument("--n_action_steps", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--log_freq", type=int, default=100)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ds_root = Path(args.root).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load dataset metadata (for features + stats) ----
    meta = LeRobotDatasetMetadata(args.repo_id, root=ds_root)
    policy_features = dataset_to_policy_features(meta.features)

    # output = ACTION only
    output_features = {k: ft for k, ft in policy_features.items() if ft.type is FeatureType.ACTION}

    # input = everything else (state-only dataset면 observation.environment_state만 남음)
    input_features = {k: ft for k, ft in policy_features.items() if k not in output_features}

    # (안전장치) 혹시 다른 observation 키가 섞였을 때만 남길 것 지정하고 싶으면 여기서 필터링 가능
    # input_features = {k: ft for k, ft in input_features.items() if k == "observation.environment_state"}

    if len(output_features) == 0:
        raise RuntimeError(f"No ACTION feature found. Available: {list(policy_features.keys())}")
    if len(input_features) == 0:
        raise RuntimeError(f"No input features found. Available: {list(policy_features.keys())}")

    # ---- ACT config ----
    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
    )

    # ---- delta timestamps for action chunking ----
    delta_timestamps = resolve_delta_timestamps(cfg, meta)

    # ---- policy ----
    policy = ACTPolicy(cfg, dataset_stats=meta.stats)
    policy.train()
    policy.to(device)

    # ---- dataset + dataloader ----
    dataset = LeRobotDataset(
        args.repo_id,
        root=ds_root,
        delta_timestamps=delta_timestamps,
        # images 없으니 image_transforms 불필요
    )

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # ---- train loop ----
    step = 0
    done = False
    while not done:
        for batch in loader:
            inp = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            loss, _ = policy.forward(inp)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % args.log_freq == 0:
                print(f"step={step} loss={loss.item():.6f}")

            step += 1
            if step >= args.steps:
                done = True
                break

    # ---- save ----
    policy.save_pretrained(str(out_dir))
    print(f"[OK] saved policy -> {out_dir}")


if __name__ == "__main__":
    main()