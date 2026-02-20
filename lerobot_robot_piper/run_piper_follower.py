from __future__ import annotations

import time
import argparse

from .piper_config import PiperRobotConfig
from .piper_follower import PiperFollower

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="can0")
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--no-torque", action="store_true")
    args = ap.parse_args()

    cfg = PiperRobotConfig(port=args.port)
    robot = PiperFollower(cfg)

    robot.connect(calibrate=False)

    if args.no_torque:
        robot.bus.disable_torque()

    dt = 1.0 / args.hz

    # 현재 자세 유지(hold) 테스트
    obs = robot.get_observation()
    hold = {f"joint_{i}.pos": obs[f"joint_{i}.pos"] for i in range(1, 7)}

    print("Connected. Holding current pose. Ctrl+C to stop.")
    try:
        while True:
            t0 = time.perf_counter()

            obs = robot.get_observation()
            # 필요하면 디버그 출력
            # print({k: round(obs[k], 2) for k in hold.keys()})

            robot.send_action(hold)

            elapsed = time.perf_counter() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        pass
    finally:
        robot.disconnect()

if __name__ == "__main__":
    main()