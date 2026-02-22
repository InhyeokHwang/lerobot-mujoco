from quest3_teleop import Quest3Teleop
import time

teleop = Quest3Teleop()

def print_pose(name, pose):
    p = pose.pos
    q = pose.quat
    print(f"[{name} POSE]")
    print(f"  pos : x={p[0]: .3f}, y={p[1]: .3f}, z={p[2]: .3f}")
    print(f"  quat: x={q[0]: .3f}, y={q[1]: .3f}, z={q[2]: .3f}, w={q[3]: .3f}")

def print_state(name, st):
    print(f"[{name} STATE]")
    print(f"  trigger        : {st.trigger:.2f}")
    print(f"  squeeze        : {st.squeeze:.2f}")
    print(f"  touchpad       : {st.touchpad}")
    print(f"  thumbstick     : {st.thumbstick}")
    print(f"  button0        : {st.button0}")
    print(f"  button1        : {st.button1}")
    print(f"  trigger_value  : {st.trigger_value:.2f}")
    print(f"  squeeze_value  : {st.squeeze_value:.2f}")
    print(f"  touchpad_xy    : {st.touchpad_xy}")
    print(f"  thumbstick_xy  : {st.thumbstick_xy}")

print("=== Quest3 Teleop Debug Start ===")

PRINT_DT = 0.3  # seconds

last_print = 0.0

while True:
    frame = teleop.read()
    now = time.time()

    if now - last_print < PRINT_DT:
        time.sleep(0.01)
        continue

    last_print = now

    print("\n" + "=" * 60)

    print_pose("RIGHT", frame.right_pose)
    print_state("RIGHT", frame.right_state)

    if frame.right_state.squeeze > 0.5:
        print("  >>> RIGHT CLUTCH (SQUEEZE ACTIVE)")

    print_pose("LEFT", frame.left_pose)
    print_state("LEFT", frame.left_state)

    if frame.left_state.squeeze > 0.5:
        print("  >>> LEFT CLUTCH (SQUEEZE ACTIVE)")