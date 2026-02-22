# ------------------------------------------------------------------------------
# This script loads a MuJoCo robot model from an MJCF (or URDF-converted XML) file 
# and launches the built-in MuJoCo viewer to visualize the model in its initial state.
# If you observe the robot falling in simulation it's natural since this code is specifically build to just load the robot.
#
# Ensure that the XML path is correctly specified before running.
# ------------------------------------------------------------------------------


import mujoco
from mujoco.viewer import launch
from pathlib import Path

# Path to your MuJoCo XML model MJCF
# resolved relative to the repository root (parent of this scripts/ folder)
MJCF_FILE_PATH = str(
    (Path(__file__).resolve().parent.parent / "description" / "dual_arm" / "scene.xml")
)

# Load the MuJoCo model from the specified XML path
model = mujoco.MjModel.from_xml_path(MJCF_FILE_PATH)

# Create a corresponding data object to hold simulation state
data = mujoco.MjData(model)

# Perform one simulation step (optional, just initializes dynamics)
mujoco.mj_step(model, data)

# Launch the interactive MuJoCo viewer with the model and data
viewer = launch(model, data)

# Keep the viewer open and continuously render the model in its current state
# No additional simulation steps are performed in this loop
while viewer and viewer.is_running():
    viewer.render()