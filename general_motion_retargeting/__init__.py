# Lazy imports — core retarget deps (mink, scipy, rich, etc.) are optional.
# They are only installed when the user does:  pip install -e ".[retarget]"
# This allows `import general_motion_retargeting` to succeed in the training-only
# environment without pulling in retarget-specific packages.

try:
    from rich import print as _rich_print  # noqa: F811
except ImportError:
    _rich_print = print  # fallback to builtin print

try:
    from .params import (
        IK_CONFIG_ROOT,
        ASSET_ROOT,
        ROBOT_XML_DICT,
        IK_CONFIG_DICT,
        ROBOT_BASE_DICT,
        VIEWER_CAM_DISTANCE_DICT,
    )
    from .motion_retarget import GeneralMotionRetargeting
    from .robot_motion_viewer import RobotMotionViewer, draw_frame
    from .data_loader import load_robot_motion
    from .kinematics_model import KinematicsModel
    from .neck_retarget import human_head_to_robot_neck
except ImportError:
    pass  # retarget dependencies not installed — training-only mode

try:
    from .xrobot_utils import XRobotStreamer, XRobotRecorder
except ImportError:
    XRobotStreamer = None
    XRobotRecorder = None