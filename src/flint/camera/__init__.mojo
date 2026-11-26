"""Flint v5 Camera Rig Package (`flint.camera`)."""
from flint.camera.rig import (
    SpringArm,
    CameraRig,
    CAM_MODE_FIRST_PERSON,
    CAM_MODE_ORBIT_FOLLOW,
    CAM_MODE_RAIL_DOLLY,
    CAM_MODE_FREE_FLY,
)
from flint.camera.vcam import (
    VirtualCamera,
    CameraBrain,
    VCAM_FOLLOW,
    VCAM_ORBIT,
    VCAM_FIXED,
    VCAM_LOCK_ON,
    VCAM_FIRST_PERSON,
)
from flint.camera.plugin import (
    CameraPlugin,
    CameraBrainResource,
    VirtualCameraComponent,
    register_camera_components,
)
