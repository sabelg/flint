"""Flint v5 Camera Rig Package (`flint.camera`)."""
from flint.camera.rig import (
    SpringArm,
    CameraRig,
    CAM_MODE_FIRST_PERSON,
    CAM_MODE_ORBIT_FOLLOW,
    CAM_MODE_RAIL_DOLLY,
    CAM_MODE_FREE_FLY,
)
# --- sota:camera hook (physical camera model) ---
from flint.camera.physical import (
    ExposureTriangle,
    PhysicalCameraModel,
    PHYS_EXPOSURE_APERTURE_PRIORITY,
    PHYS_EXPOSURE_MANUAL,
    PHYS_EXPOSURE_PROGRAM,
    PHYS_EXPOSURE_SHUTTER_PRIORITY,
)
# --- end sota:camera hook ---
