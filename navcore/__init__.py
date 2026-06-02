"""nav-core: Navigation math foundation.

Public surface:
    Rotations and frame transforms — via the _navcore C++ extension.
    GNSS error-budget math         — via the _navcore C++ extension.
    ESKF estimator                 — via the _navcore C++ extension + Python wrapper.
    TrajectoryEstimate             — pure Python dataclass (temporary, pre nav-data v0.2.0).

Import the C++ bindings directly for low-level use:
    from navcore._navcore import Eskf, NominalState, quaternion_to_dcm, ...

Or use the high-level Python wrappers:
    from navcore.estimator import run_eskf
    from navcore.types import TrajectoryEstimate
"""

from navcore.types import TrajectoryEstimate
from navcore.estimator import run_eskf

try:
    from navcore._navcore import (  # type: ignore[import]
        Quaternion,
        NominalState,
        Eskf,
        quaternion_to_dcm,
        dcm_to_quaternion,
        quaternion_to_euler_zyx,
        euler_zyx_to_quaternion,
        rotate_vector,
        llh_to_ecef,
        ecef_to_llh,
        llh_to_ned,
        ned_to_enu,
        enu_to_ned,
        apply_lever_arm,
        compute_uere,
        dop_to_position_sigma,
        pdop_from_hdop_vdop,
        epu_to_nacp,
        nacp_to_epu_threshold,
        epu_95_to_sigma_h,
        timing_to_range_sigma,
        timing_to_tdoa_range_sigma,
        gnss_position_covariance_ned,
        ESKF_STATE_DIM,
        SPEED_OF_LIGHT_M_PER_S,
        WGS84_A,
    )
except ImportError as exc:
    raise ImportError(
        "nav-core C++ extension not found.  "
        "Build with: pip install -e . (requires CMake, pybind11).\n"
        f"Original error: {exc}"
    ) from exc

__version__ = "0.1.0"
__all__ = [
    "TrajectoryEstimate",
    "run_eskf",
    "Quaternion",
    "NominalState",
    "Eskf",
    "quaternion_to_dcm",
    "dcm_to_quaternion",
    "quaternion_to_euler_zyx",
    "euler_zyx_to_quaternion",
    "rotate_vector",
    "llh_to_ecef",
    "ecef_to_llh",
    "llh_to_ned",
    "ned_to_enu",
    "enu_to_ned",
    "apply_lever_arm",
    "compute_uere",
    "dop_to_position_sigma",
    "pdop_from_hdop_vdop",
    "epu_to_nacp",
    "nacp_to_epu_threshold",
    "epu_95_to_sigma_h",
    "timing_to_range_sigma",
    "timing_to_tdoa_range_sigma",
    "gnss_position_covariance_ned",
    "ESKF_STATE_DIM",
    "SPEED_OF_LIGHT_M_PER_S",
    "WGS84_A",
]
