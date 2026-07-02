"""nav-core: Navigation math building blocks.

nav-core is a library of mathematical functions and data structures for
navigation algorithm development.  It is NOT an algorithm — it provides
the primitives that algorithm implementations can be built from.

Building blocks provided:
    Rotations      — quaternion [w,x,y,z] ↔ DCM ↔ ZYX Euler; composition; vector rotation.
    Frame transforms — WGS-84 LLH ↔ ECEF ↔ NED/ENU; lever-arm translation.
    GNSS error budget — UERE composition; DOP → σ_pos; NACp/EPU (DO-260B); timing-to-range.
    Ranging geometry — Euclidean range, TDOA range-difference, line-of-sight unit
                       vector, azimuth/elevation bearing, local-beacon range-DOP.
    EKF / ESKF     — general-purpose filter building blocks (see examples/ for usage).

All functions are thin wrappers around the C++ extension _navcore.  Import
directly from navcore or from navcore._navcore for the raw C++ bindings.

Examples showing how to compose these into an estimator live in examples/.
"""

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
        range_m,
        range_diff_m,
        los_unit,
        bearing_from_ned,
        range_dop,
        ESKF_STATE_DIM,
        SPEED_OF_LIGHT_M_PER_S,
        WGS84_A,
        WGS84_F,
        WGS84_E2,
    )
except ImportError as exc:
    raise ImportError(
        "nav-core C++ extension not found.  "
        "Build with: pip install -e . (requires CMake, pybind11).\n"
        f"Original error: {exc}"
    ) from exc

__version__ = "0.2.2"
__all__ = [
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
    "range_m",
    "range_diff_m",
    "los_unit",
    "bearing_from_ned",
    "range_dop",
    "ESKF_STATE_DIM",
    "SPEED_OF_LIGHT_M_PER_S",
    "WGS84_A",
    "WGS84_F",
    "WGS84_E2",
]
