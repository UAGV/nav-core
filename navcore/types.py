"""nav-core output types.

TrajectoryEstimate is defined here temporarily until nav-data v0.2.0 ships.
When nav-data >= v0.2.0 is tagged (by the nav-eval workstream), this type
will move to navdata.types and this module will re-export it from there.
The migration is mechanical: swap the import site in estimator.py and here.

State ordering for the covariance array (FIXED, shared with nav-eval):
    dim 0– 2:  δp   position error [m]    NED
    dim 3– 5:  δv   velocity error [m/s]  NED
    dim 6– 8:  δψ   attitude error [rad]  rotation vector
    dim 9–11:  δb_g gyro bias error [rad/s]
    dim 12–14: δb_a accel bias error [m/s²]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class TrajectoryEstimate:
    """Output of a nav-core estimator run.

    This is what nav-eval consumes to compute trajectory-error and consistency
    metrics.  The shapes mirror GroundTruth so the two can be aligned and
    differenced directly.

    Attributes
    ----------
    timestamps_ns : (N,) int64
        Nanoseconds since the Unix epoch, one per estimated pose.
    position_ned_m : (N, 3) float64
        NED position estimate [m].
    velocity_ned_m_per_s : (N, 3) float64
        NED velocity estimate [m/s].
    attitude_quat : (N, 4) float64
        Attitude quaternion [w, x, y, z], body-from-NED.
    covariance : (N, 15, 15) float64
        Full 15-state error covariance per epoch.
        State ordering declared in meta['covariance_state_order'].
    innovation : (N, M) float64, optional
        Filter innovation (residual) at each update epoch.
    innovation_covariance : (N, M, M) float64, optional
        Innovation covariance S at each update epoch.
        Used by nav-eval to compute NIS = νᵀ S⁻¹ ν.
    meta : dict
        At minimum includes:
          'covariance_state_order': list of 15 labels
          'body_convention':  e.g. 'FRD'
          'world_convention': e.g. 'NED'
          'estimator':        e.g. 'ESKF-15'
          'navcore_version':  e.g. '0.1.0'
    """

    timestamps_ns: np.ndarray
    position_ned_m: np.ndarray
    velocity_ned_m_per_s: np.ndarray
    attitude_quat: np.ndarray
    covariance: np.ndarray
    innovation: Optional[np.ndarray] = None
    innovation_covariance: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.meta:
            self.meta = _default_meta()
        _check_timestamps(self.timestamps_ns, "TrajectoryEstimate.timestamps_ns")
        _check_shape_2d(self.position_ned_m, 3, "TrajectoryEstimate.position_ned_m")
        _check_shape_2d(self.velocity_ned_m_per_s, 3, "TrajectoryEstimate.velocity_ned_m_per_s")
        _check_shape_2d(self.attitude_quat, 4, "TrajectoryEstimate.attitude_quat")
        _check_covariance(self.covariance)

    def __len__(self) -> int:
        return int(self.timestamps_ns.shape[0])

    def summary(self) -> str:
        """One-line human summary."""
        estimator = self.meta.get("estimator", "unknown")
        return (
            f"TrajectoryEstimate(n={len(self):,}, estimator={estimator!r}, "
            f"duration={(self.timestamps_ns[-1]-self.timestamps_ns[0])/1e9:.1f}s)"
        )


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _default_meta() -> Dict[str, Any]:
    return {
        "covariance_state_order": [
            "dp_N_m", "dp_E_m", "dp_D_m",
            "dv_N_m_per_s", "dv_E_m_per_s", "dv_D_m_per_s",
            "dpsi_x_rad", "dpsi_y_rad", "dpsi_z_rad",
            "db_gx_rad_per_s", "db_gy_rad_per_s", "db_gz_rad_per_s",
            "db_ax_m_per_s2", "db_ay_m_per_s2", "db_az_m_per_s2",
        ],
        "body_convention":  "FRD",
        "world_convention": "NED",
        "estimator":        "ESKF-15",
        "navcore_version":  "0.1.0",
    }


def _check_timestamps(arr: np.ndarray, label: str) -> None:
    if arr.dtype != np.int64:
        raise TypeError(f"{label} must be int64 nanoseconds, got dtype {arr.dtype}")
    if arr.ndim != 1:
        raise ValueError(f"{label} must be 1-D, got shape {arr.shape}")


def _check_shape_2d(arr: np.ndarray, cols: int, label: str) -> None:
    if arr.ndim != 2 or arr.shape[1] != cols:
        raise ValueError(f"{label} must have shape (N, {cols}), got {arr.shape}")


def _check_covariance(arr: np.ndarray) -> None:
    if arr.ndim != 3 or arr.shape[1] != 15 or arr.shape[2] != 15:
        raise ValueError(
            f"TrajectoryEstimate.covariance must have shape (N, 15, 15), got {arr.shape}"
        )
