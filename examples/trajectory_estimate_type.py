"""Example: a TrajectoryEstimate dataclass that wraps ESKF output.

This is NOT part of the nav-core library.  It shows how a downstream package
(e.g. nav-eval) might define an output type that consumes the arrays returned
by an ESKF-based estimator.

When nav-data v0.2.0 is tagged, TrajectoryEstimate will be defined there and
this file will be superseded by that definition.

State ordering (matches navcore.ESKF_STATE_DIM convention):
    0-2:   dp   position error   [m]      NED
    3-5:   dv   velocity error   [m/s]    NED
    6-8:   dpsi attitude error   [rad]    rotation vector
    9-11:  dbg  gyro bias error  [rad/s]
    12-14: dba  accel bias error [m/s2]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class TrajectoryEstimate:
    """Output of an estimator run — wraps ESKF output arrays.

    Attributes
    ----------
    timestamps_ns : (N,) int64
    position_ned_m : (N, 3) float64
    velocity_ned_m_per_s : (N, 3) float64
    attitude_quat : (N, 4) float64  [w, x, y, z] body-from-NED
    covariance : (N, 15, 15) float64  full error covariance per epoch
    meta : dict  includes covariance_state_order, estimator, etc.
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
            self.meta = {
                "covariance_state_order": [
                    "dp_N_m", "dp_E_m", "dp_D_m",
                    "dv_N_m_per_s", "dv_E_m_per_s", "dv_D_m_per_s",
                    "dpsi_x_rad", "dpsi_y_rad", "dpsi_z_rad",
                    "dbg_x_rad_per_s", "dbg_y_rad_per_s", "dbg_gz_rad_per_s",
                    "dba_x_m_per_s2", "dba_y_m_per_s2", "dba_z_m_per_s2",
                ],
                "body_convention":  "FRD",
                "world_convention": "NED",
            }

    def __len__(self) -> int:
        return int(self.timestamps_ns.shape[0])

    def summary(self) -> str:
        duration = (int(self.timestamps_ns[-1]) - int(self.timestamps_ns[0])) / 1e9
        return f"TrajectoryEstimate(n={len(self):,}, duration={duration:.1f}s)"
