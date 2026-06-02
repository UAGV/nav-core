"""Example: building an INS/GNSS estimator with nav-core building blocks.

This file is NOT part of the nav-core library.  It demonstrates how the
nav-core primitives (Eskf, frame transforms, GNSS error-budget functions)
can be composed to build an algorithm.  A production implementation would
live in its own package and would add alignment, outlier rejection, and
integrity monitoring on top of this skeleton.

Usage::

    # Requires nav-data installed
    from navdata.store import hdf5
    from examples.eskf_ins_example import run_eskf_example

    dataset = hdf5.read("path/to/dataset.h5")
    result = run_eskf_example(dataset)
    print(f"Final NED position: {result['position_ned_m'][-1]}")
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

import navcore


def run_eskf_example(
    dataset: Any,
    *,
    sigma_gyro_rad_per_s: float = 1e-3,
    sigma_accel_m_per_s2: float = 1e-2,
    gravity_m_per_s2: float = 9.80665,
) -> Dict[str, np.ndarray]:
    """Run an ESKF over a NavDataset and return raw numpy arrays.

    Returns a plain dict so the caller decides what to do with the results.
    State ordering of the covariance matches ESKF_STATE_DIM convention:
      0-2: dp [m], 3-5: dv [m/s], 6-8: dpsi [rad], 9-11: dbg, 12-14: dba.
    """
    if dataset.imu is None:
        raise ValueError("dataset must contain IMU data")

    imu = dataset.imu
    N = len(imu)

    nom = navcore.NominalState()
    nom.position_ned_m = np.zeros(3)
    nom.velocity_ned_m_per_s = np.zeros(3)
    nom.q_body_from_ned = np.array([1.0, 0.0, 0.0, 0.0])

    P0 = np.diag([
        1.0, 1.0, 1.0,
        0.1, 0.1, 0.1,
        1e-4, 1e-4, 1e-4,
        1e-6, 1e-6, 1e-6,
        1e-4, 1e-4, 1e-4,
    ]).astype(np.float64)

    eskf = navcore.Eskf(nom, P0, gravity_m_per_s2)

    gnss = dataset.gnss
    gnss_lookup: Dict[int, int] = {}
    ref_llh: Optional[np.ndarray] = None
    if gnss is not None and len(gnss) > 0:
        for idx, ts in enumerate(gnss.timestamps_ns):
            gnss_lookup[int(ts)] = idx
        ref_llh = gnss.position_llh[0].copy()

    timestamps_ns = imu.timestamps_ns.astype(np.int64)
    out_pos  = np.empty((N, 3),      dtype=np.float64)
    out_vel  = np.empty((N, 3),      dtype=np.float64)
    out_quat = np.empty((N, 4),      dtype=np.float64)
    out_cov  = np.empty((N, 15, 15), dtype=np.float64)

    for i in range(N):
        if i > 0:
            dt_s = (int(timestamps_ns[i]) - int(timestamps_ns[i - 1])) * 1e-9
            if dt_s > 0.0:
                Q = _build_Q(dt_s, sigma_gyro_rad_per_s, sigma_accel_m_per_s2)
                eskf.predict(
                    imu.angular_velocity[i].astype(np.float64),
                    imu.linear_acceleration[i].astype(np.float64),
                    dt_s, Q,
                )

        if gnss is not None and ref_llh is not None and int(timestamps_ns[i]) in gnss_lookup:
            g_idx = gnss_lookup[int(timestamps_ns[i])]
            pos_ned = navcore.llh_to_ned(gnss.position_llh[g_idx].astype(np.float64), ref_llh)
            sh = float(gnss.h_accuracy_m[g_idx]) if gnss.h_accuracy_m is not None else 5.0
            sv = float(gnss.v_accuracy_m[g_idx]) if gnss.v_accuracy_m is not None else 10.0
            eskf.update_gnss_position(pos_ned, np.array([sh*sh, sh*sh, sv*sv]))

        ns = eskf.nominal_state
        out_pos[i]  = np.asarray(ns.position_ned_m)
        out_vel[i]  = np.asarray(ns.velocity_ned_m_per_s)
        out_quat[i] = np.asarray(ns.q_body_from_ned)
        out_cov[i]  = eskf.error_covariance

    return {
        "timestamps_ns":        timestamps_ns,
        "position_ned_m":       out_pos,
        "velocity_ned_m_per_s": out_vel,
        "attitude_quat":        out_quat,
        "covariance":           out_cov,
    }


def _build_Q(dt_s: float, sg: float, sa: float) -> np.ndarray:
    d = np.array([
        1e-6*dt_s, 1e-6*dt_s, 1e-6*dt_s,
        sa**2*dt_s, sa**2*dt_s, sa**2*dt_s,
        sg**2*dt_s, sg**2*dt_s, sg**2*dt_s,
        (1e-5)**2*dt_s, (1e-5)**2*dt_s, (1e-5)**2*dt_s,
        (1e-4)**2*dt_s, (1e-4)**2*dt_s, (1e-4)**2*dt_s,
    ])
    return np.diag(d).astype(np.float64)
