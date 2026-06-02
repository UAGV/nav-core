"""High-level Python wrapper: run an ESKF over a NavDataset.

This is the only layer in nav-core that imports from nav-data.  The C++
core knows nothing about NavDataset; this function marshals numpy arrays
from the dataset into the C++ Eskf and packs the results into a
TrajectoryEstimate.

Usage example::

    from navdata.store import hdf5
    from navcore.estimator import run_eskf

    dataset = hdf5.read("path/to/dataset.h5")
    estimate = run_eskf(dataset)
    print(estimate.summary())
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from navcore.types import TrajectoryEstimate, _default_meta


def run_eskf(
    dataset: Any,  # navdata.types.NavDataset — typed as Any to avoid hard import
    *,
    initial_position_ned_m: Optional[np.ndarray] = None,
    initial_velocity_ned_m_per_s: Optional[np.ndarray] = None,
    initial_attitude_quat: Optional[np.ndarray] = None,
    sigma_gyro_rad_per_s: float = 1e-3,
    sigma_accel_m_per_s2: float = 1e-2,
    sigma_gyro_bias_rad_per_s: float = 1e-5,
    sigma_accel_bias_m_per_s2: float = 1e-4,
    gnss_sigma_h_m: Optional[float] = None,
    gnss_sigma_v_m: Optional[float] = None,
    gravity_m_per_s2: float = 9.80665,
) -> TrajectoryEstimate:
    """Run a 15-state ESKF over a standardised NavDataset.

    The ESKF is driven by IMU data (predict step) and fuses GNSS position
    measurements (update step) whenever a GNSS fix is available.  Output
    timestamps match the IMU timestamps.

    Parameters
    ----------
    dataset : NavDataset
        From navdata.store.hdf5.read().  Must contain at least imu data.
    initial_position_ned_m : (3,) float64, optional
        Starting NED position.  Defaults to [0, 0, 0] if not provided.
    initial_velocity_ned_m_per_s : (3,) float64, optional
        Starting NED velocity.  Defaults to [0, 0, 0].
    initial_attitude_quat : (4,) float64 [w,x,y,z], optional
        Initial attitude.  Defaults to identity (level, north-pointing).
    sigma_gyro_rad_per_s : float
        Gyroscope white-noise spectral density [rad/s/√Hz] for Q.
    sigma_accel_m_per_s2 : float
        Accelerometer white-noise spectral density [m/s²/√Hz] for Q.
    sigma_gyro_bias_rad_per_s : float
        Gyro bias random-walk spectral density for Q.
    sigma_accel_bias_m_per_s2 : float
        Accel bias random-walk spectral density for Q.
    gnss_sigma_h_m : float, optional
        Override horizontal GNSS noise [m].  If None, uses h_accuracy_m
        from the dataset or defaults to 5.0 m.
    gnss_sigma_v_m : float, optional
        Override vertical GNSS noise [m].  If None, uses v_accuracy_m
        from the dataset or defaults to 10.0 m.
    gravity_m_per_s2 : float
        Local gravity magnitude.

    Returns
    -------
    TrajectoryEstimate
        Estimated trajectory at IMU rate with full 15-state covariance.
    """
    from navcore._navcore import Eskf, NominalState  # type: ignore[import]

    if dataset.imu is None:
        raise ValueError("run_eskf: dataset has no IMU data")

    imu = dataset.imu
    N = len(imu)
    timestamps_ns: np.ndarray = imu.timestamps_ns.astype(np.int64)

    # Initial state
    p0 = np.zeros(3) if initial_position_ned_m is None else np.asarray(initial_position_ned_m, dtype=np.float64)
    v0 = np.zeros(3) if initial_velocity_ned_m_per_s is None else np.asarray(initial_velocity_ned_m_per_s, dtype=np.float64)
    q0 = np.array([1.0, 0.0, 0.0, 0.0]) if initial_attitude_quat is None else np.asarray(initial_attitude_quat, dtype=np.float64)

    nom = NominalState()
    nom.position_ned_m         = p0
    nom.velocity_ned_m_per_s   = v0
    nom.q_body_from_ned         = q0

    # Initial covariance — diagonal, conservative priors
    P0 = np.diag([
        1.0, 1.0, 1.0,           # δp [m²]
        0.1, 0.1, 0.1,           # δv [m²/s²]
        (1e-2)**2, (1e-2)**2, (1e-2)**2,  # δψ [rad²]
        (1e-3)**2, (1e-3)**2, (1e-3)**2,  # δb_g [rad²/s²]
        (1e-2)**2, (1e-2)**2, (1e-2)**2,  # δb_a [m²/s⁴]
    ]).astype(np.float64)

    eskf = Eskf(nom, P0, gravity_m_per_s2)

    # Build GNSS lookup table (timestamp → index)
    gnss = dataset.gnss
    gnss_lookup: Dict[int, int] = {}
    if gnss is not None:
        for idx, ts in enumerate(gnss.timestamps_ns):
            gnss_lookup[int(ts)] = idx

    # Reference LLH for GNSS → NED conversion
    ref_llh: Optional[np.ndarray] = None
    if gnss is not None and len(gnss) > 0:
        ref_llh = gnss.position_llh[0].copy()

    # Output buffers
    out_pos   = np.empty((N, 3), dtype=np.float64)
    out_vel   = np.empty((N, 3), dtype=np.float64)
    out_quat  = np.empty((N, 4), dtype=np.float64)
    out_cov   = np.empty((N, 15, 15), dtype=np.float64)

    for i in range(N):
        # --- Predict ---
        if i == 0:
            dt_s = 0.0  # no propagation on first step
        else:
            dt_ns = int(timestamps_ns[i]) - int(timestamps_ns[i - 1])
            dt_s = dt_ns * 1e-9  # nanoseconds → seconds, double precision

        if dt_s > 0.0:
            Q = _build_process_noise_Q(
                dt_s,
                sigma_gyro_rad_per_s,
                sigma_accel_m_per_s2,
                sigma_gyro_bias_rad_per_s,
                sigma_accel_bias_m_per_s2,
            )
            eskf.predict(
                imu.angular_velocity[i].astype(np.float64),
                imu.linear_acceleration[i].astype(np.float64),
                dt_s,
                Q,
            )

        # --- GNSS update (if a fix coincides with this IMU timestamp) ---
        if gnss is not None and ref_llh is not None and int(timestamps_ns[i]) in gnss_lookup:
            g_idx = gnss_lookup[int(timestamps_ns[i])]
            from navcore._navcore import llh_to_ned  # type: ignore[import]
            pos_ned = llh_to_ned(gnss.position_llh[g_idx], ref_llh)

            sh = float(gnss.h_accuracy_m[g_idx]) if (gnss.h_accuracy_m is not None) else (gnss_sigma_h_m or 5.0)
            sv = float(gnss.v_accuracy_m[g_idx]) if (gnss.v_accuracy_m is not None) else (gnss_sigma_v_m or 10.0)
            if gnss_sigma_h_m is not None:
                sh = gnss_sigma_h_m
            if gnss_sigma_v_m is not None:
                sv = gnss_sigma_v_m
            cov_diag = np.array([sh * sh, sh * sh, sv * sv], dtype=np.float64)
            eskf.update_gnss_position(pos_ned, cov_diag)

        # --- Store ---
        ns = eskf.nominal_state
        out_pos[i]  = np.asarray(ns.position_ned_m)
        out_vel[i]  = np.asarray(ns.velocity_ned_m_per_s)
        out_quat[i] = np.asarray(ns.q_body_from_ned)
        out_cov[i]  = eskf.error_covariance

    meta = _default_meta()
    meta["dataset_name"]      = dataset.name
    meta["gravity_m_per_s2"]  = gravity_m_per_s2
    meta["reference_llh"]     = ref_llh.tolist() if ref_llh is not None else None

    return TrajectoryEstimate(
        timestamps_ns=timestamps_ns,
        position_ned_m=out_pos,
        velocity_ned_m_per_s=out_vel,
        attitude_quat=out_quat,
        covariance=out_cov,
        meta=meta,
    )


def _build_process_noise_Q(
    dt_s: float,
    sigma_gyro_rad_per_s: float,
    sigma_accel_m_per_s2: float,
    sigma_gyro_bias_rad_per_s: float,
    sigma_accel_bias_m_per_s2: float,
) -> np.ndarray:
    """Discrete-time process noise covariance Q (15×15).

    Uses the continuous-time noise densities scaled by dt (first-order
    discretisation).  For IMU rates >= 50 Hz this is accurate enough for
    navigation-grade sensors.

    State ordering mirrors ESKF_STATE_DIM:
      0-2:   δp  (driven by δv noise, negligible direct noise → small floor)
      3-5:   δv  (driven by accel noise)
      6-8:   δψ  (driven by gyro noise)
      9-11:  δb_g (gyro bias random walk)
      12-14: δb_a (accel bias random walk)
    """
    q_pos   = (1e-6 * dt_s) * np.ones(3)  # position: negligible direct noise
    q_vel   = (sigma_accel_m_per_s2 ** 2 * dt_s) * np.ones(3)
    q_psi   = (sigma_gyro_rad_per_s  ** 2 * dt_s) * np.ones(3)
    q_bg    = (sigma_gyro_bias_rad_per_s  ** 2 * dt_s) * np.ones(3)
    q_ba    = (sigma_accel_bias_m_per_s2  ** 2 * dt_s) * np.ones(3)
    q_diag  = np.concatenate([q_pos, q_vel, q_psi, q_bg, q_ba])
    return np.diag(q_diag).astype(np.float64)
