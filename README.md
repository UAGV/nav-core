# nav-core

Navigation math foundation for the nav-toolkit. Provides:

- **Rotations** — quaternion `[w,x,y,z]` ↔ DCM ↔ ZYX Euler; normalisation; composition; vector rotation.
- **Frame transforms** — ECEF ↔ geodetic LLH ↔ NED/ENU (WGS-84); body↔nav; lever-arm application.
- **GNSS error-budget math** — UERE composition; DOP → σ_pos; HDOP/VDOP/PDOP; NACp/EPU (DO-260B); σ_r = c·σ_t (timing-to-range, with √2 TDOA factor).
- **Estimators** — a general EKF (predict/update with Jacobians) and a 15-state ESKF for INS (IMU mechanisation + GNSS position fusion).

The C++ core is header-only and dependency-free. Python bindings are provided via pybind11.

## State ordering (15-state ESKF)

Fixed across the toolkit — nav-eval uses this to compute NEES:

| Index | State | Units |
|-------|-------|-------|
| 0–2   | δp    | m (NED) |
| 3–5   | δv    | m/s (NED) |
| 6–8   | δψ    | rad (rotation vector) |
| 9–11  | δb_g  | rad/s |
| 12–14 | δb_a  | m/s² |

## Install

```bash
# Requires CMake >= 3.18, pybind11, a C++17 compiler
pip install -e .
```

## Quick start

```python
from navdata.store import hdf5
from navcore.estimator import run_eskf

dataset = hdf5.read("path/to/dataset.h5")
estimate = run_eskf(dataset)
print(estimate.summary())
# TrajectoryEstimate(n=10000, estimator='ESKF-15', duration=100.0s)
```

## Low-level API

```python
import navcore

# Quaternion math
q = navcore.euler_zyx_to_quaternion(roll_rad=0.1, pitch_rad=0.0, yaw_rad=1.57)
R = navcore.quaternion_to_dcm(q)

# Frame transforms
ecef = navcore.llh_to_ecef(51.5, -0.12, 50.0)
ned  = navcore.llh_to_ned(pos_llh, ref_llh)

# GNSS error budget
uere  = navcore.compute_uere(clock=1.0, ephemeris=0.5, iono=2.0, tropo=0.5, multipath=0.3, noise=0.3)
sigma_h = navcore.dop_to_position_sigma(hdop=1.2, sigma_uere_m=uere)
nacp  = navcore.epu_to_nacp(epu_95_m=2.0 * sigma_h)
sigma_r = navcore.timing_to_range_sigma(sigma_time_s=10e-9)   # ~3 m per 10 ns

# ESKF
import numpy as np
nom = navcore.NominalState()
nom.position_ned_m       = np.zeros(3)
nom.velocity_ned_m_per_s = np.zeros(3)
nom.q_body_from_ned      = np.array([1.0, 0.0, 0.0, 0.0])
P0   = np.eye(15) * 1e-2
eskf = navcore.Eskf(nom, P0, gravity_m_per_s2=9.80665)

eskf.predict(gyro_body_rad_per_s, accel_body_m_per_s2, dt_s=0.01, Q_15x15=Q)
eskf.update_gnss_position(gnss_ned_m, position_cov_diagonal_m2)
print(eskf.nominal_state.position_ned_m)
```

## Tests

```bash
# Python
pytest tests/python/ -q

# C++ (requires Catch2)
cmake -B build -DBUILD_TESTS=ON && cmake --build build
cd build && ctest --output-on-failure
```

## Dependencies

| Layer | Deps |
|-------|------|
| C++ core (headers) | C++17 stdlib only |
| Python bindings | pybind11 >= 2.11, numpy >= 1.24 |
| High-level API | + nav-data (types + store only) |
| Build | CMake >= 3.18, scikit-build-core >= 0.6 |

## Conventions

- Timestamps: `int64` nanoseconds since Unix epoch.
- Quaternions: `[w, x, y, z]`, body-from-world (NED).
- Body frame: FRD (Forward-Right-Down).
- World frame: NED (North-East-Down).
- All floating-point: `double` / `float64`.
