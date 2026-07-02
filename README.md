# nav-core

Math building blocks for navigation algorithm development.

nav-core is a **library of functions and data structures**, not an algorithm.
It provides the primitives that navigation algorithm implementations can be built
from: rotation math, coordinate-frame transforms, GNSS error-budget functions,
and filter building blocks (EKF/ESKF).  What you do with those building blocks
— the algorithm itself — lives elsewhere.

## Building blocks

| Module | What it provides |
|--------|-----------------|
| `quaternion.hpp` | `[w,x,y,z]` quaternion ↔ DCM ↔ ZYX Euler; Hamilton product; vector rotation |
| `frames.hpp` | WGS-84 LLH ↔ ECEF ↔ NED/ENU; Bowring iterative ECEF→LLH; lever-arm translation |
| `gnss_error.hpp` | UERE composition; σ_pos = DOP·σ_UERE; PDOP; NACp/EPU (DO-260B); σ_r = c·σ_t and √2 TDOA factor |
| `ekf.hpp` | Template `Ekf<N>`: predict/update with caller-supplied Jacobians; exposes innovation ν and S for NIS |
| `eskf.hpp` | `Eskf`: 15-state error-state KF building block (the filter engine for aided inertial navigation, INS/GNSS); `predict()`, `update()`, `update_gnss_position()` |

The C++ headers are dependency-free.  Python bindings expose everything to numpy.

## Documentation

[`docs/`](docs/) documents the Python API the way you use it while building an
algorithm — each function group with a concept section, the conventions that
bite, and **worked numerical examples you can check by hand** (every number is
produced by the shipped module and pinned by [`tests/python/test_bindings.py`](tests/python/test_bindings.py)).

- [`docs/README.md`](docs/README.md) — index + the conventions that bite (start here)
- [`docs/rotations.md`](docs/rotations.md) — quaternions, DCMs, ZYX Euler, `rotate_vector`
- [`docs/frames.md`](docs/frames.md) — LLH ↔ ECEF ↔ NED/ENU and the antenna lever arm
- [`docs/gnss-error.md`](docs/gnss-error.md) — UERE → DOP → σ_pos, NACp/EPU, timing-to-range
- [`docs/ranging.md`](docs/ranging.md) — range, TDOA, line-of-sight Jacobian, bearing, range-DOP
- [`docs/eskf.md`](docs/eskf.md) — driving the 15-state ESKF engine end to end

## Examples

`examples/` shows how to compose these building blocks into an algorithm:

- [`eskf_ins_example.py`](examples/eskf_ins_example.py) — driving the ESKF with IMU data and GNSS position fixes over a nav-data `NavDataset`.
- [`trajectory_estimate_type.py`](examples/trajectory_estimate_type.py) — a `TrajectoryEstimate` dataclass wrapping ESKF output (precursor to nav-data v0.2.0).

## Install

```bash
# Requires CMake >= 3.18, pybind11, a C++17 compiler
pip install -e .
```

## Quick reference

```python
import navcore
import numpy as np

# --- Rotations ---
q = navcore.euler_zyx_to_quaternion(roll_rad=0.1, pitch_rad=0.0, yaw_rad=1.57)
R = navcore.quaternion_to_dcm(q)                    # (3, 3) float64
v_body = navcore.rotate_vector(q, v_world)          # rotate world → body

# --- Frame transforms ---
ecef = navcore.llh_to_ecef(lat_deg=51.5, lon_deg=-0.12, height_m=50.0)
ned  = navcore.llh_to_ned(pos_llh, ref_llh)         # NED offset from ref [m]
enu  = navcore.ned_to_enu(ned)

# --- GNSS error budget ---
uere    = navcore.compute_uere(1.0, 0.5, 2.0, 0.5, 0.3, 0.3)  # σ_clock, σ_eph, σ_iono, ...
sigma_h = navcore.dop_to_position_sigma(hdop=1.2, sigma_uere_m=uere)
nacp    = navcore.epu_to_nacp(epu_95_m=2.0 * sigma_h)
sigma_r = navcore.timing_to_range_sigma(sigma_time_s=10e-9)    # ~3 m per 10 ns

# --- Filter building blocks ---
nom = navcore.NominalState()
nom.position_ned_m       = np.zeros(3)
nom.velocity_ned_m_per_s = np.zeros(3)
nom.q_body_from_ned      = np.array([1.0, 0.0, 0.0, 0.0])
P0   = np.eye(15) * 1e-2
eskf = navcore.Eskf(nom, P0, gravity_m_per_s2=9.80665)

eskf.predict(gyro_body_rad_per_s, accel_body_m_per_s2, dt_s=0.01, Q_15x15=Q)
eskf.update_gnss_position(gnss_ned_m, position_cov_diagonal_m2)
P = eskf.error_covariance           # (15, 15) — use P[0:3, 0:3] for position sub-block
```

## Tests

```bash
# Python
pytest tests/python/ -q

# C++ (Catch2 fetched automatically)
cmake -B build -DBUILD_TESTS=ON && cmake --build build
cd build && ctest --output-on-failure
```

## ESKF state ordering (15-state error model)

Fixed — downstream packages that consume the covariance must use this ordering:

| Index | State | Units |
|-------|-------|-------|
| 0–2   | δp    | m (NED) |
| 3–5   | δv    | m/s (NED) |
| 6–8   | δψ    | rad (rotation vector) |
| 9–11  | δb_g  | rad/s |
| 12–14 | δb_a  | m/s² |

## Conventions

- Quaternions: `[w, x, y, z]`, body-from-world (NED).  Body frame FRD, world frame NED.
- Timestamps (where applicable): `int64` nanoseconds since Unix epoch — never `float64`.
- All floating-point: `double` / `float64`.

## Dependencies

| Layer | Deps |
|-------|------|
| C++ headers | C++17 stdlib only |
| Python bindings | pybind11 ≥ 2.11, numpy ≥ 1.24 |
| Build | CMake ≥ 3.18, scikit-build-core ≥ 0.6 |
