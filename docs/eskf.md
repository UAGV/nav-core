# The ESKF — a 15-state error-state Kalman filter engine

`Eskf` is the one piece of nav-core that is a *filter*, not a stateless function —
and it is deliberately an **engine, not an algorithm**. It mechanises an IMU and
exposes a generic measurement update; *which* sensors you fuse, *when*, and *how
you reject outliers* is your algorithm. This page shows how to drive it, with
every worked number produced by the shipped module.

`from navcore import Eskf, NominalState`

---

## Why error-state, and what "15-state" means

Attitude lives on the rotation manifold SO(3), not in a vector space. Tracking a
quaternion directly forces a norm constraint every step and awkward linearisation.
The ESKF instead splits the state in two:

1. A **nominal state** propagated through the *exact* nonlinear IMU kinematics.
2. A small **error state** `δx ∈ ℝ¹⁵` in the tangent space around the nominal,
   on which a *linear* Kalman filter runs.
3. After each update the estimated error is injected into the nominal state and
   the error state is reset to zero.

This keeps the covariance well-conditioned and is the industry standard for
INS/GNSS (Farrell 2008; Solà 2017).

**The 15-state error order is fixed across the whole toolkit** (nav-eval reads the
covariance positionally, so it can never change):

| Indices | Error state | Symbol | Units |
|---------|-------------|--------|-------|
| 0–2 | position error (NED) | δp | m |
| 3–5 | velocity error (NED) | δv | m/s |
| 6–8 | attitude error (small rotation vector) | δψ | rad |
| 9–11 | gyro bias error | δb_g | rad/s |
| 12–14 | accel bias error | δb_a | m/s² |

The **position covariance sub-block is `error_covariance[0:3, 0:3]`** — that is
what nav-eval's position NEES uses.

---

## The conventions that bite

- **Feed the accelerometer raw.** The IMU specific force is physical: `f = a − g`,
  so at level rest the accelerometer reads `[0, 0, −9.80665]` ("1 g up" in FRD).
  The filter reconstructs `a = Rᵀf + g` internally. Do **not** add gravity back
  yourself — that bug injected ~2 g of phantom acceleration in an earlier version
  (logged in `nav-core-build-brief.md §A`).
- **`q_body_from_ned` is `[w, x, y, z]`, body-from-world.** Seed it from
  [Rotations](rotations.md).
- **`Q` (process noise) is 15×15 and scaled by `dt`.** You supply it each
  `predict`; it is typically diagonal (see Example 6).
- **`dt_s` is a float in seconds**, computed from the int64-ns timestamps
  (`dt_s = (t_ns[k] − t_ns[k−1]) * 1e-9`).

---

## API

### `NominalState` (the full, non-error state)

```python
nominal = navcore.NominalState()        # all-zero, identity quaternion
nominal.position_ned_m         # (3,) m
nominal.velocity_ned_m_per_s   # (3,) m/s
nominal.q_body_from_ned        # (4,) [w,x,y,z]
nominal.bias_gyro_rad_per_s    # (3,) rad/s
nominal.bias_accel_m_per_s2    # (3,) m/s²
```

### `Eskf`

| Member | Signature | Purpose |
|--------|-----------|---------|
| constructor | `Eskf(initial_nominal: NominalState, initial_covariance_15x15: (15,15), gravity_m_per_s2=9.80665)` | build the filter |
| `predict` | `predict(gyro_body_rad_per_s: (3,), accel_body_m_per_s2: (3,), dt_s: float, Q_15x15: (15,15))` | IMU mechanisation + error propagation |
| `update_gnss_position` | `update_gnss_position(gnss_position_ned_m: (3,), position_cov_diagonal_m2: (3,))` | built-in GNSS position fix (diagonal R) |
| `update` | `update(innovation_z: list[float], H_Mx15: list[float], R_MxM: list[float], M: int)` | generic update — *your* measurement model |
| `nominal_state` | property → `NominalState` | the current estimate |
| `error_covariance` | property → `(15,15)` | error covariance; position block is `[0:3,0:3]` |

> **Read this before using `innovation` / `innovation_covariance`.** These two
> read-only properties exist, but they return an **empty array after every
> built-in update**: `update` and `update_gnss_position` both inject the error and
> rebuild the inner filter (the "reset" step), which clears them. To gate
> outliers or compute a NIS you must form the innovation and its covariance
> **yourself, before calling `update`** — see Example 7. (Exposing the
> pre-reset innovation through the bindings is a candidate fix under **NAV-015**.)

---

## Worked examples

### Example 1: construct, and verify it sits still

A filter at the origin, level, at rest, integrated for 1 s of "1 g up"
specific-force samples, must not move:

```python
import numpy as np, navcore

def build_eskf():
    nominal = navcore.NominalState()
    nominal.position_ned_m       = np.zeros(3)
    nominal.velocity_ned_m_per_s = np.zeros(3)
    nominal.q_body_from_ned      = np.array([1.0, 0.0, 0.0, 0.0])
    # initial covariance: 1 m² pos, 0.01 (m/s)² vel, small attitude/bias
    P0 = np.diag([1.0]*3 + [0.01]*3 + [1e-4]*3 + [1e-6]*3 + [1e-4]*3)
    return navcore.Eskf(nominal, P0, gravity_m_per_s2=9.80665)

def build_Q(dt):
    return np.diag([1e-6*dt]*3 + [1e-4*dt]*3 + [1e-6*dt]*3 + [1e-10*dt]*3 + [1e-8*dt]*3)

eskf = build_eskf()
dt, Q = 0.01, build_Q(0.01)
gyro  = np.zeros(3)
accel = np.array([0.0, 0.0, -9.80665])     # physical f = a − g: 1 g up at rest
for _ in range(100):                        # 1 second at 100 Hz
    eskf.predict(gyro, accel, dt, Q)

eskf.nominal_state.position_ned_m        # → [0.0, 0.0, 0.0]
eskf.nominal_state.velocity_ned_m_per_s  # → [0.0, 0.0, 0.0]
```

Pinned by `test_eskf_stationary_stays_at_origin`. This is the test that proves the
gravity sign is right — feed `[0,0,+9.80665]` instead and the platform "falls" at
2 g.

### Example 2: a constant acceleration integrates correctly

Same filter, but the body now feels 1 m/s² to the North (plus the 1 g up):

```python
import numpy as np
eskf = build_eskf()
accel = np.array([1.0, 0.0, -9.80665])     # 1 m/s² north, 1 g up
for _ in range(100):                        # 1 second
    eskf.predict(np.zeros(3), accel, 0.01, build_Q(0.01))

eskf.nominal_state.velocity_ned_m_per_s[0]   # → 1.0    (v = a·t)
eskf.nominal_state.position_ned_m[0]          # → 0.5    (p = ½·a·t²)
```

Pinned by `test_eskf_constant_north_accel`. Textbook kinematics, straight out of
the mechanisation.

### Example 3: a GNSS update, worked by hand

The built-in GNSS update is a 3-axis position fix with diagonal noise. Start with
a deliberately wrong North position (10 m, true is ~8 m) and a 4 m² prior on it:

```python
import numpy as np, navcore
nominal = navcore.NominalState()
nominal.position_ned_m  = np.array([10.0, 0.0, 0.0])
nominal.q_body_from_ned = np.array([1.0, 0.0, 0.0, 0.0])
P0 = np.diag([4.0, 1.0, 1.0] + [0.01]*3 + [1e-4]*3 + [1e-6]*3 + [1e-4]*3)
eskf = navcore.Eskf(nominal, P0, 9.80665)

eskf.update_gnss_position(np.array([8.0, 0.0, 0.0]),   # GNSS says 8 m North
                          np.array([1.0, 1.0, 4.0]))    # R diagonal (m²)

eskf.nominal_state.position_ned_m[0]   # → 8.4
eskf.error_covariance[0, 0]            # → 0.8
```

Check the Kalman update by hand on the North axis: gain
`K = P / (P + R) = 4 / (4 + 1) = 0.8`; correction
`δp_N = K·(8 − 10) = −1.6`; updated position `10 − 1.6 = 8.4`; updated variance
`(1 − K)·P = 0.2·4 = 0.8`. Both pinned by `test_eskf_gnss_update_corrects_error`.

### Example 4: choosing the GNSS R from the error budget

The `position_cov_diagonal_m2` argument above is exactly the diagonal of
`gnss_position_covariance_ned` from [GNSS error budget](gnss-error.md):

```python
import numpy as np, navcore
R_full = navcore.gnss_position_covariance_ned(hdop=1.0, vdop=2.0, sigma_uere_m=1.5)
R_diag = np.diag(R_full)                  # → [2.25, 2.25, 9.0]
eskf.update_gnss_position(position_ned, R_diag)
```

That is the whole chain from "the receiver reported HDOP = 1.0" to a fused fix.

### Example 5: generic `update`

For any sensor that is not a 3-axis GNSS fix, use the generic `update`. You supply
the **innovation** (measurement minus your prediction), the **measurement
Jacobian** `H` (M×15, partial of the measurement w.r.t. the error state), and the
**measurement noise** `R` (M×M). A 1-D barometric/lidar altitude fix observes the
Down position (index 2). The platform thinks it is at D = −100 m (100 m up); the
sensor implies D = −98 m:

```python
import numpy as np, navcore
nominal = navcore.NominalState()
nominal.position_ned_m  = np.array([0.0, 0.0, -100.0])
nominal.q_body_from_ned = np.array([1.0, 0.0, 0.0, 0.0])
P0 = np.diag([1.0, 1.0, 9.0] + [0.01]*3 + [1e-4]*3 + [1e-6]*3 + [1e-4]*3)
eskf = navcore.Eskf(nominal, P0, 9.80665)

innovation = [(-98.0) - (-100.0)]         # z − h(x_nominal) = +2.0 on the D axis
H = np.zeros((1, 15)); H[0, 2] = 1.0       # selects δp_D (index 2)
R = [[4.0]]                                 # (2 m)² measurement noise
eskf.update(innovation, H.flatten().tolist(), np.array(R).flatten().tolist(), M=1)

eskf.nominal_state.position_ned_m[2]   # → -98.6154
eskf.error_covariance[2, 2]            # → 2.7692
```

By hand: `K = 9 / (9 + 4) = 0.6923`; correction `0.6923·2 = +1.3846`;
`−100 + 1.3846 = −98.6154`; variance `(1 − 0.6923)·9 = 2.7692`. **The pattern
generalises:** a velocity update puts identity in columns 3–5; a magnetometer
heading update fills the attitude columns 6–8; a **UWB range** update puts
`−los_unit` (from [Ranging geometry](ranging.md)) in the position columns 0–2.
A new sensor costs you a Jacobian and a noise matrix — never a library change.

### Example 6: building the process noise `Q`

`Q` is how fast you let the filter trust its IMU less. Diagonal, scaled by `dt`:
the velocity block from the accelerometer noise density, the attitude block from
the gyro noise density, the bias blocks from the bias-instability random walk.
(The toy `build_Q(dt)` in Examples 1–2 used fixed constants; this is the realistic
form you would actually tune.)

```python
import numpy as np
def build_process_noise(dt, accel_noise_density, gyro_noise_density,
                        accel_bias_rw, gyro_bias_rw):
    """15×15 diagonal process noise, scaled by the timestep dt."""
    position_q   = [1e-9] * 3                          # δp: negligible direct noise
    velocity_q   = [accel_noise_density**2 * dt] * 3   # δv driven by accel white noise
    attitude_q   = [gyro_noise_density**2  * dt] * 3   # δψ driven by gyro white noise
    gyro_bias_q  = [gyro_bias_rw**2  * dt] * 3         # δb_g random walk
    accel_bias_q = [accel_bias_rw**2 * dt] * 3         # δb_a random walk
    return np.diag(position_q + velocity_q + attitude_q + gyro_bias_q + accel_bias_q)
```

**The honest-covariance rule:** match these densities to the IMU's *actual* noise.
nav-data carries the four calibration figures on the dataset's `/imu` group, and
nav-sim's `ImuSensor` uses the same values to corrupt the truth — so reading
process noise straight from the dataset is what makes the filter's reported
covariance *consistent* (ANEES ≈ 3) rather than guessed.

### Example 7: gating outliers and computing NIS yourself

Because the built-in `innovation` accessor is cleared by the reset (see the API
note), compute the normalised innovation squared (NIS) before you call `update`,
and skip the update if it is implausible:

```python
import numpy as np

def gated_update(eskf, z_measured, h_predicted, H, R, chi2_gate):
    innovation = np.asarray(z_measured) - np.asarray(h_predicted)
    P = eskf.error_covariance
    S = H @ P @ H.T + R                       # innovation covariance
    nis = float(innovation @ np.linalg.solve(S, innovation))
    if nis > chi2_gate:                       # e.g. 7.81 for 3 DOF at 95%
        return False                           # reject — likely multipath / blunder
    eskf.update(innovation.tolist(), H.flatten().tolist(),
                R.flatten().tolist(), M=len(innovation))
    return True
```

NIS is the per-update analogue of the NEES that nav-eval reports over a whole run.
A filter whose NIS sits around `M` (the measurement dimension) is consistent; one
that is persistently large is over-confident in its prediction.

### Example 8: the canonical INS/GNSS loop

Putting it together — predict on every IMU sample, update when a GNSS fix lands
(this is the shape of `examples/eskf_ins_example.py`):

```python
import numpy as np, navcore

eskf = build_eskf()
out_position   = np.zeros((num_imu, 3))
out_covariance = np.zeros((num_imu, 15, 15))

for k in range(num_imu):
    dt_s = (imu_timestamps_ns[k] - imu_timestamps_ns[k-1]) * 1e-9
    eskf.predict(imu_gyro[k], imu_accel[k], dt_s, build_Q(dt_s))   # raw accel
    if gnss_fix_available_at(k):
        position_ned = navcore.llh_to_ned(gnss_fix_llh, reference_llh)
        eskf.update_gnss_position(position_ned, gnss_R_diag)
    out_position[k]   = eskf.nominal_state.position_ned_m
    out_covariance[k] = eskf.error_covariance

# horizontal 1-sigma you can publish as a quality scalar:
horizontal_sigma_m = np.sqrt(out_covariance[:, 0, 0] + out_covariance[:, 1, 1])
```

The `out_position` / `out_covariance` arrays are exactly what you wrap in a
`navdata.AlgorithmOutput` for nav-eval to score.

---

## Where to take this next

- **Score it.** Wrap the loop's output in an `AlgorithmOutput` (with the 15-state
  covariance) and run `naveval.evaluate` — the position NEES reads
  `error_covariance[0:3,0:3]` and tells you whether your reported uncertainty
  honestly bounds the truth.
- **See a complete algorithm.** The nav-algorithm-template's `example-algorithm`
  branch wraps this engine as a streaming `initialise/ingest/extract` object that
  also fuses magnetometer, barometer, and lidar via the generic `update` — scoring
  ~0.4 m ATE / ~0.6° attitude / ANEES ~1.6.
- **Generate test data with known truth.** nav-sim produces a `NavDataset`
  byte-identical to a real one, so you validate this filter on perfect truth
  before trusting field data.
- **Limits to know.** The Python `Eskf` exposes only `predict`,
  `update_gnss_position`, and a generic `update`; there is no built-in ZUPT,
  non-holonomic constraint, or joint (non-diagonal) GNSS covariance, and the
  innovation accessors are reset (above). These are tracked enhancements under
  **NAV-015**; until then, build them on top with the generic `update`.
