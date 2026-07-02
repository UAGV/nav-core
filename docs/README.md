# nav-core documentation

nav-core is the **math building-block library** the rest of the toolkit — and
your algorithm — is composed from. It is *not* an algorithm: it gives you
rotations, frame transforms, the GNSS error budget, ranging geometry, and a
15-state error-state Kalman filter *engine*. What you fuse, when, and how is your
algorithm.

These pages document the Python API (`import navcore`) the way you actually use
it while **building an algorithm**: each function group has a concept section, the
conventions that bite, the full signature list, and **worked numerical examples
you can check by hand**. Every number on these pages was produced by running the
shipped `navcore` and matches the unit tests in
[`tests/python/test_bindings.py`](../tests/python/test_bindings.py) — so the docs
and the code cannot silently drift.

> **How to read the worked examples.** Each one is a runnable snippet followed by
> the value it returns. Paste it into a REPL with `navcore` installed and you
> should see the same number. Where a result is also pinned by a unit test, that
> is noted — those are the contract guarantees you can lean on.

## The pages

| Page | Functions covered | Use it when you are… |
|------|-------------------|----------------------|
| [Rotations](rotations.md) | `quaternion_to_dcm`, `dcm_to_quaternion`, `euler_zyx_to_quaternion`, `quaternion_to_euler_zyx`, `rotate_vector`, `Quaternion` | moving a vector between body and world, composing attitudes, initialising attitude |
| [Frame transforms](frames.md) | `llh_to_ecef`, `ecef_to_llh`, `llh_to_ned`, `ned_to_enu`, `enu_to_ned`, `apply_lever_arm` | turning a GNSS LLH fix into a local NED measurement, handling an antenna lever arm |
| [GNSS error budget](gnss-error.md) | `compute_uere`, `dop_to_position_sigma`, `pdop_from_hdop_vdop`, `gnss_position_covariance_ned`, `epu_to_nacp`, `nacp_to_epu_threshold`, `epu_95_to_sigma_h`, `timing_to_range_sigma`, `timing_to_tdoa_range_sigma` | deciding the GNSS measurement-noise `R` for your filter, reasoning about integrity / NACp |
| [Ranging geometry](ranging.md) | `range_m`, `range_diff_m`, `los_unit`, `bearing_from_ned`, `range_dop` | fusing UWB / acoustic / local-beacon range or bearing measurements |
| [The ESKF](eskf.md) | `Eskf`, `NominalState`, `predict`, `update_gnss_position`, `update` | building an INS/GNSS (or INS/anything) estimator on the 15-state filter engine |

## The conventions that bite (read once)

These are correctness requirements, not preferences. Every page restates the ones
relevant to it, but the whole toolkit holds to them:

- **Quaternions are `[w, x, y, z]`, body-from-world.** Not `[x, y, z, w]` (the
  SciPy order). A reversed quaternion is the single most common integration bug
  — see [Rotations](rotations.md).
- **Frames: world = NED** (North-East-Down), **body = FRD** (Forward-Right-Down).
- **Vectors are `(3,)` `[x, y, z]` float64**; matrices are row-major float64.
- **IMU specific force is physical: `f = a − g`.** At level rest an accelerometer
  reads `[0, 0, −9.80665]` ("1 g up" in FRD). Feed the accelerometer **raw** to
  `Eskf.predict` — see [The ESKF](eskf.md).
- **The 15-state error order is fixed:** `δp(0:3), δv(3:6), δψ(6:9), δb_g(9:12),
  δb_a(12:15)`. nav-eval reads the covariance positionally, so this cannot change.

## Where this sits

```
nav-core (this library) ──▶ your algorithm ──▶ AlgorithmOutput ──▶ nav-eval
        math primitives        you compose them      the contract       scoring
```

nav-core has no toolkit dependencies — it is the bottom of the stack. The C++
headers are dependency-free; one pybind11 file (`src/bindings.cpp`) exposes
everything to numpy. For the API at a glance see the
[README quick reference](../README.md#quick-reference); for the end-to-end
pipeline these primitives feed into, see the
[nav-algorithm-template](https://github.com/UAGV/nav-algorithm-template) and its
worked fixed-wing EKF.
