# Frame transforms — LLH, ECEF, NED/ENU, lever arms

A GNSS receiver reports geodetic latitude/longitude/height. Your filter runs in a
local Cartesian frame. The functions here bridge the two, and handle the antenna
lever arm. The one you will call most is `llh_to_ned` — it is in the inner loop of
every position filter in the toolkit.

`from navcore import ( llh_to_ecef, ecef_to_llh, llh_to_ned, ned_to_enu,
enu_to_ned, apply_lever_arm )`

---

## The frames, and the conventions that bite

| Frame | Meaning | nav-core representation |
|-------|---------|-------------------------|
| **LLH** | geodetic latitude, longitude, height | `[latitude_deg, longitude_deg, height_m]` — **degrees**, ellipsoidal height in metres |
| **ECEF** | Earth-Centred Earth-Fixed | `[X, Y, Z]` metres |
| **NED** | local tangent plane, origin at a reference LLH | `[North, East, Down]` metres |
| **ENU** | local tangent plane | `[East, North, Up]` metres |

- The ellipsoid is **WGS-84**: `WGS84_A = 6378137.0` m (semi-major axis),
  `WGS84_F = 1/298.257223563` (flattening), `WGS84_E2 = 2f − f²` (eccentricity²).
  Exposed as module constants.
- `llh_to_ned(pos_llh, ref_llh)` returns the NED **offset of `pos` from `ref`** —
  it goes through ECEF, so it is exact at any baseline (no flat-Earth assumption).
- **There is no `ned_to_llh` in nav-core.** The inverse is composed where needed;
  nav-sim provides `navsim.sensors.ned_to_llh`, verified as an exact round-trip
  against `llh_to_ned`. (Adding `ned_to_llh` to nav-core is roadmap **NAV-015**.)

---

## API

| Function | Signature → returns | Notes |
|----------|---------------------|-------|
| `llh_to_ecef` | `(latitude_deg, longitude_deg, height_m) → [X,Y,Z]` m | closed-form |
| `ecef_to_llh` | `(ecef_x_m, ecef_y_m, ecef_z_m) → [lat_deg, lon_deg, h_m]` | Bowring iteration, sub-mm |
| `llh_to_ned` | `(pos_llh: (3,), ref_llh: (3,)) → [N,E,D]` m | the GNSS→local-offset workhorse |
| `ned_to_enu` | `(ned_xyz: (3,)) → [E,N,U]` | for plotting / ENU consumers |
| `enu_to_ned` | `(enu_xyz: (3,)) → [N,E,D]` | inverse of the above |
| `apply_lever_arm` | `(p_antenna_world_m: (3,), q_body_from_world: (4,), lever_arm_body_m: (3,)) → (3,)` | removes a body-frame antenna offset |

---

## Worked examples

### 1. LLH → ECEF at two reference points you can check

```python
import navcore
navcore.llh_to_ecef(0.0, 0.0, 0.0)
# → [6378137.0, 0.0, 0.0]      X = semi-major axis a, on the equator/prime-meridian

navcore.llh_to_ecef(90.0, 0.0, 0.0)
# → [0.0, 0.0, 6356752.314245]  Z = semi-minor axis b = a(1−f) at the pole
```

The equator case is pinned by `test_llh_to_ecef_equator` (`X == WGS84_A`). The
pole case returns the semi-minor axis `b = a(1−f) = 6356752.31 m`, which is the
flattening of the Earth made concrete.

### 2. ECEF → LLH round-trips exactly

```python
import navcore
ecef = navcore.llh_to_ecef(51.5, -0.12, 50.0)
# → [3978670.930179, -8332.921094, 4968401.587699]
navcore.ecef_to_llh(ecef[0], ecef[1], ecef[2])
# → [51.5, -0.12, 50.0]        recovered to 1e-9° / 1e-3 m
```

Pinned by `test_ecef_llh_round_trip`. Bowring's iteration converges in 3–4 steps
at any altitude up to LEO.

### 3. LLH → NED — turning a GNSS fix into a filter measurement

This is the call your update step makes on every fix. Reference origin
`(51.5°, −0.12°, 50 m)`; a fix one milli-degree north, one milli-degree east, and
10 m higher:

```python
import numpy as np, navcore
ref = np.array([51.5, -0.12, 50.0])
fix = np.array([51.501, -0.119, 60.0])
navcore.llh_to_ned(fix, ref)
# → [111.259365, 69.439649, -9.998652]
```

Read it off: ~111 m North (a milli-degree of latitude ≈ 111 m), ~69 m East (a
milli-degree of longitude is shorter at this latitude, ×cos 51.5°), and Down ≈
−10 m (10 m *higher* than the reference, and Down is negative-up). Identity check,
pinned by `test_llh_to_ned_same_point_is_zero`:

```python
navcore.llh_to_ned(ref, ref)        # → [0.0, 0.0, 0.0]
```

### 4. NED ↔ ENU — for plotting and ENU-native tools

```python
import numpy as np, navcore
navcore.ned_to_enu(np.array([1.0, 2.0, -3.0]))   # → [2.0, 1.0, 3.0]
navcore.enu_to_ned(np.array([2.0, 1.0, 3.0]))     # → [1.0, 2.0, -3.0]
```

Swap the first two axes and flip the third. Pinned by `test_ned_enu_round_trip`.

### 5. Lever arm — when the GNSS antenna is not at the IMU

A GNSS antenna sits some fixed offset from the navigation reference point (the
IMU). `apply_lever_arm` removes that offset, expressed in the body frame, so the
position you feed the filter is the IMU's, not the antenna's:

```
p_reference_world = p_antenna_world − R_world_from_body · lever_arm_body
```

**Identity attitude**, antenna 0.5 m forward of the IMU:

```python
import numpy as np, navcore
q_identity = np.array([1.0, 0.0, 0.0, 0.0])
navcore.apply_lever_arm(np.array([10.0, 5.0, 2.0]), q_identity, np.array([0.5, 0.0, 0.0]))
# → [9.5, 5.0, 2.0]      the forward offset subtracts straight off North
```

Pinned by `test_apply_lever_arm_identity_attitude`. Now **yaw the body 90°** (nose
to the East) with the same forward lever arm — the offset should now act along
East, not North:

```python
import math, numpy as np, navcore
q_yaw90 = navcore.euler_zyx_to_quaternion(0.0, 0.0, math.pi/2)
navcore.apply_lever_arm(np.array([10.0, 5.0, 2.0]), q_yaw90, np.array([0.5, 0.0, 0.0]))
# → [10.0, 5.5, 2.0]      North untouched; +0.5 m shifted onto the East axis
```

The forward lever arm rotated with the body, so it now resolves onto East — and
because we *subtract* it, the reference point is at East 5.0 while the antenna
reading was effectively 5.5. This attitude-dependence is exactly why a lever arm
must be applied with the current attitude, not as a constant offset.

---

## Where to take this next

- **Wire it into the ESKF.** The canonical loop is
  `position_ned = llh_to_ned(gnss_fix_llh, reference_llh)` →
  `eskf.update_gnss_position(position_ned, R_diag)`. See [The ESKF](eskf.md).
- **Set the measurement noise honestly.** The `R` you pass alongside the NED
  position comes from the GNSS error budget — turn HDOP/VDOP and a UERE into a
  covariance with [GNSS error budget](gnss-error.md).
- **Pick the reference origin once.** Use the dataset's
  `meta["reference_llh_origin"]` so your NED frame matches the one nav-sim/nav-eval
  use; otherwise your trajectory will be offset from the truth.
