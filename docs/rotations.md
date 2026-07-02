# Rotations — quaternions, DCMs, Euler angles

Attitude is the part of a navigation state that trips people up most, because
every library picks slightly different conventions. nav-core fixes one set and
holds to it everywhere. This page is the reference for the rotation primitives and
— just as important — the conventions you must get right when you build an
algorithm on them.

`from navcore import ( quaternion_to_dcm, dcm_to_quaternion,
euler_zyx_to_quaternion, quaternion_to_euler_zyx, rotate_vector, Quaternion )`

---

## The conventions that bite

| Convention | nav-core's choice | The trap it avoids |
|------------|-------------------|--------------------|
| Quaternion element order | `[w, x, y, z]` | SciPy / ROS often use `[x, y, z, w]`. Feeding a `[x,y,z,w]` array where nav-core expects `[w,x,y,z]` silently rotates everything wrong. |
| Rotation sense | **body-from-world**: `q` takes a world vector *into* the body frame | If you store world-from-body you must conjugate before every use. |
| World / body frames | world = **NED**, body = **FRD** | Mixing ENU/NED or FLU/FRD flips signs on the down/right axes. |
| Euler sequence | **ZYX** (yaw → pitch → roll), returned as `[roll_rad, pitch_rad, yaw_rad]` | A different sequence gives different angles for the same attitude. |

The single operative definition to remember:

```
v_body = quaternion_to_dcm(q) @ v_world = rotate_vector(q, v_world)
```

`rotate_vector(q, v)` and `quaternion_to_dcm(q) @ v` return the same thing to
machine precision (verified below) — use whichever your maths reads more clearly.

---

## API

| Function | Signature → returns | Notes |
|----------|---------------------|-------|
| `quaternion_to_dcm` | `(q_wxyz: (4,)) → (3,3)` | body-from-world rotation matrix (row-major) |
| `dcm_to_quaternion` | `(R: (3,3)) → (4,)` | inverse of the above (Shepperd's method, stable) |
| `euler_zyx_to_quaternion` | `(roll_rad, pitch_rad, yaw_rad) → (4,)` | three scalars in, quaternion out |
| `quaternion_to_euler_zyx` | `(q_wxyz: (4,)) → (3,)` | returns `[roll_rad, pitch_rad, yaw_rad]` |
| `rotate_vector` | `(q_wxyz: (4,), v_xyz: (3,)) → (3,)` | rotates world → body |
| `Quaternion(w,x,y,z)` | class | C++-native form: `.norm()`, `.normalised()`, `.conjugate()`, `*` (Hamilton product). Most Python code just uses `(4,)` numpy arrays. |

All arrays are `float64`.

---

## Worked examples

### 1. Identity — the do-nothing rotation

```python
import numpy as np, navcore
q_identity = np.array([1.0, 0.0, 0.0, 0.0])
navcore.quaternion_to_dcm(q_identity)
# → [[1, 0, 0],
#    [0, 1, 0],
#    [0, 0, 1]]
navcore.rotate_vector(q_identity, np.array([3.0, -1.5, 7.2]))
# → [3.0, -1.5, 7.2]   (unchanged)
```

Pinned by `test_quaternion_to_dcm_identity` and `test_rotate_vector_identity`.

### 2. A 90° yaw — building a quaternion and using it three ways

A vehicle yawed 90° about the down axis:

```python
import math, numpy as np, navcore
q_yaw90 = navcore.euler_zyx_to_quaternion(roll_rad=0.0, pitch_rad=0.0, yaw_rad=math.pi/2)
# → [0.707107, 0.0, 0.0, 0.707107]      (w = x-comp = cos45° = 0.70710678)

navcore.quaternion_to_dcm(q_yaw90)
# → [[ 0., -1.,  0.],
#    [ 1.,  0.,  0.],
#    [ 0.,  0.,  1.]]

navcore.rotate_vector(q_yaw90, np.array([1.0, 0.0, 0.0]))
# → [0., 1., 0.]
```

The last line is pinned by `test_rotate_vector_90deg_yaw`: a world-North unit
vector `[1,0,0]` maps to `[0,1,0]` in body axes after a 90° yaw. Notice the DCM
and `rotate_vector` agree — the first column of the DCM *is* the image of
`[1,0,0]`.

> **Known convention caveat (tracked).** The C++ test
> `test_quaternion_to_dcm_90deg_yaw` asserts `R[0,1] == +1`, but the
> implementation returns `R[0,1] == −1` (above). This is an open question about
> the literal DCM sign convention, logged in
> [`nav-core-build-brief.md`](../../nav-data/EcosystemDesignDocs/nav-core-build-brief.md)
> and roadmap item **NAV-002** — *not* a bug that affects algorithms. Rotations
> are applied self-consistently around the toolkit: nav-sim generates
> measurements and your filter consumes them through the *same* `navcore`
> functions, so any sign convention cancels in the loop. When you need an
> unambiguous, test-guaranteed operation, use `rotate_vector`.

### 3. Composing rotations — two 90° yaws make a 180° yaw

The `Quaternion` class's `*` is the Hamilton product, which chains rotations:

```python
import math, navcore
c = math.cos(math.pi/4)                       # 0.70710678
q_yaw90 = navcore.Quaternion(c, 0.0, 0.0, c)
q_yaw180 = q_yaw90 * q_yaw90
# → Quaternion(w=0.0, x=0.0, y=0.0, z=1.0)    a pure 180° rotation about z
```

`[0,0,0,1]` is the canonical 180°-about-down quaternion. This is how you compose a
mounting rotation with a measured attitude, or accumulate a delta-rotation.

### 4. Euler round-trip — the form humans read

Euler angles are convenient for configuration files and reports; convert to a
quaternion for any actual maths.

```python
import navcore
q = navcore.euler_zyx_to_quaternion(0.3, -0.2, 1.1)
# → [0.830942, 0.178359, -0.006436, 0.526955]
navcore.quaternion_to_euler_zyx(q)
# → [0.3, -0.2, 1.1]      (roll, pitch, yaw — recovered exactly)
```

Pinned by `test_euler_zyx_round_trip`. (Near pitch = ±90° the ZYX
parameterisation is gimbal-locked: roll is set to 0 and folded into yaw. Work in
quaternions through any steep-pitch manoeuvre.)

### 5. Gravity in the body frame — the basis of accelerometer levelling

This is the rotation you reach for when initialising attitude. The NED gravity
vector is `[0, 0, 9.80665]` (down-positive). For a vehicle at 10° of pitch:

```python
import math, numpy as np, navcore
q = navcore.euler_zyx_to_quaternion(0.0, math.radians(10.0), 0.0)
g_ned = np.array([0.0, 0.0, 9.80665])
navcore.rotate_vector(q, g_ned)
# → [1.702907, 0.0, 9.657665]
```

Check the magnitudes by hand: `9.80665·sin10° = 1.7029` along forward,
`9.80665·cos10° = 9.6577` along down, and the rotated vector still has length
`9.80665`. **Why this matters for algorithm development:** at rest the
accelerometer measures the specific force `f = a − g = −g`, so the *direction* of
the measured acceleration in body axes fixes roll and pitch. A coarse alignment
inverts exactly this relationship to seed the ESKF's initial
`q_body_from_ned` (the shipped template's EKF does this — see its
`example-algorithm` branch). Heading then comes from a magnetometer, because
gravity alone leaves yaw unobservable.

---

## Where to take this next

- **Feed an initial attitude into the filter.** A quaternion built here becomes
  `NominalState.q_body_from_ned` — see [The ESKF](eskf.md).
- **Rotate a measurement model.** A magnetometer or sun-sensor update needs the
  reference vector rotated into body axes with `rotate_vector`; the resulting
  Jacobian goes into the ESKF generic `update`.
- **Lever arms and frames.** Position measurements that involve attitude (a GNSS
  antenna offset from the IMU) combine these rotations with the frame transforms
  in [Frame transforms](frames.md).
- **Gaps you might fill.** nav-core has no `slerp`, no quaternion small-angle
  `exp`/`log` map, and no `ned_to_llh`; these are candidate additions tracked
  under roadmap **NAV-015** if your algorithm needs them.
