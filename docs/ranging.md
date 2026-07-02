# Ranging geometry — range, TDOA, line-of-sight, bearing, range-DOP

Where [GNSS error budget](gnss-error.md) turns *timing* noise into *range* noise,
this page provides the *geometry* that range-type sensors share: the distance
between two points, the time-difference of two distances (TDOA), the
line-of-sight unit vector (which doubles as the range measurement Jacobian), the
azimuth/elevation bearing of a displacement, and a position dilution-of-precision
for an arbitrary local-beacon constellation (the UWB / acoustic analogue of GNSS
DOP).

These are the primitives behind nav-sim's UWB, TDOA, rangefinder, and
range/bearing sensor models — and the ones you reuse when you write the matching
measurement update in your filter.

`from navcore import ( range_m, range_diff_m, los_unit, bearing_from_ned,
range_dop )`

---

## Conventions that bite

- All positions and displacements are in a **common world frame**; the bearing
  and DOP helpers assume that frame is **NED**.
- **Azimuth** is measured from North toward East, in `[−π, π]`.
- **Elevation** is positive **upward**, in `[−π/2, π/2]` (note: Down is
  negative-up, so elevation uses `−D`).
- `range_dop` needs **at least 3 beacons** with non-degenerate geometry, or it
  raises / returns infinite DOP.

---

## API

| Function | Signature → returns | Notes |
|----------|---------------------|-------|
| `range_m` | `(a: (3,), b: (3,)) → float` | Euclidean distance |
| `range_diff_m` | `(user, anchor_a, anchor_b) → float` | TDOA: `range(user,a) − range(user,b)` |
| `los_unit` | `(user, target) → (3,)` | unit line-of-sight; **range Jacobian = −los_unit** |
| `bearing_from_ned` | `(delta_ned: (3,)) → (azimuth_rad, elevation_rad)` | returns a tuple |
| `range_dop` | `(beacons: list[(3,)], user: (3,)) → {hdop, vdop, pdop, gdop}` | dict; no clock state, so `gdop == pdop` |

---

## Worked examples

### 1. Range and the 3-4-5 triangle

```python
import numpy as np, navcore
navcore.range_m(np.array([0.0, 0.0, 0.0]), np.array([3.0, 4.0, 0.0]))
# → 5.0       √(3² + 4²)
```

Pinned by `test_range_m`.

### 2. TDOA range difference

A user at the origin, anchor A 5 m away, anchor B 12 m away:

```python
import numpy as np, navcore
navcore.range_diff_m(np.array([0.0, 0.0, 0.0]),
                     np.array([5.0, 0.0, 0.0]),
                     np.array([0.0, 12.0, 0.0]))
# → -7.0      5 − 12; negative because anchor A is the nearer anchor
```

Pinned by `test_range_diff_tdoa`. A real TDOA sensor measures `Δt` between two
anchors; multiply by `c` (see `timing_to_tdoa_range_sigma` for its noise) to get
this geometric quantity.

### 3. Line-of-sight — and why it is the Jacobian

```python
import numpy as np, navcore
navcore.los_unit(np.array([0.0, 0.0, 0.0]), np.array([3.0, 4.0, 0.0]))
# → [0.6, 0.8, 0.0]      the unit vector user → target
```

Pinned by `test_los_unit`. The reason this function matters for filtering: for a
range measurement `h(user) = ‖target − user‖`, the partial derivative with
respect to the user position is

```
∂h/∂user = −los_unit(user, target)
```

So a range update's measurement Jacobian `H` is built directly from `los_unit` —
no separate derivative to derive. See the worked range-update example in
[The ESKF](eskf.md#example-5-generic-update).

### 4. Bearing of a displacement

```python
import math, numpy as np, navcore
az, el = navcore.bearing_from_ned(np.array([1.0, 1.0, 0.0]))
math.degrees(az), math.degrees(el)
# → (45.0, 0.0)      due north-east, level

az, el = navcore.bearing_from_ned(np.array([0.0, 0.0, -1.0]))
math.degrees(az), math.degrees(el)
# → (0.0, 90.0)      straight up (D = −1 is one metre up)
```

Pinned by `test_bearing_from_ned`. Azimuth `= atan2(E, N)`, elevation
`= atan2(−D, √(N²+E²))`.

### 5. Range-DOP for a local beacon constellation

The UWB/acoustic analogue of GNSS DOP. Three beacons placed so their unit
line-of-sight vectors point along +N, +E, +D (geometry matrix `G = I₃`):

```python
import numpy as np, navcore
beacons = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
navcore.range_dop(beacons, np.array([0.0, 0.0, 0.0]))
# → {'hdop': 1.4142, 'vdop': 1.0, 'pdop': 1.7321, 'gdop': 1.7321}
#     HDOP = √2, VDOP = 1, PDOP = GDOP = √3
```

Pinned by `test_range_dop_orthogonal`. The maths: `range_dop` builds the normal
matrix `M = GᵀG = Σ uᵢuᵢᵀ`, inverts it to `Q = M⁻¹`, then reads
`HDOP = √(Q_NN + Q_EE)`, `VDOP = √Q_DD`, `PDOP = √(trace Q)`. With orthogonal
unit LOS, `M = I₃`, so `Q = I₃`. Fewer than three beacons raises (pinned by
`test_range_dop_too_few_beacons_raises`); collinear geometry returns `inf`.

Use this to **lay out beacons before a flight**: try candidate anchor positions,
compute the DOP over your operating volume, and place anchors where the DOP stays
low — exactly how you would reason about a GNSS constellation, but for your own
infrastructure.

---

## Where to take this next

- **Build a range measurement update.** Combine `los_unit` (the Jacobian) with
  the ESKF generic `update` to fuse a UWB two-way range — see
  [The ESKF](eskf.md#example-5-generic-update).
- **Generate matching test data.** nav-sim's `UwbSensor`, `TdoaSensor`,
  `Rangefinder`, and `RangeBearingSensor` form their true measurements from these
  same functions, so a filter you build here can be validated on simulated data
  with known truth.
- **Noise model.** Pair the geometry here with the timing-to-range noise in
  [GNSS error budget](gnss-error.md) to set the measurement `R`.
