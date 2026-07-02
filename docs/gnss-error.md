# GNSS error budget — UERE, DOP, NACp/EPU, timing-to-range

This is the PNT-domain heart of nav-core: the maths that turns a GNSS receiver's
geometry and ranging noise into a position uncertainty your filter can use, and
into the integrity language (NACp/EPU) aviation is written in. Every formula here
follows the RTCA DO-229 / ICAO Annex 10 framework.

When you build an estimator, these functions answer one practical question:
**what measurement-noise covariance `R` should I trust this GNSS fix with?**

`from navcore import ( compute_uere, dop_to_position_sigma, pdop_from_hdop_vdop,
gnss_position_covariance_ned, epu_to_nacp, nacp_to_epu_threshold,
epu_95_to_sigma_h, timing_to_range_sigma, timing_to_tdoa_range_sigma )`

---

## The error chain (notation defined before use)

The quantities, in the order they compose:

- **σ_UERE** — *User Equivalent Range Error*: the 1-sigma ranging error to one
  satellite, in metres, formed from independent error sources (clock, ephemeris,
  ionosphere, troposphere, multipath, receiver noise) added in quadrature.
- **DOP** — *Dilution of Precision*: a purely geometric, unitless multiplier from
  range error to position error. `HDOP` → horizontal, `VDOP` → vertical, `PDOP` →
  3-D, with `PDOP² = HDOP² + VDOP²`.
- **σ_pos** — the position 1-sigma: `σ_pos = DOP · σ_UERE`. This is the key
  formula; everything else is a special case or a packaging of it.
- **EPU (95%)** — *Estimated Position Uncertainty*, the 95th-percentile horizontal
  error. **NACp** is the DO-260B 4-bit integrity category that EPU maps to.

---

## API

| Function | Signature → returns | Formula |
|----------|---------------------|---------|
| `compute_uere` | `(σ_clock, σ_eph, σ_iono, σ_tropo, σ_multipath, σ_noise) → σ_UERE` m | quadrature sum |
| `dop_to_position_sigma` | `(dop, sigma_uere_m) → σ_pos` m | `dop · σ_UERE` |
| `pdop_from_hdop_vdop` | `(hdop, vdop) → pdop` | `√(hdop² + vdop²)` |
| `gnss_position_covariance_ned` | `(hdop, vdop, sigma_uere_m) → (3,3)` m² | `diag(σ_H², σ_H², σ_V²)` |
| `epu_to_nacp` | `(epu_95_m) → int` | DO-260B Table 2-23 lookup |
| `nacp_to_epu_threshold` | `(nacp) → epu_upper_bound_m` | inverse lookup |
| `epu_95_to_sigma_h` | `(epu_95_m, use_two_sigma_approximation=True) → σ_H` m | `/2` or `/2.4477` |
| `timing_to_range_sigma` | `(sigma_time_s) → σ_r` m | `c · σ_t` |
| `timing_to_tdoa_range_sigma` | `(sigma_time_s) → σ_r` m | `√2 · c · σ_t` |

`SPEED_OF_LIGHT_M_PER_S = 299792458.0` is exposed as a constant.

---

## Worked examples

### 1. Composing a UERE

Two equal 1 m sources add in quadrature to √2 (pinned by
`test_uere_two_equal_components`):

```python
import navcore
navcore.compute_uere(0.0, 0.0, 1.0, 0.0, 0.0, 1.0)   # iono = noise = 1 m
# → 1.4142135624        = √2
```

A realistic open-sky budget — clock 0.4, ephemeris 0.8, iono 1.0, tropo 0.2,
multipath 0.5, receiver noise 0.3 m:

```python
navcore.compute_uere(0.4, 0.8, 1.0, 0.2, 0.5, 0.3)
# → 1.4765 m            √(0.16+0.64+1.0+0.04+0.25+0.09)
```

The ionosphere dominates — which is why dual-frequency / SBAS corrections, which
attack the iono term, are what move you between NACp levels.

### 2. DOP → position sigma

```python
import navcore
navcore.dop_to_position_sigma(dop=1.5, sigma_uere_m=2.0)   # → 3.0 m
navcore.pdop_from_hdop_vdop(1.0, 2.0)                       # → 2.2361  (= √5)
```

Pinned by `test_dop_to_position_sigma` and `test_pdop_from_hdop_vdop`.

### 3. The full open-sky budget → a NACp

This is the chain end to end. Typical open-sky geometry HDOP = 1.2, VDOP = 2.1,
σ_UERE = 1.5 m:

```python
import navcore
sigma_h = navcore.dop_to_position_sigma(1.2, 1.5)   # → 1.80 m  (1-sigma horizontal)
sigma_v = navcore.dop_to_position_sigma(2.1, 1.5)   # → 3.15 m  (1-sigma vertical)
epu_95  = 2.0 * sigma_h                               # → 3.60 m  (DO-260B 2σ approximation)
navcore.epu_to_nacp(epu_95)                           # → 10
```

So this geometry supports **NACp = 10** (EPU < 10 m, typical SBAS-corrected
single-frequency). The lookup thresholds, pinned by `test_epu_to_nacp`:

```python
navcore.epu_to_nacp(2.0)    # → 11   (EPU < 3 m   — precise GNSS)
navcore.epu_to_nacp(3.0)    # → 10   (threshold is strict <, so exactly 3 m is NACp 10)
navcore.epu_to_nacp(8.0)    # → 10
navcore.epu_to_nacp(25.0)   # → 9    (EPU < 30 m)
navcore.nacp_to_epu_threshold(11)   # → 3.0   (the worst EPU still consistent with NACp 11)
```

### 4. EPU → 1-sigma (the number your filter actually wants)

A receiver may report a 95% EPU; your filter needs a 1-sigma. Two conventions:

```python
import navcore
navcore.epu_95_to_sigma_h(10.0)                                  # → 5.0    (DO-260B 2σ)
navcore.epu_95_to_sigma_h(10.0, use_two_sigma_approximation=False)
# → 4.0855   (Rayleigh 95th percentile: divide by 2.4477, the statistically correct factor)
```

The 2σ form is the DO-260B convention; the Rayleigh form is the exact 95th
percentile of a 2-D circular Gaussian. Use Rayleigh when you want the honest
1-sigma; use 2σ when you must match avionics paperwork.

### 5. Timing → range

The fundamental GNSS identity — a timing error *is* a range error at the speed of
light (~30 cm per nanosecond):

```python
import navcore
navcore.timing_to_range_sigma(1e-9)    # → 0.2998 m   (1 ns ≈ 30 cm)
navcore.timing_to_range_sigma(10e-9)   # → 2.9979 m
navcore.timing_to_tdoa_range_sigma(10e-9)
# → 4.2397 m    = √2 × 2.9979 (TDOA combines two independent clocks)
```

Pinned by `test_timing_to_range_sigma` and
`test_timing_to_tdoa_range_sigma_is_sqrt2_times_one_way`. The √2 factor is the
bridge to [Ranging geometry](ranging.md), where TDOA geometry is handled.

### 6. Straight to a covariance matrix — the one you feed the filter

`gnss_position_covariance_ned` packages HDOP/VDOP/σ_UERE into the NED position
covariance, assuming isotropic horizontal error uncorrelated with vertical:

```python
import navcore
navcore.gnss_position_covariance_ned(hdop=1.0, vdop=2.0, sigma_uere_m=1.5)
# → [[2.25, 0.0, 0.0],     σ_H = 1.0·1.5 = 1.5  → σ_H² = 2.25
#    [0.0, 2.25, 0.0],
#    [0.0, 0.0, 9.0]]       σ_V = 2.0·1.5 = 3.0  → σ_V² = 9.0
```

Pinned by `test_gnss_position_covariance_ned_shape_and_values`. The diagonal of
this matrix is exactly the `position_cov_diagonal_m2` argument to
`Eskf.update_gnss_position` — this is the function that closes the loop from
"receiver reported HDOP=1.0" to "filter measurement noise".

> **A subtle bias to know.** A receiver's reported `h_accuracy_m` is a *2-D radial*
> figure, so each of the North and East axes carries **half** the variance:
> `σ_axis² = h_accuracy² / 2`. Using the radial value per-axis double-counts the
> noise and makes the filter over-conservative. This exact issue is documented in
> the template EKF's integration findings.

---

## Where to take this next

- **Set `R` for `update_gnss_position`.** Take the diagonal of
  `gnss_position_covariance_ned(...)` (or build it from `epu_95_to_sigma_h`) and
  pass it straight in — see [The ESKF](eskf.md).
- **Reason about integrity.** `epu_to_nacp` lets your algorithm report a NACp
  alongside its estimate; whether your *reported* covariance honestly bounds the
  *true* error is the NEES/ANEES question that nav-eval scores — the same
  overbounding question, stated as a filter metric.
- **Range-based positioning.** For UWB/acoustic ranging rather than satellites,
  the geometric DOP analogue is `range_dop` in [Ranging geometry](ranging.md).
