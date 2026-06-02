/**
 * @file gnss_error.hpp
 * @brief GNSS error-budget math: UERE, DOP, NACp/EPU, timing-to-range.
 *
 * This is the author's primary PNT domain.  Every formula is grounded in the
 * RTCA DO-229 / ICAO Annex 10 performance framework used in aviation.
 *
 * ─── The error chain ────────────────────────────────────────────────────────
 *
 *   σ_UERE   — User Equivalent Range Error: 1-sigma ranging error per
 *              satellite, composed from all uncorrelated sources:
 *
 *     σ²_UERE = σ²_clock + σ²_ephemeris + σ²_ionosphere + σ²_troposphere
 *             + σ²_multipath + σ²_noise
 *
 *   DOP      — Dilution of Precision: purely geometric, a scalar multiplier
 *              that maps range noise to position noise.
 *
 *   σ_pos = DOP · σ_UERE    [this is the key formula]
 *
 *   Specific DOPs:
 *     HDOP → horizontal position (σ_H = HDOP · σ_UERE)
 *     VDOP → vertical position   (σ_V = VDOP · σ_UERE)
 *     PDOP → 3-D position        (σ_P = PDOP · σ_UERE);  PDOP² = HDOP² + VDOP²
 *     TDOP → time/clock          (σ_t = TDOP · σ_UERE / c)
 *
 * ─── NACp / EPU relationship ────────────────────────────────────────────────
 *
 *   NACp (Navigation Accuracy Category — Position, DO-260B Table 2-23) is a
 *   4-bit field encoding Estimated Position Uncertainty (EPU, 95% horizontal):
 *
 *     NACp=11 → EPU < 3 m    (precise GNSS with SBAS / dual-freq)
 *     NACp=10 → EPU < 10 m   (typical SBAS-corrected single-freq)
 *     NACp= 9 → EPU < 30 m   (autonomous C/A, benign conditions)
 *     NACp= 8 → EPU < 0.1 NM  ≈ 185 m
 *     NACp= 7 → EPU < 0.3 NM
 *     …
 *
 *   EPU (95%) ≈ 2·σ_H (95% ≈ 2σ for 2-D Gaussian) so:
 *     σ_H = EPU_95 / 2  (the 1-sigma figure to feed into a filter).
 *
 *   For a 2-D circular Gaussian of 1-sigma radius σ, the 95th percentile radius
 *   is approximately 2.45·σ (Rayleigh distribution), not 2σ. DO-260B uses the
 *   2-sigma approximation for simplicity; this header provides both.
 *
 * ─── Timing to range ────────────────────────────────────────────────────────
 *
 *   Range noise from a timing noise σ_t:
 *     σ_r = c · σ_t                         (one-way range)
 *     σ_r_tdoa = √2 · c · σ_t               (TDOA: two independent clocks)
 *
 *   where c ≈ 299 792 458 m/s.
 *
 *   Worked case — σ_t = 10 ns:
 *     σ_r       = 299 792 458 × 10e-9 ≈ 2.998 m
 *     σ_r_tdoa  = √2 × 2.998          ≈ 4.239 m                         ✓
 *
 * ─── Worked DOP example ─────────────────────────────────────────────────────
 *
 *   Typical open-sky values: HDOP=1.2, VDOP=2.1, σ_UERE=1.5 m.
 *     σ_H = 1.2 × 1.5 = 1.80 m   (1-sigma horizontal)
 *     σ_V = 2.1 × 1.5 = 3.15 m   (1-sigma vertical)
 *     EPU_95 ≈ 2 × 1.80 = 3.60 m  → NACp ≥ 10                          ✓
 */

#pragma once
#include <array>
#include <cmath>
#include <stdexcept>

namespace navcore {

/// Speed of light [m/s]
inline constexpr double SPEED_OF_LIGHT_M_PER_S = 299'792'458.0;

// -------------------------------------------------------------------------- //
// UERE composition                                                           //
// -------------------------------------------------------------------------- //

/**
 * Compose individual 1-sigma range-error contributions into a total UERE.
 *
 * σ_UERE = √(σ_clock² + σ_ephemeris² + σ_iono² + σ_tropo² + σ_multipath² + σ_noise²)
 *
 * Worked case — all zero except σ_iono = 1.0 and σ_noise = 1.0:
 *   UERE = √2 ≈ 1.4142 m.                                                ✓
 *
 * All arguments in metres (1-sigma).  Pass 0.0 for unknown components.
 */
[[nodiscard]] inline double
compute_uere_m(double sigma_clock_m,
               double sigma_ephemeris_m,
               double sigma_iono_m,
               double sigma_tropo_m,
               double sigma_multipath_m,
               double sigma_noise_m) noexcept {
    return std::sqrt(
        sigma_clock_m     * sigma_clock_m     +
        sigma_ephemeris_m * sigma_ephemeris_m +
        sigma_iono_m      * sigma_iono_m      +
        sigma_tropo_m     * sigma_tropo_m     +
        sigma_multipath_m * sigma_multipath_m +
        sigma_noise_m     * sigma_noise_m
    );
}

// -------------------------------------------------------------------------- //
// DOP → position accuracy                                                    //
// -------------------------------------------------------------------------- //

/**
 * Compute position accuracy from a DOP factor and UERE.
 *
 * σ_pos = dop · sigma_uere_m
 *
 * Works for any DOP type (HDOP → σ_H, VDOP → σ_V, PDOP → σ_P, TDOP → σ_t_equiv).
 *
 * Worked case — HDOP=1.5, σ_UERE=2.0: σ_H = 3.0 m.                      ✓
 */
[[nodiscard]] inline constexpr double
dop_to_position_sigma_m(double dop, double sigma_uere_m) noexcept {
    return dop * sigma_uere_m;
}

/**
 * PDOP from HDOP and VDOP.
 *
 * PDOP² = HDOP² + VDOP²
 *
 * Worked case — HDOP=1.0, VDOP=2.0: PDOP = √5 ≈ 2.236.                  ✓
 */
[[nodiscard]] inline double
pdop_from_hdop_vdop(double hdop, double vdop) noexcept {
    return std::sqrt(hdop * hdop + vdop * vdop);
}

// -------------------------------------------------------------------------- //
// NACp ↔ EPU                                                                //
// -------------------------------------------------------------------------- //

/** EPU thresholds per NACp level (metres, DO-260B Table 2-23). */
struct NacpEntry {
    int nacp;
    double epu_threshold_m;
};

/**
 * Minimum NACp value consistent with a given 95% EPU.
 *
 * Implements the DO-260B lookup: find the highest NACp whose EPU threshold
 * is strictly greater than the input EPU.  Returns 0 if EPU ≥ 18.52 km.
 *
 * Worked cases:
 *   epu = 2.0 m  → NACp = 11  (threshold 3 m).                          ✓
 *   epu = 8.0 m  → NACp = 10  (threshold 10 m).                         ✓
 *   epu = 25.0 m → NACp =  9  (threshold 30 m).                         ✓
 *
 * @param epu_95_m  Estimated position uncertainty at 95% confidence [metres].
 */
[[nodiscard]] inline int
epu_to_nacp(double epu_95_m) noexcept {
    // Thresholds from DO-260B Table 2-23 (subset, aviation-relevant).
    // Listed in descending NACp order; first threshold > epu gives the answer.
    static constexpr NacpEntry TABLE[] = {
        {11, 3.0},
        {10, 10.0},
        { 9, 30.0},
        { 8, 185.2},       // 0.1 NM
        { 7, 555.6},       // 0.3 NM
        { 6, 1111.2},      // 0.6 NM
        { 5, 18520.0},     // 10 NM
        { 4, 185200.0},    // 100 NM
    };
    for (const auto& entry : TABLE) {
        if (epu_95_m < entry.epu_threshold_m) return entry.nacp;
    }
    return 0;
}

/**
 * Upper bound on EPU (95%) for a given NACp level [metres].
 *
 * Returns the DO-260B threshold (the worst EPU consistent with that NACp).
 * Returns infinity for NACp ≤ 3 (undefined / very poor).
 *
 * Worked case — NACp=11: 3.0 m.                                          ✓
 */
[[nodiscard]] inline double
nacp_to_epu_threshold_m(int nacp) noexcept {
    switch (nacp) {
        case 11: return 3.0;
        case 10: return 10.0;
        case  9: return 30.0;
        case  8: return 185.2;
        case  7: return 555.6;
        case  6: return 1111.2;
        case  5: return 18520.0;
        case  4: return 185200.0;
        default: return 1e30;  // undefined / unknown
    }
}

/**
 * Convert a 95% EPU to an approximate 1-sigma horizontal position std dev.
 *
 * Two conventions offered:
 *   two_sigma  = true  (DO-260B approximation): σ = EPU_95 / 2
 *   two_sigma  = false (Rayleigh 95th percentile):  σ = EPU_95 / 2.4477
 *
 * Worked case — EPU_95 = 10 m, two_sigma = true:   σ_H ≈ 5.0 m.         ✓
 * Worked case — EPU_95 = 10 m, two_sigma = false:  σ_H ≈ 4.086 m.       ✓
 */
[[nodiscard]] inline double
epu_95_to_sigma_h_m(double epu_95_m, bool use_two_sigma_approximation = true) noexcept {
    if (use_two_sigma_approximation) return epu_95_m / 2.0;
    // Rayleigh: P(r < k·σ) = 1 − exp(−k²/2); for P = 0.95 → k ≈ 2.4477
    return epu_95_m / 2.4477;
}

// -------------------------------------------------------------------------- //
// Timing to range                                                            //
// -------------------------------------------------------------------------- //

/**
 * Convert 1-sigma clock/timing noise to 1-sigma one-way range noise.
 *
 * σ_r = c · σ_t
 *
 * This is the fundamental identity: a timing error of 1 ns corresponds to
 * ~30 cm of range error.  In aviation GNSS the dominant UERE contributors
 * are all ultimately timing errors on the satellite or user side.
 *
 * Worked case — σ_t = 10e-9 s:
 *   σ_r = 299 792 458 × 10e-9 ≈ 2.998 m.                                ✓
 *
 * @param sigma_time_s  1-sigma timing noise [seconds].
 * @returns             1-sigma range noise [metres].
 */
[[nodiscard]] inline constexpr double
timing_to_range_sigma_m(double sigma_time_s) noexcept {
    return SPEED_OF_LIGHT_M_PER_S * sigma_time_s;
}

/**
 * Convert 1-sigma timing noise to 1-sigma TDOA range noise.
 *
 * When the range difference (TDOA) is formed from two independent clocks each
 * with noise σ_t, the combined noise is √2·σ_t, so:
 *
 *   σ_r_tdoa = √2 · c · σ_t
 *
 * Worked case — σ_t = 10 ns:
 *   σ_r = c × 10e-9 ≈ 2.998 m
 *   σ_r_tdoa = √2 × 2.998 ≈ 4.239 m.                                     ✓
 *
 * @param sigma_time_s  1-sigma per-clock timing noise [seconds].
 * @returns             1-sigma TDOA-derived range noise [metres].
 */
[[nodiscard]] inline double
timing_to_tdoa_range_sigma_m(double sigma_time_s) noexcept {
    return std::sqrt(2.0) * SPEED_OF_LIGHT_M_PER_S * sigma_time_s;
}

// -------------------------------------------------------------------------- //
// Covariance matrix from DOP geometry                                       //
// -------------------------------------------------------------------------- //

/**
 * Build a 3×3 NED position covariance matrix from HDOP, VDOP, σ_UERE.
 *
 * Assumes horizontal error is isotropic (equal N and E noise) and uncorrelated
 * with vertical, which is the standard approximation:
 *
 *   P_pos = diag(σ_N², σ_E², σ_D²)
 *   where σ_N = σ_E = HDOP · σ_UERE,  σ_D = VDOP · σ_UERE
 *
 * Worked case — HDOP=1.0, VDOP=2.0, σ_UERE=1.5:
 *   σ_H = 1.5,  σ_D = 3.0
 *   P = diag(2.25, 2.25, 9.0).                                           ✓
 *
 * @returns  3×3 diagonal covariance [m²], row-major.
 */
[[nodiscard]] inline std::array<std::array<double, 3>, 3>
gnss_position_covariance_ned_m2(double hdop, double vdop,
                                 double sigma_uere_m) noexcept {
    const double sigma_h = hdop * sigma_uere_m;
    const double sigma_v = vdop * sigma_uere_m;
    const double sh2 = sigma_h * sigma_h;
    const double sv2 = sigma_v * sigma_v;
    return {{
        {sh2, 0.0, 0.0},
        {0.0, sh2, 0.0},
        {0.0, 0.0, sv2},
    }};
}

} // namespace navcore
