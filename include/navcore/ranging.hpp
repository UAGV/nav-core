/**
 * @file ranging.hpp
 * @brief Range / TDOA / bearing geometry for local-beacon positioning.
 *
 * The complement to gnss_error.hpp. Where gnss_error.hpp turns *timing* noise
 * into *range* noise (σ_r = c·σ_t) and composes the satellite error budget, this
 * header provides the *geometry* that range-type sensors share: Euclidean range
 * between two points, the time-difference (TDOA) of two ranges, the line-of-sight
 * unit vector (which is also the range measurement Jacobian), the azimuth/elevation
 * bearing of a displacement, and a position Dilution-of-Precision for an arbitrary
 * local beacon constellation (the UWB/acoustic analogue of GNSS DOP).
 *
 * These primitives are reused by:
 *   - nav-sim sensor models (UWB two-way ranging, TDOA, rangefinder, range/bearing),
 *   - any estimator forming a range or bearing measurement and its Jacobian.
 *
 * Frame convention: positions and displacements are in a common world frame; the
 * DOP/bearing helpers assume that frame is NED (North, East, Down), matching the
 * toolkit default, so azimuth is measured from North toward East and elevation is
 * positive upward.
 *
 * ─── Worked range example ───────────────────────────────────────────────────
 *   user = [0,0,0], target = [3,4,0]:
 *     range_m = √(3² + 4²) = 5.0 m                                          ✓
 *     los_unit = [0.6, 0.8, 0.0]                                           ✓
 *
 * ─── Worked TDOA example ────────────────────────────────────────────────────
 *   user = [0,0,0], a = [5,0,0], b = [0,12,0]:
 *     range_diff_m = 5 − 12 = −7.0 m                                        ✓
 *
 * ─── Worked bearing example ─────────────────────────────────────────────────
 *   delta_ned = [1,1,0]: azimuth = atan2(1,1) = 45°, elevation = 0°         ✓
 *   delta_ned = [0,0,-1]: azimuth = 0°, elevation = atan2(1,0) = 90° (up)   ✓
 *
 * ─── Worked DOP example ─────────────────────────────────────────────────────
 *   Three beacons on the +N, +E, +D axes (unit LOS = I₃): geometry matrix G = I₃,
 *   so Q = (GᵀG)⁻¹ = I₃ and
 *     HDOP = √(Q_NN + Q_EE) = √2 ≈ 1.414
 *     VDOP = √Q_DD          = 1.0
 *     PDOP = √(trace Q)     = √3 ≈ 1.732                                    ✓
 */

#pragma once
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace navcore {

using Vec3 = std::array<double, 3>;

// -------------------------------------------------------------------------- //
// Range and TDOA                                                             //
// -------------------------------------------------------------------------- //

/**
 * Euclidean range [m] between two points in a common frame.
 *
 * Worked case — a=[0,0,0], b=[3,4,0]: 5.0.                                  ✓
 */
[[nodiscard]] inline double range_m(const Vec3& a, const Vec3& b) noexcept {
    const double dx = a[0] - b[0];
    const double dy = a[1] - b[1];
    const double dz = a[2] - b[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * TDOA range difference [m]: range(user, anchor_a) − range(user, anchor_b).
 *
 * This is the geometric quantity a time-difference-of-arrival measurement
 * observes (after scaling the measured Δt by c). Sign convention: positive when
 * anchor_a is the farther anchor.
 *
 * Worked case — user=[0,0,0], a=[5,0,0], b=[0,12,0]: 5 − 12 = −7.0.         ✓
 */
[[nodiscard]] inline double
range_diff_m(const Vec3& user, const Vec3& anchor_a, const Vec3& anchor_b) noexcept {
    return range_m(user, anchor_a) - range_m(user, anchor_b);
}

/**
 * Unit line-of-sight vector from @p user toward @p target.
 *
 * This is also the range measurement Jacobian: for h(user) = ‖target − user‖,
 * ∂h/∂user = −los_unit(user, target). Returns {0,0,0} for coincident points.
 *
 * Worked case — user=[0,0,0], target=[3,4,0]: [0.6, 0.8, 0.0].             ✓
 */
[[nodiscard]] inline Vec3 los_unit(const Vec3& user, const Vec3& target) noexcept {
    const double r = range_m(user, target);
    if (r <= 0.0) return {0.0, 0.0, 0.0};
    return {(target[0] - user[0]) / r,
            (target[1] - user[1]) / r,
            (target[2] - user[2]) / r};
}

// -------------------------------------------------------------------------- //
// Bearing                                                                    //
// -------------------------------------------------------------------------- //

/** Azimuth/elevation bearing [rad] of a displacement. */
struct Bearing {
    double azimuth_rad;    ///< measured from North toward East, in [−π, π]
    double elevation_rad;  ///< positive upward, in [−π/2, π/2]
};

/**
 * Bearing of an NED displacement [N, E, D].
 *
 *   azimuth   = atan2(E, N)
 *   elevation = atan2(−D, √(N² + E²))      (−D because Down is negative-up)
 *
 * Worked case — [1,1,0]:  azimuth = 45°, elevation = 0°.                    ✓
 * Worked case — [0,0,-1]: azimuth = 0°,  elevation = 90° (straight up).     ✓
 */
[[nodiscard]] inline Bearing bearing_from_ned(const Vec3& delta_ned) noexcept {
    const double n = delta_ned[0];
    const double e = delta_ned[1];
    const double d = delta_ned[2];
    return {std::atan2(e, n), std::atan2(-d, std::hypot(n, e))};
}

// -------------------------------------------------------------------------- //
// Range-DOP for a local beacon constellation                                 //
// -------------------------------------------------------------------------- //

/** Position Dilution of Precision for a ranging geometry (no clock state). */
struct RangeDop {
    double hdop;  ///< horizontal: √(Q_NN + Q_EE)
    double vdop;  ///< vertical:   √Q_DD
    double pdop;  ///< position:   √(trace Q)
    double gdop;  ///< geometric:  equals pdop for clock-free two-way ranging
};

namespace detail {
/// Invert a symmetric 3×3 matrix M (row-major). Returns false if near-singular.
inline bool invert3x3(const std::array<std::array<double, 3>, 3>& m,
                      std::array<std::array<double, 3>, 3>& inv) noexcept {
    const double a = m[0][0], b = m[0][1], c = m[0][2];
    const double d = m[1][0], e = m[1][1], f = m[1][2];
    const double g = m[2][0], h = m[2][1], i = m[2][2];
    const double A =  (e * i - f * h);
    const double B = -(d * i - f * g);
    const double C =  (d * h - e * g);
    const double det = a * A + b * B + c * C;
    if (std::abs(det) < 1e-12) return false;
    const double inv_det = 1.0 / det;
    inv[0][0] = A * inv_det;
    inv[0][1] = -(b * i - c * h) * inv_det;
    inv[0][2] =  (b * f - c * e) * inv_det;
    inv[1][0] = B * inv_det;
    inv[1][1] =  (a * i - c * g) * inv_det;
    inv[1][2] = -(a * f - c * d) * inv_det;
    inv[2][0] = C * inv_det;
    inv[2][1] = -(a * h - b * g) * inv_det;
    inv[2][2] =  (a * e - b * d) * inv_det;
    return true;
}
}  // namespace detail

/**
 * Position DOP for ranging to a set of @p beacons from @p user (all in NED).
 *
 * Builds the geometry matrix G whose rows are the unit line-of-sight vectors to
 * each beacon, forms the normal matrix M = GᵀG (= Σ uᵢ uᵢᵀ), inverts it, and
 * reads the DOPs off the diagonal of Q = M⁻¹. There is no clock unknown (two-way
 * ranging cancels it), so this is the pure 3-state position DOP and GDOP = PDOP.
 *
 * Requires at least 3 beacons with non-degenerate geometry; throws
 * std::invalid_argument for fewer than 3, and returns +inf DOPs if the geometry
 * is singular (e.g. all beacons collinear with the user).
 *
 * Worked case — beacons giving unit LOS along +N, +E, +D (M = I₃):
 *   HDOP = √2 ≈ 1.414, VDOP = 1.0, PDOP = GDOP = √3 ≈ 1.732.                ✓
 */
[[nodiscard]] inline RangeDop
range_dop(const std::vector<Vec3>& beacons, const Vec3& user) {
    if (beacons.size() < 3)
        throw std::invalid_argument("range_dop: need at least 3 beacons");

    std::array<std::array<double, 3>, 3> M{};  // GᵀG = Σ u uᵀ
    for (const auto& b : beacons) {
        const Vec3 u = los_unit(user, b);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                M[r][c] += u[r] * u[c];
    }

    std::array<std::array<double, 3>, 3> Q{};
    if (!detail::invert3x3(M, Q)) {
        const double inf = std::numeric_limits<double>::infinity();
        return {inf, inf, inf, inf};
    }

    const double hdop = std::sqrt(Q[0][0] + Q[1][1]);
    const double vdop = std::sqrt(Q[2][2]);
    const double pdop = std::sqrt(Q[0][0] + Q[1][1] + Q[2][2]);
    return {hdop, vdop, pdop, pdop};
}

}  // namespace navcore
