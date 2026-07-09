/**
 * @file frames.hpp
 * @brief Frame transforms: ECEF ↔ geodetic (LLH) ↔ NED/ENU; body ↔ nav; lever arm.
 *
 * Conventions (toolkit-wide):
 *   ECEF — Earth-Centred Earth-Fixed, [X, Y, Z] in metres.
 *   LLH  — geodetic [latitude_deg, longitude_deg, height_m].
 *   NED  — North-East-Down local tangent plane, origin at the reference LLH.
 *   ENU  — East-North-Up  local tangent plane.
 *   Body — FRD (Forward-Right-Down) by default; convention in meta.
 *
 * WGS-84 ellipsoid constants used throughout:
 *   a  = 6 378 137.0 m          (semi-major axis)
 *   f  = 1 / 298.257223563      (flattening)
 *   e² = 2f − f²                (eccentricity squared)
 *
 * Worked cases (checked in tests):
 *
 *   LLH → ECEF, at lat=0°, lon=0°, h=0:
 *     X = a, Y = 0, Z = 0  ≈ [6 378 137, 0, 0].                        ✓
 *
 *   LLH → ECEF, at lat=90°, lon=0°, h=0:
 *     X ≈ 0, Y ≈ 0, Z ≈ b  (b = a(1-f) ≈ 6 356 752.31 m).             ✓
 *
 *   ECEF → LLH round-trip for the two cases above:
 *     Should return the original inputs within 1e-9 deg / 1e-3 m.        ✓
 *
 *   NED offset of [1000, 0, 0] from lat=51°, lon=0°, h=0:
 *     Moves ~0.009° north; east and down unchanged.                      ✓
 */

#pragma once
#include <array>
#include <cmath>

#include "quaternion.hpp"  // Quaternion, quaternion_to_dcm (used by apply_lever_arm)

namespace navcore {

// -------------------------------------------------------------------------- //
// WGS-84 constants                                                           //
// -------------------------------------------------------------------------- //

/// Semi-major axis [m]
inline constexpr double WGS84_A = 6'378'137.0;
/// Flattening
inline constexpr double WGS84_F = 1.0 / 298.257'223'563;
/// First eccentricity squared = 2f − f²
inline constexpr double WGS84_E2 = 2.0 * WGS84_F - WGS84_F * WGS84_F;
/// Semi-minor axis b = a(1 − f) [m]
inline constexpr double WGS84_B = WGS84_A * (1.0 - WGS84_F);

/// π
inline constexpr double PI = 3.141592653589793238462643383279502884;
inline constexpr double DEG_TO_RAD = PI / 180.0;
inline constexpr double RAD_TO_DEG = 180.0 / PI;

// -------------------------------------------------------------------------- //
// LLH ↔ ECEF                                                                //
// -------------------------------------------------------------------------- //

/**
 * Convert geodetic LLH to ECEF Cartesian coordinates.
 *
 * Uses the closed-form:
 *   N(φ) = a / sqrt(1 − e²·sin²φ)
 *   X = (N + h)·cos φ·cos λ
 *   Y = (N + h)·cos φ·sin λ
 *   Z = (N(1 − e²) + h)·sin φ
 *
 * Worked case — lat=0°, lon=0°, h=0:
 *   N = a,  X = a ≈ 6 378 137,  Y = 0,  Z = 0.                         ✓
 *
 * @param latitude_deg   Geodetic latitude  [degrees].
 * @param longitude_deg  Geodetic longitude [degrees].
 * @param height_m       Ellipsoidal height [metres].
 * @returns  ECEF [X, Y, Z] in metres.
 */
[[nodiscard]] inline std::array<double, 3>
llh_to_ecef(double latitude_deg, double longitude_deg, double height_m) noexcept {
    const double lat = latitude_deg  * DEG_TO_RAD;
    const double lon = longitude_deg * DEG_TO_RAD;
    const double sin_lat = std::sin(lat), cos_lat = std::cos(lat);
    const double sin_lon = std::sin(lon), cos_lon = std::cos(lon);
    const double N = WGS84_A / std::sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat);
    return {
        (N + height_m) * cos_lat * cos_lon,
        (N + height_m) * cos_lat * sin_lon,
        (N * (1.0 - WGS84_E2) + height_m) * sin_lat,
    };
}

/**
 * Convert ECEF to geodetic LLH using the Bowring iterative method.
 *
 * Converges to sub-millimetre accuracy in 3–4 iterations for all altitudes
 * up to LEO orbit height.
 *
 * @param ecef_x_m, ecef_y_m, ecef_z_m  ECEF coordinates [metres].
 * @returns  [latitude_deg, longitude_deg, height_m].
 */
[[nodiscard]] inline std::array<double, 3>
ecef_to_llh(double ecef_x_m, double ecef_y_m, double ecef_z_m) noexcept {
    const double p = std::sqrt(ecef_x_m * ecef_x_m + ecef_y_m * ecef_y_m);
    const double longitude_rad = std::atan2(ecef_y_m, ecef_x_m);

    // Bowring: iterate on reduced latitude β
    double lat = std::atan2(ecef_z_m, p * (1.0 - WGS84_E2));
    for (int i = 0; i < 10; ++i) {
        const double sin_lat = std::sin(lat);
        const double N = WGS84_A / std::sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat);
        lat = std::atan2(ecef_z_m + WGS84_E2 * N * sin_lat, p);
    }
    const double sin_lat = std::sin(lat), cos_lat = std::cos(lat);
    const double N = WGS84_A / std::sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat);
    const double height_m = (p / cos_lat) - N;  // numerically safe away from poles

    return {lat * RAD_TO_DEG, longitude_rad * RAD_TO_DEG, height_m};
}

// -------------------------------------------------------------------------- //
// ECEF ↔ NED (local tangent plane)                                          //
// -------------------------------------------------------------------------- //

/**
 * Rotation matrix from ECEF to NED at a reference geodetic point.
 *
 * The NED frame has its origin at ref_llh; axes point North, East, Down
 * (tangent to the WGS-84 ellipsoid at that point).
 *
 * R_ned_from_ecef · v_ecef = v_ned
 *
 * @param ref_lat_deg  Reference latitude  [degrees].
 * @param ref_lon_deg  Reference longitude [degrees].
 * @returns  3×3 rotation matrix (row-major).
 */
[[nodiscard]] inline std::array<std::array<double, 3>, 3>
rotation_ned_from_ecef(double ref_lat_deg, double ref_lon_deg) noexcept {
    const double lat = ref_lat_deg * DEG_TO_RAD;
    const double lon = ref_lon_deg * DEG_TO_RAD;
    const double sl = std::sin(lat), cl = std::cos(lat);
    const double so = std::sin(lon), co = std::cos(lon);
    // Rows: N-hat, E-hat, D-hat in ECEF
    return {{
        {-sl * co,  -sl * so,   cl},
        {    -so,       co,    0.0},
        {-cl * co,  -cl * so,  -sl},
    }};
}

/**
 * Convert an ECEF displacement to a NED displacement.
 *
 * Useful for converting positions: v_ned = R · (r_ecef - origin_ecef).
 *
 * @param delta_ecef  ECEF displacement [m], [dX, dY, dZ].
 * @param ref_lat_deg, ref_lon_deg  Reference point geodetic latitude/longitude.
 * @returns  NED displacement [m], [dN, dE, dD].
 */
[[nodiscard]] inline std::array<double, 3>
ecef_delta_to_ned(const std::array<double, 3>& delta_ecef,
                  double ref_lat_deg, double ref_lon_deg) noexcept {
    const auto R = rotation_ned_from_ecef(ref_lat_deg, ref_lon_deg);
    return {
        R[0][0]*delta_ecef[0] + R[0][1]*delta_ecef[1] + R[0][2]*delta_ecef[2],
        R[1][0]*delta_ecef[0] + R[1][1]*delta_ecef[1] + R[1][2]*delta_ecef[2],
        R[2][0]*delta_ecef[0] + R[2][1]*delta_ecef[1] + R[2][2]*delta_ecef[2],
    };
}

/**
 * Convert a GNSS position in LLH to a NED position relative to a reference LLH.
 *
 * This is the standard "GNSS measurement → local NED offset" operation used in
 * every position filter.  For small offsets (<100 km) a flat-Earth approximation
 * introduces < 0.1% error; for larger offsets use this ECEF path.
 *
 * @param pos_llh      Position  [lat_deg, lon_deg, h_m].
 * @param ref_llh      Reference [lat_deg, lon_deg, h_m].
 * @returns  NED offset [dN, dE, dD] in metres.
 */
[[nodiscard]] inline std::array<double, 3>
llh_to_ned(const std::array<double, 3>& pos_llh,
           const std::array<double, 3>& ref_llh) noexcept {
    const auto p_ecef   = llh_to_ecef(pos_llh[0], pos_llh[1], pos_llh[2]);
    const auto ref_ecef = llh_to_ecef(ref_llh[0], ref_llh[1], ref_llh[2]);
    const std::array<double, 3> delta_ecef{
        p_ecef[0] - ref_ecef[0],
        p_ecef[1] - ref_ecef[1],
        p_ecef[2] - ref_ecef[2],
    };
    return ecef_delta_to_ned(delta_ecef, ref_llh[0], ref_llh[1]);
}

// -------------------------------------------------------------------------- //
// NED ↔ ENU                                                                 //
// -------------------------------------------------------------------------- //

/**
 * Convert NED [N, E, D] to ENU [E, N, U].
 *
 * Worked case: [1, 2, -3] NED → [2, 1, 3] ENU.                           ✓
 */
[[nodiscard]] inline constexpr std::array<double, 3>
ned_to_enu(const std::array<double, 3>& ned) noexcept {
    return {ned[1], ned[0], -ned[2]};
}

/**
 * Convert ENU [E, N, U] to NED [N, E, D].
 *
 * Worked case: [2, 1, 3] ENU → [1, 2, -3] NED.                           ✓
 */
[[nodiscard]] inline constexpr std::array<double, 3>
enu_to_ned(const std::array<double, 3>& enu) noexcept {
    return {enu[1], enu[0], -enu[2]};
}

// -------------------------------------------------------------------------- //
// Lever-arm application                                                      //
// -------------------------------------------------------------------------- //

/**
 * Apply a body-frame lever arm to translate a position measurement.
 *
 * When a GNSS antenna is offset from the IMU (or reference point) by a known
 * body-frame vector l_body_m, the position measured by the antenna differs from
 * the IMU position by:
 *
 *   p_ref_world = p_antenna_world − R · l_body_m
 *
 * where R = quaternion_to_dcm(q) is the **body→world** (active) rotation — the
 * NAV-021 convention: rotate_vector(q, v_body) = v_world, no transpose. (The
 * pre-NAV-022 code applied Rᵀ, derived from reading the parameter name as a
 * world→body map — the same transpose family the rest of the toolkit shed.)
 *
 * Worked cases —
 *   q = identity, l_body = [0.5, 0, 0]:
 *     p_ref = p_antenna − [0.5, 0, 0].                                    ✓
 *   q = 90° yaw [cos 45°, 0, 0, sin 45°], l_body = [0.5, 0, 0]:
 *     body-x points along world-y, so p_ref = p_antenna − [0, 0.5, 0].    ✓
 *
 * @param p_antenna_world_m  Antenna position in world frame [m].
 * @param q_body_from_world  Attitude quaternion (body-from-world).
 * @param lever_arm_body_m   Lever-arm vector in body frame [m].
 * @returns  Reference-point position in world frame [m].
 */
[[nodiscard]] inline std::array<double, 3>
apply_lever_arm(const std::array<double, 3>& p_antenna_world_m,
                const Quaternion& q_body_from_world,
                const std::array<double, 3>& lever_arm_body_m) noexcept {
    // Rotate lever arm from body to world: l_world = R · l_body  [NAV-022]
    const auto R = quaternion_to_dcm(q_body_from_world);
    const double lw_x = R[0][0]*lever_arm_body_m[0] + R[0][1]*lever_arm_body_m[1] + R[0][2]*lever_arm_body_m[2];
    const double lw_y = R[1][0]*lever_arm_body_m[0] + R[1][1]*lever_arm_body_m[1] + R[1][2]*lever_arm_body_m[2];
    const double lw_z = R[2][0]*lever_arm_body_m[0] + R[2][1]*lever_arm_body_m[1] + R[2][2]*lever_arm_body_m[2];
    return {
        p_antenna_world_m[0] - lw_x,
        p_antenna_world_m[1] - lw_y,
        p_antenna_world_m[2] - lw_z,
    };
}

} // namespace navcore
