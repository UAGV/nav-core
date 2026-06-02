/**
 * @file quaternion.hpp
 * @brief Quaternion math: [w, x, y, z] convention, body-from-world sense.
 *
 * All quaternions use the Hamilton convention, [w, x, y, z], and represent
 * the rotation that takes a vector FROM world INTO body (i.e. q rotates a
 * world-frame vector into the body frame).
 *
 * Worked cases (also checked in the test suite):
 *
 *   Identity  q = [1, 0, 0, 0]:
 *     DCM = I₃,  rotates any vector v → v.                     ✓
 *
 *   90° yaw (+z) with FRD/NED:  q = [cos45°, 0, 0, sin45°] ≈ [0.7071, 0, 0, 0.7071]
 *     DCM₃₃ has R[0][0]=0, R[0][1]=1 → body-x maps to world-y (east).  ✓
 *     rotate_vector_by_quaternion(q, [1,0,0]) ≈ [0, 1, 0]              ✓
 *
 *   Compose two 90° yaws → 180° yaw:
 *     q1⊗q1 = [0, 0, 0, 1], which is a 180° rotation about z.          ✓
 */

#pragma once
#include <array>
#include <cmath>
#include <stdexcept>

namespace navcore {

// Quaternion stored as [w, x, y, z].
struct Quaternion {
    double w{1.0}, x{0.0}, y{0.0}, z{0.0};

    /** Squared norm — fast path used before normalisation checks. */
    [[nodiscard]] constexpr double norm_squared() const noexcept {
        return w * w + x * x + y * y + z * z;
    }

    [[nodiscard]] double norm() const noexcept { return std::sqrt(norm_squared()); }

    /** Return a unit quaternion.  Throws if the input is near-zero. */
    [[nodiscard]] Quaternion normalised() const {
        const double n = norm();
        if (n < 1e-15) throw std::domain_error("navcore: cannot normalise near-zero quaternion");
        return {w / n, x / n, y / n, z / n};
    }

    /** Conjugate — equal to the inverse for a unit quaternion. */
    [[nodiscard]] constexpr Quaternion conjugate() const noexcept { return {w, -x, -y, -z}; }

    /**
     * Hamilton product q_out = (*this) ⊗ rhs.
     *
     * Under the active rotation interpretation this chains rotations: first
     * apply rhs then (*this).  Under the passive (coordinate-transform)
     * interpretation used here (q rotates world→body), q_AB ⊗ q_BC rotates
     * world→C via frame B.
     */
    [[nodiscard]] constexpr Quaternion operator*(const Quaternion& rhs) const noexcept {
        return {
            w * rhs.w - x * rhs.x - y * rhs.y - z * rhs.z,
            w * rhs.x + x * rhs.w + y * rhs.z - z * rhs.y,
            w * rhs.y - x * rhs.z + y * rhs.w + z * rhs.x,
            w * rhs.z + x * rhs.y - y * rhs.x + z * rhs.w,
        };
    }
};

// -------------------------------------------------------------------------- //
// Direction-Cosine Matrix (DCM)                                              //
// -------------------------------------------------------------------------- //

/**
 * Convert a unit quaternion to the 3×3 DCM (rotation matrix).
 *
 * The DCM R satisfies  v_body = R * v_world  for quaternion q (body-from-world).
 *
 * Worked case — identity q = [1,0,0,0]:
 *   R = diag(1,1,1).                                                    ✓
 *
 * Worked case — 90° yaw q ≈ [0.7071, 0, 0, 0.7071]:
 *   R[0][0] ≈ 0,  R[0][1] ≈ 1  → body-x sees world-y.                  ✓
 *
 * @param q  Unit quaternion [w,x,y,z] (body-from-world).
 * @returns  Row-major 3×3 DCM as std::array<std::array<double,3>,3>.
 */
[[nodiscard]] inline std::array<std::array<double, 3>, 3>
quaternion_to_dcm(const Quaternion& q) noexcept {
    const double w2 = q.w * q.w, x2 = q.x * q.x, y2 = q.y * q.y, z2 = q.z * q.z;
    const double wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;
    const double xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
    return {{
        {w2 + x2 - y2 - z2,  2.0 * (xy - wz),     2.0 * (xz + wy)},
        {2.0 * (xy + wz),     w2 - x2 + y2 - z2,   2.0 * (yz - wx)},
        {2.0 * (xz - wy),     2.0 * (yz + wx),      w2 - x2 - y2 + z2},
    }};
}

/**
 * Build a unit quaternion from a DCM (rotation matrix, row-major).
 *
 * Uses Shepperd's method (numerically stable for all rotations).
 *
 * @param R  Row-major 3×3 rotation matrix.
 * @returns  Unit quaternion [w,x,y,z].
 */
[[nodiscard]] inline Quaternion
dcm_to_quaternion(const std::array<std::array<double, 3>, 3>& R) noexcept {
    const double trace = R[0][0] + R[1][1] + R[2][2];
    Quaternion q;
    if (trace > 0.0) {
        const double s = 0.5 / std::sqrt(trace + 1.0);
        q.w = 0.25 / s;
        q.x = (R[2][1] - R[1][2]) * s;
        q.y = (R[0][2] - R[2][0]) * s;
        q.z = (R[1][0] - R[0][1]) * s;
    } else if (R[0][0] > R[1][1] && R[0][0] > R[2][2]) {
        const double s = 2.0 * std::sqrt(1.0 + R[0][0] - R[1][1] - R[2][2]);
        q.w = (R[2][1] - R[1][2]) / s;
        q.x = 0.25 * s;
        q.y = (R[0][1] + R[1][0]) / s;
        q.z = (R[0][2] + R[2][0]) / s;
    } else if (R[1][1] > R[2][2]) {
        const double s = 2.0 * std::sqrt(1.0 + R[1][1] - R[0][0] - R[2][2]);
        q.w = (R[0][2] - R[2][0]) / s;
        q.x = (R[0][1] + R[1][0]) / s;
        q.y = 0.25 * s;
        q.z = (R[1][2] + R[2][1]) / s;
    } else {
        const double s = 2.0 * std::sqrt(1.0 + R[2][2] - R[0][0] - R[1][1]);
        q.w = (R[1][0] - R[0][1]) / s;
        q.x = (R[0][2] + R[2][0]) / s;
        q.y = (R[1][2] + R[2][1]) / s;
        q.z = 0.25 * s;
    }
    return q;
}

// -------------------------------------------------------------------------- //
// Euler angles (ZYX / yaw-pitch-roll)                                       //
// -------------------------------------------------------------------------- //

/** ZYX Euler angles [roll, pitch, yaw] in radians. */
struct EulerAnglesRad {
    double roll_rad{0.0}, pitch_rad{0.0}, yaw_rad{0.0};
};

/**
 * Convert a unit quaternion to ZYX Euler angles (yaw-pitch-roll).
 *
 * Convention: Rz(yaw) * Ry(pitch) * Rx(roll) = DCM.
 *
 * Worked case — identity q = [1,0,0,0]:
 *   roll=0, pitch=0, yaw=0.                                              ✓
 *
 * Worked case — 90° yaw q ≈ [0.7071, 0, 0, 0.7071]:
 *   roll=0, pitch=0, yaw≈π/2.                                            ✓
 */
[[nodiscard]] inline EulerAnglesRad
quaternion_to_euler_zyx(const Quaternion& q) noexcept {
    const auto R = quaternion_to_dcm(q);
    // Pitch: -arcsin(R[2][0]).  Clamped to [-1,1] for float precision at ±90°.
    const double sin_pitch = -R[2][0];
    const double pitch_rad = std::asin(std::max(-1.0, std::min(1.0, sin_pitch)));
    double roll_rad, yaw_rad;
    if (std::abs(sin_pitch) < 1.0 - 1e-10) {
        roll_rad = std::atan2(R[2][1], R[2][2]);
        yaw_rad  = std::atan2(R[1][0], R[0][0]);
    } else {
        // Gimbal lock: pitch ≈ ±90°; arbitrarily set roll to 0.
        roll_rad = 0.0;
        yaw_rad  = std::atan2(-R[0][1], R[1][1]);
    }
    return {roll_rad, pitch_rad, yaw_rad};
}

/**
 * Build a unit quaternion from ZYX Euler angles.
 *
 * Computes q = q_z ⊗ q_y ⊗ q_x where each q_i is a half-angle rotation.
 *
 * Worked case — all angles zero: returns identity [1,0,0,0].             ✓
 */
[[nodiscard]] inline Quaternion
euler_zyx_to_quaternion(const EulerAnglesRad& e) noexcept {
    const double cr = std::cos(e.roll_rad  * 0.5), sr = std::sin(e.roll_rad  * 0.5);
    const double cp = std::cos(e.pitch_rad * 0.5), sp = std::sin(e.pitch_rad * 0.5);
    const double cy = std::cos(e.yaw_rad   * 0.5), sy = std::sin(e.yaw_rad   * 0.5);
    return {
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    };
}

// -------------------------------------------------------------------------- //
// Vector rotation                                                            //
// -------------------------------------------------------------------------- //

/**
 * Rotate a 3-vector using quaternion sandwich: v_out = q ⊗ [0,v] ⊗ q*.
 *
 * With body-from-world q this takes v_world → v_body.
 *
 * Worked case — identity q: v_out = v_in.                                ✓
 * Worked case — 90° yaw q, v=[1,0,0]: v_out ≈ [0,1,0].                  ✓
 *
 * @param q  Unit quaternion [w,x,y,z].
 * @param v  Input 3-vector [x,y,z].
 * @returns  Rotated 3-vector.
 */
[[nodiscard]] inline std::array<double, 3>
rotate_vector_by_quaternion(const Quaternion& q,
                             const std::array<double, 3>& v) noexcept {
    // t = 2 * cross(q.xyz, v)
    const double tx = 2.0 * (q.y * v[2] - q.z * v[1]);
    const double ty = 2.0 * (q.z * v[0] - q.x * v[2]);
    const double tz = 2.0 * (q.x * v[1] - q.y * v[0]);
    return {
        v[0] + q.w * tx + q.y * tz - q.z * ty,
        v[1] + q.w * ty + q.z * tx - q.x * tz,
        v[2] + q.w * tz + q.x * ty - q.y * tx,
    };
}

} // namespace navcore
