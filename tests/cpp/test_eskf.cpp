#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include "navcore/eskf.hpp"

using namespace navcore;
using Catch::Matchers::WithinAbs;

static NominalState make_stationary_state() {
    NominalState s;
    // Zero position, velocity, identity attitude, zero biases
    return s;
}

static Eskf::ErrorStateMat make_P0() {
    Eskf::ErrorStateMat P{};
    P[0]   = 1.0;   // δp_N
    P[16]  = 1.0;   // δp_E
    P[32]  = 1.0;   // δp_D
    P[48]  = 0.01;  // δv_N
    P[64]  = 0.01;  // δv_E
    P[80]  = 0.01;  // δv_D
    P[96]  = 1e-4;  // δψ_x
    P[112] = 1e-4;  // δψ_y
    P[128] = 1e-4;  // δψ_z
    P[144] = 1e-6;  // δb_gx
    P[160] = 1e-6;  // δb_gy
    P[176] = 1e-6;  // δb_gz
    P[192] = 1e-4;  // δb_ax
    P[208] = 1e-4;  // δb_ay
    P[224] = 1e-4;  // δb_az
    return P;
}

static Eskf::ErrorStateMat make_Q(double dt_s) {
    Eskf::ErrorStateMat Q{};
    // Modest noise values
    const double q_v   = 0.01 * 0.01 * dt_s;
    const double q_psi = 1e-3 * 1e-3 * dt_s;
    const double q_bg  = 1e-5 * 1e-5 * dt_s;
    const double q_ba  = 1e-4 * 1e-4 * dt_s;
    Q[0]  = 1e-6 * dt_s;  Q[16] = 1e-6 * dt_s;  Q[32] = 1e-6 * dt_s;
    Q[48] = q_v;  Q[64] = q_v;  Q[80] = q_v;
    Q[96] = q_psi; Q[112] = q_psi; Q[128] = q_psi;
    Q[144] = q_bg; Q[160] = q_bg; Q[176] = q_bg;
    Q[192] = q_ba; Q[208] = q_ba; Q[224] = q_ba;
    return Q;
}

TEST_CASE("ESKF: stationary at rest with gravity-matched accel → stays at origin", "[eskf]") {
    // At rest, FRD body, NED world.  Gravity g = 9.80665 m/s² along NED-D.
    // Specific force is the physical f = a − g, so at rest the accelerometer reads
    // the "1 g up" reaction [0, 0, −g] in body frame (FRD: up is −body-z).
    // After mechanisation: a_ned = f + g = [0,0,0], so position and velocity hold.
    auto P0 = make_P0();
    Eskf eskf(make_stationary_state(), P0, DEFAULT_GRAVITY_M_PER_S2);
    const double dt = 0.01;
    const auto Q = make_Q(dt);

    const std::array<double,3> gyro{0.0, 0.0, 0.0};
    const std::array<double,3> accel{0.0, 0.0, -DEFAULT_GRAVITY_M_PER_S2};

    for (int i = 0; i < 100; ++i)
        eskf.predict(gyro, accel, dt, Q);

    const auto& ns = eskf.nominal_state();
    REQUIRE_THAT(ns.position_ned_m[0],       WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(ns.position_ned_m[1],       WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(ns.position_ned_m[2],       WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(ns.velocity_ned_m_per_s[0], WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(ns.velocity_ned_m_per_s[1], WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(ns.velocity_ned_m_per_s[2], WithinAbs(0.0, 1e-10));
}

TEST_CASE("ESKF: constant north acceleration → north velocity and position", "[eskf]") {
    // IMU reads specific force f = a − g = [a_fwd, 0, −g] where a_fwd is forward
    // (North in body FRD if nose points North); the −g is the gravity reaction.
    auto P0 = make_P0();
    Eskf eskf(make_stationary_state(), P0, DEFAULT_GRAVITY_M_PER_S2);
    const double dt      = 0.01;  // 100 Hz
    const double a_north = 1.0;   // 1 m/s² north
    const auto Q = make_Q(dt);

    const std::array<double,3> gyro{0.0, 0.0, 0.0};
    const std::array<double,3> accel{a_north, 0.0, -DEFAULT_GRAVITY_M_PER_S2};

    int N = 100;  // 1 second of data
    for (int i = 0; i < N; ++i)
        eskf.predict(gyro, accel, dt, Q);

    // After 1 s: v_N ≈ 1 m/s,  p_N ≈ 0.5 m  (kinematic)
    const auto& ns = eskf.nominal_state();
    REQUIRE_THAT(ns.velocity_ned_m_per_s[0], WithinAbs(1.0, 1e-8));
    REQUIRE_THAT(ns.position_ned_m[0],        WithinAbs(0.5, 1e-7));
    // East and Down should be zero
    REQUIRE_THAT(ns.velocity_ned_m_per_s[1], WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(ns.velocity_ned_m_per_s[2], WithinAbs(0.0, 1e-10));
}

TEST_CASE("ESKF: GNSS update corrects position error", "[eskf]") {
    // Inject a position error into the error state by initialising with a
    // non-zero δp, then check the GNSS update removes it.
    NominalState s;
    s.position_ned_m = {10.0, 0.0, 0.0};  // nominal says we're 10 m north
    auto P0 = make_P0();
    P0[0] = 4.0;  // δp_N variance = 4 m² (nominal may be off by ~2 m)
    Eskf eskf(s, P0, DEFAULT_GRAVITY_M_PER_S2);

    // GNSS says position is 8 m north (2 m south of nominal)
    const std::array<double,3> gnss_ned{8.0, 0.0, 0.0};
    const std::array<double,3> gnss_cov_diag{1.0, 1.0, 4.0};  // 1 m² horizontal
    eskf.update_gnss_position(gnss_ned, gnss_cov_diag);

    // After update: position should have moved toward 8 m
    // S = P[0][0] + R[0][0] = 4 + 1 = 5; K = 4/5 = 0.8
    // δp_N estimated = K * (8 - 10) = -1.6, so nominal ← 10 + (-1.6) = 8.4
    const auto& ns = eskf.nominal_state();
    REQUIRE_THAT(ns.position_ned_m[0], WithinAbs(8.4, 1e-9));
    // Covariance should decrease
    REQUIRE(eskf.error_covariance()[0] < 4.0);
}

TEST_CASE("ESKF: attitude quaternion remains unit after predict", "[eskf]") {
    auto P0 = make_P0();
    Eskf eskf(make_stationary_state(), P0, DEFAULT_GRAVITY_M_PER_S2);
    const double dt = 0.01;
    const auto Q = make_Q(dt);

    // Constant 0.1 rad/s yaw rate
    const std::array<double,3> gyro{0.0, 0.0, 0.1};
    const std::array<double,3> accel{0.0, 0.0, -DEFAULT_GRAVITY_M_PER_S2};

    for (int i = 0; i < 200; ++i)
        eskf.predict(gyro, accel, dt, Q);

    const auto& q = eskf.nominal_state().q_body_from_ned;
    const double norm_sq = q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z;
    REQUIRE_THAT(norm_sq, WithinAbs(1.0, 1e-10));
}
