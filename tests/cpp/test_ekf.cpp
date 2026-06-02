#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "navcore/ekf.hpp"

using namespace navcore;
using Catch::Matchers::WithinAbs;

// ─── 1-D toy worked example from ekf.hpp docstring ───────────────────────────
//
//  Initial:  x=0, P=1.
//  Predict:  F=[[1]], Q=[[0.1]] → x⁻=0, P⁻=1.1
//  Update:   z=0.5, H=[[1]], R=[[0.25]]
//             S=1.35, K≈0.8148, x⁺≈0.4074, P⁺≈0.2037

TEST_CASE("1-D EKF: predict propagates x and P", "[ekf]") {
    Ekf<1> ekf(std::array<double,1>{0.0}, std::array<double,1>{1.0});
    ekf.predict(std::array<double,1>{1.0}, std::array<double,1>{0.1});
    REQUIRE_THAT(ekf.state()[0],      WithinAbs(0.0, 1e-14));
    REQUIRE_THAT(ekf.covariance()[0], WithinAbs(1.1, 1e-14));
}

TEST_CASE("1-D EKF: update gives hand-computed posterior", "[ekf]") {
    Ekf<1> ekf(std::array<double,1>{0.0}, std::array<double,1>{1.1});
    ekf.update(
        {0.5},       // z
        {1.0},       // H (1×1)
        {0.25},      // R (1×1)
        1
    );
    const double K     = 1.1 / 1.35;           // ≈ 0.81481
    const double x_exp = K * 0.5;              // ≈ 0.40741
    const double P_exp = (1.0 - K) * 1.1;     // ≈ 0.20370
    REQUIRE_THAT(ekf.state()[0],      WithinAbs(x_exp, 1e-10));
    REQUIRE_THAT(ekf.covariance()[0], WithinAbs(P_exp, 1e-10));
}

TEST_CASE("1-D EKF: innovation is z - H*x", "[ekf]") {
    Ekf<1> ekf(std::array<double,1>{0.0}, std::array<double,1>{1.1});
    ekf.update({0.5}, {1.0}, {0.25}, 1);
    REQUIRE_THAT(ekf.innovation()[0], WithinAbs(0.5, 1e-14));
}

TEST_CASE("1-D EKF: innovation covariance S = H*P*Ht + R", "[ekf]") {
    Ekf<1> ekf(std::array<double,1>{0.0}, std::array<double,1>{1.1});
    ekf.update({0.5}, {1.0}, {0.25}, 1);
    REQUIRE_THAT(ekf.innovation_covariance()[0], WithinAbs(1.35, 1e-12));
}

// ─── 2-D example: independent position + velocity ────────────────────────────
//
//  State [pos, vel], initial x=[0,1], P=diag(1,0.1)
//  Predict with constant-velocity F=[[1,dt],[0,1]], dt=0.5, Q=0
//   x⁻ = [0 + 1*0.5, 1] = [0.5, 1]
//   P⁻ = F*P*Fᵀ  (computed by hand)

TEST_CASE("2-D EKF: constant-velocity predict", "[ekf]") {
    using S2 = std::array<double, 2>;
    using M4 = std::array<double, 4>;
    Ekf<2> ekf(S2{0.0, 1.0}, M4{1.0, 0.0, 0.0, 0.1});
    const double dt = 0.5;
    // F = [[1, dt], [0, 1]]
    M4 F{1.0, dt, 0.0, 1.0};
    M4 Q{};  // zero process noise
    ekf.predict(F, Q);
    REQUIRE_THAT(ekf.state()[0], WithinAbs(0.5, 1e-14));
    REQUIRE_THAT(ekf.state()[1], WithinAbs(1.0, 1e-14));
}

TEST_CASE("2-D EKF: position measurement only", "[ekf]") {
    using S2 = std::array<double, 2>;
    using M4 = std::array<double, 4>;
    // State after predict: [0.5, 1.0], P = diag(1.1, 0.1) (approx)
    Ekf<2> ekf(S2{0.5, 1.0}, M4{1.1, 0.0, 0.0, 0.1});
    // Measure position = 0.6, R = 0.01
    ekf.update(
        {0.6},        // z
        {1.0, 0.0},   // H: measures only position (1×2)
        {0.01},       // R (1×1)
        1
    );
    // K[0] = P[0][0] * H[0] / S = 1.1 * 1 / 1.11 ≈ 0.9910
    // x⁺[0] = 0.5 + K[0] * (0.6 - 0.5) = 0.5 + 0.0991 ≈ 0.5991
    const double K  = 1.1 / 1.11;
    const double x0 = 0.5 + K * 0.1;
    REQUIRE_THAT(ekf.state()[0], WithinAbs(x0, 1e-10));
    REQUIRE_THAT(ekf.state()[1], WithinAbs(1.0, 1e-14));  // velocity unchanged
}
