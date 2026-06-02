#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include "navcore/gnss_error.hpp"

using namespace navcore;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

TEST_CASE("UERE: two equal 1 m components → √2", "[gnss_error]") {
    double uere = compute_uere_m(0.0, 0.0, 1.0, 0.0, 0.0, 1.0);
    REQUIRE_THAT(uere, WithinAbs(std::sqrt(2.0), 1e-12));
}

TEST_CASE("UERE: single 3 m component → 3 m", "[gnss_error]") {
    double uere = compute_uere_m(3.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    REQUIRE_THAT(uere, WithinAbs(3.0, 1e-12));
}

TEST_CASE("DOP → position sigma: HDOP=1.5, σ_UERE=2.0 → 3.0 m", "[gnss_error]") {
    REQUIRE_THAT(dop_to_position_sigma_m(1.5, 2.0), WithinAbs(3.0, 1e-12));
}

TEST_CASE("PDOP from HDOP=1, VDOP=2 → √5", "[gnss_error]") {
    REQUIRE_THAT(pdop_from_hdop_vdop(1.0, 2.0), WithinAbs(std::sqrt(5.0), 1e-12));
}

TEST_CASE("EPU to NACp: 2 m → 11", "[gnss_error]") {
    REQUIRE(epu_to_nacp(2.0)  == 11);
}

TEST_CASE("EPU to NACp: 8 m → 10", "[gnss_error]") {
    REQUIRE(epu_to_nacp(8.0)  == 10);
}

TEST_CASE("EPU to NACp: 25 m → 9", "[gnss_error]") {
    REQUIRE(epu_to_nacp(25.0) == 9);
}

TEST_CASE("EPU to NACp: at threshold boundary 3.0 m → 10 (not 11)", "[gnss_error]") {
    // Threshold for NACp 11 is < 3 m, so 3.0 m exactly → NACp 10
    REQUIRE(epu_to_nacp(3.0)  == 10);
}

TEST_CASE("NACp to EPU threshold: NACp=11 → 3.0 m", "[gnss_error]") {
    REQUIRE_THAT(nacp_to_epu_threshold_m(11), WithinAbs(3.0, 1e-12));
}

TEST_CASE("EPU 95 to sigma_H: 10 m, two-sigma → 5 m", "[gnss_error]") {
    REQUIRE_THAT(epu_95_to_sigma_h_m(10.0, true), WithinAbs(5.0, 1e-12));
}

TEST_CASE("EPU 95 to sigma_H: 10 m, Rayleigh → ~4.086 m", "[gnss_error]") {
    REQUIRE_THAT(epu_95_to_sigma_h_m(10.0, false), WithinAbs(10.0 / 2.4477, 1e-4));
}

TEST_CASE("Timing to range: 10 ns → ~2.998 m", "[gnss_error]") {
    double sigma_r = timing_to_range_sigma_m(10e-9);
    REQUIRE_THAT(sigma_r, WithinAbs(2.997924580, 1e-6));
}

TEST_CASE("Timing to TDOA range: 10 ns → √2 × one-way", "[gnss_error]") {
    double sigma_r_ow   = timing_to_range_sigma_m(10e-9);
    double sigma_r_tdoa = timing_to_tdoa_range_sigma_m(10e-9);
    REQUIRE_THAT(sigma_r_tdoa, WithinAbs(std::sqrt(2.0) * sigma_r_ow, 1e-10));
}

TEST_CASE("GNSS position covariance NED: HDOP=1, VDOP=2, σ_UERE=1.5", "[gnss_error]") {
    auto P = gnss_position_covariance_ned_m2(1.0, 2.0, 1.5);
    REQUIRE_THAT(P[0][0], WithinAbs(2.25, 1e-12));  // σ_H² = 1.5²
    REQUIRE_THAT(P[1][1], WithinAbs(2.25, 1e-12));
    REQUIRE_THAT(P[2][2], WithinAbs(9.0,  1e-12));  // σ_V² = 3.0²
    // Off-diagonal should be zero
    REQUIRE_THAT(P[0][1], WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(P[0][2], WithinAbs(0.0, 1e-12));
}
