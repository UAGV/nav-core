#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include "navcore/frames.hpp"

using namespace navcore;
using Catch::Matchers::WithinAbs;

TEST_CASE("LLH to ECEF: equatorial point at 0,0,0 → [a, 0, 0]", "[frames]") {
    auto ecef = llh_to_ecef(0.0, 0.0, 0.0);
    REQUIRE_THAT(ecef[0], WithinAbs(WGS84_A, 1e-3));
    REQUIRE_THAT(ecef[1], WithinAbs(0.0, 1e-3));
    REQUIRE_THAT(ecef[2], WithinAbs(0.0, 1e-3));
}

TEST_CASE("LLH to ECEF: north pole at lat=90,lon=0,h=0 → [0, 0, b]", "[frames]") {
    auto ecef = llh_to_ecef(90.0, 0.0, 0.0);
    // WGS-84 semi-minor axis b = a*(1-f)
    const double b = WGS84_A * (1.0 - WGS84_F);
    REQUIRE_THAT(ecef[0], WithinAbs(0.0, 1e-3));
    REQUIRE_THAT(ecef[1], WithinAbs(0.0, 1e-3));
    REQUIRE_THAT(ecef[2], WithinAbs(b, 1e-3));
}

TEST_CASE("LLH to ECEF: altitude adds radially at equator", "[frames]") {
    auto ecef0 = llh_to_ecef(0.0, 0.0, 0.0);
    auto ecef1 = llh_to_ecef(0.0, 0.0, 1000.0);
    REQUIRE_THAT(ecef1[0] - ecef0[0], WithinAbs(1000.0, 1e-3));
    REQUIRE_THAT(ecef1[1], WithinAbs(0.0, 1e-3));
    REQUIRE_THAT(ecef1[2], WithinAbs(0.0, 1e-3));
}

TEST_CASE("ECEF to LLH round-trip: equatorial", "[frames]") {
    double lat = 0.0, lon = 0.0, h = 0.0;
    auto ecef = llh_to_ecef(lat, lon, h);
    auto llh  = ecef_to_llh(ecef[0], ecef[1], ecef[2]);
    REQUIRE_THAT(llh[0], WithinAbs(lat, 1e-9));
    REQUIRE_THAT(llh[1], WithinAbs(lon, 1e-9));
    REQUIRE_THAT(llh[2], WithinAbs(h,   1e-3));
}

TEST_CASE("ECEF to LLH round-trip: mid-latitude", "[frames]") {
    double lat = 51.5, lon = -0.12, h = 50.0;
    auto ecef = llh_to_ecef(lat, lon, h);
    auto llh  = ecef_to_llh(ecef[0], ecef[1], ecef[2]);
    REQUIRE_THAT(llh[0], WithinAbs(lat, 1e-9));
    REQUIRE_THAT(llh[1], WithinAbs(lon, 1e-9));
    REQUIRE_THAT(llh[2], WithinAbs(h,   1e-3));
}

TEST_CASE("NED to ENU: [1,2,-3] NED → [2,1,3] ENU", "[frames]") {
    auto enu = ned_to_enu({1.0, 2.0, -3.0});
    REQUIRE_THAT(enu[0], WithinAbs(2.0, 1e-15));
    REQUIRE_THAT(enu[1], WithinAbs(1.0, 1e-15));
    REQUIRE_THAT(enu[2], WithinAbs(3.0, 1e-15));
}

TEST_CASE("ENU to NED round-trip", "[frames]") {
    std::array<double,3> ned{5.0, -3.0, 2.0};
    auto enu = ned_to_enu(ned);
    auto ned2 = enu_to_ned(enu);
    REQUIRE_THAT(ned2[0], WithinAbs(ned[0], 1e-15));
    REQUIRE_THAT(ned2[1], WithinAbs(ned[1], 1e-15));
    REQUIRE_THAT(ned2[2], WithinAbs(ned[2], 1e-15));
}

TEST_CASE("LLH to NED: same point gives zero offset", "[frames]") {
    std::array<double,3> ref{51.5, -0.12, 50.0};
    auto ned = llh_to_ned(ref, ref);
    REQUIRE_THAT(ned[0], WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(ned[1], WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(ned[2], WithinAbs(0.0, 1e-6));
}

TEST_CASE("LLH to NED: 1 km north moves N component ~1000 m", "[frames]") {
    // Rough check: 1 degree latitude ≈ 111 km
    std::array<double,3> ref{51.0, 0.0, 0.0};
    std::array<double,3> pos{51.0 + 1.0/111.0, 0.0, 0.0};  // ~1 km north
    auto ned = llh_to_ned(pos, ref);
    REQUIRE_THAT(ned[0], WithinAbs(1000.0, 5.0));   // North ≈ 1000 m (±5 m tolerance)
    REQUIRE_THAT(ned[1], WithinAbs(0.0, 1.0));       // East ≈ 0
    REQUIRE_THAT(ned[2], WithinAbs(0.0, 1.0));       // Down ≈ 0
}

TEST_CASE("Lever arm: identity attitude, lever along x → subtract lever from x", "[frames]") {
    Quaternion q{1.0, 0.0, 0.0, 0.0};  // identity
    std::array<double,3> p_ant{10.0, 5.0, 2.0};
    std::array<double,3> lever{ 0.5, 0.0, 0.0};
    auto p_ref = apply_lever_arm(p_ant, q, lever);
    REQUIRE_THAT(p_ref[0], WithinAbs(9.5, 1e-13));
    REQUIRE_THAT(p_ref[1], WithinAbs(5.0, 1e-13));
    REQUIRE_THAT(p_ref[2], WithinAbs(2.0, 1e-13));
}

TEST_CASE("Lever arm: 90 deg yaw rotates the body-x lever onto world-y", "[frames]") {
    // Body→world is the ACTIVE rotation (NAV-021 convention): at +90° yaw the
    // body-x lever points along world-y, so p_ref = p_ant − [0, 0.5, 0]. The
    // pre-NAV-022 transposed code gave p_ant − [0, −0.5, 0] — pinned here so the
    // wrong direction cannot return.
    const double half_sqrt2 = std::sqrt(0.5);
    Quaternion q{half_sqrt2, 0.0, 0.0, half_sqrt2};  // +90° yaw about NED-down
    std::array<double,3> p_ant{10.0, 20.0, 30.0};
    std::array<double,3> lever{ 0.5, 0.0, 0.0};
    auto p_ref = apply_lever_arm(p_ant, q, lever);
    REQUIRE_THAT(p_ref[0], WithinAbs(10.0, 1e-13));
    REQUIRE_THAT(p_ref[1], WithinAbs(19.5, 1e-13));
    REQUIRE_THAT(p_ref[2], WithinAbs(30.0, 1e-13));
}
