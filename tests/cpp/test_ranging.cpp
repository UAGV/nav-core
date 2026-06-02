#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <vector>
#include "navcore/ranging.hpp"

using namespace navcore;
using Catch::Matchers::WithinAbs;

TEST_CASE("range: [0,0,0]→[3,4,0] = 5", "[ranging]") {
    REQUIRE_THAT(range_m({0, 0, 0}, {3, 4, 0}), WithinAbs(5.0, 1e-12));
}

TEST_CASE("range_diff (TDOA): 5 − 12 = −7", "[ranging]") {
    REQUIRE_THAT(range_diff_m({0, 0, 0}, {5, 0, 0}, {0, 12, 0}),
                 WithinAbs(-7.0, 1e-12));
}

TEST_CASE("los_unit: [0,0,0]→[3,4,0] = [0.6,0.8,0]", "[ranging]") {
    const Vec3 u = los_unit({0, 0, 0}, {3, 4, 0});
    REQUIRE_THAT(u[0], WithinAbs(0.6, 1e-12));
    REQUIRE_THAT(u[1], WithinAbs(0.8, 1e-12));
    REQUIRE_THAT(u[2], WithinAbs(0.0, 1e-12));
}

TEST_CASE("los_unit: coincident points → zero vector", "[ranging]") {
    const Vec3 u = los_unit({1, 2, 3}, {1, 2, 3});
    REQUIRE_THAT(u[0], WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(u[1], WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(u[2], WithinAbs(0.0, 1e-12));
}

TEST_CASE("bearing: [1,1,0] → az 45°, el 0°", "[ranging]") {
    const Bearing b = bearing_from_ned({1, 1, 0});
    REQUIRE_THAT(b.azimuth_rad, WithinAbs(M_PI / 4.0, 1e-12));
    REQUIRE_THAT(b.elevation_rad, WithinAbs(0.0, 1e-12));
}

TEST_CASE("bearing: [0,0,-1] → az 0°, el 90° (up)", "[ranging]") {
    const Bearing b = bearing_from_ned({0, 0, -1});
    REQUIRE_THAT(b.azimuth_rad, WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(b.elevation_rad, WithinAbs(M_PI / 2.0, 1e-12));
}

TEST_CASE("range_dop: orthogonal-axis beacons (G=I₃)", "[ranging]") {
    // Beacons at unit distance along +N, +E, +D → unit LOS = identity rows.
    const std::vector<Vec3> beacons = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    const RangeDop dop = range_dop(beacons, {0, 0, 0});
    REQUIRE_THAT(dop.hdop, WithinAbs(std::sqrt(2.0), 1e-9));
    REQUIRE_THAT(dop.vdop, WithinAbs(1.0, 1e-9));
    REQUIRE_THAT(dop.pdop, WithinAbs(std::sqrt(3.0), 1e-9));
    REQUIRE_THAT(dop.gdop, WithinAbs(std::sqrt(3.0), 1e-9));
}

TEST_CASE("range_dop: fewer than 3 beacons throws", "[ranging]") {
    const std::vector<Vec3> beacons = {{1, 0, 0}, {0, 1, 0}};
    REQUIRE_THROWS_AS(range_dop(beacons, {0, 0, 0}), std::invalid_argument);
}

TEST_CASE("range_dop: collinear geometry is singular → inf", "[ranging]") {
    // All beacons on the N axis: geometry matrix is rank-1 → not invertible.
    const std::vector<Vec3> beacons = {{1, 0, 0}, {2, 0, 0}, {3, 0, 0}};
    const RangeDop dop = range_dop(beacons, {0, 0, 0});
    REQUIRE(std::isinf(dop.pdop));
}
