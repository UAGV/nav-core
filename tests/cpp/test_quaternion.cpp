#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include "navcore/quaternion.hpp"

using namespace navcore;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

static constexpr double EPS = 1e-12;
static constexpr double EPS_DEG = 1e-10;

TEST_CASE("Identity quaternion: norm is 1", "[quaternion]") {
    Quaternion q{1.0, 0.0, 0.0, 0.0};
    REQUIRE_THAT(q.norm(), WithinAbs(1.0, EPS));
}

TEST_CASE("Identity quaternion: DCM is identity", "[quaternion]") {
    Quaternion q{1.0, 0.0, 0.0, 0.0};
    auto R = quaternion_to_dcm(q);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            REQUIRE_THAT(R[i][j], WithinAbs(i == j ? 1.0 : 0.0, EPS));
}

TEST_CASE("Identity quaternion: rotate vector is no-op", "[quaternion]") {
    Quaternion q{1.0, 0.0, 0.0, 0.0};
    auto v = rotate_vector_by_quaternion(q, {3.0, -1.5, 7.2});
    REQUIRE_THAT(v[0], WithinAbs(3.0,  EPS));
    REQUIRE_THAT(v[1], WithinAbs(-1.5, EPS));
    REQUIRE_THAT(v[2], WithinAbs(7.2,  EPS));
}

TEST_CASE("90 degree yaw: quaternion rotates x-axis to y-axis", "[quaternion]") {
    // q = [cos45°, 0, 0, sin45°], 90° rotation about z
    const double c = std::cos(M_PI / 4.0);
    const double s = std::sin(M_PI / 4.0);
    Quaternion q{c, 0.0, 0.0, s};
    auto v = rotate_vector_by_quaternion(q, {1.0, 0.0, 0.0});
    REQUIRE_THAT(v[0], WithinAbs(0.0, 1e-14));
    REQUIRE_THAT(v[1], WithinAbs(1.0, 1e-14));
    REQUIRE_THAT(v[2], WithinAbs(0.0, 1e-14));
}

TEST_CASE("Composition of two 90 degree yaws gives 180 degree yaw", "[quaternion]") {
    const double c = std::cos(M_PI / 4.0);
    const double s = std::sin(M_PI / 4.0);
    Quaternion q{c, 0.0, 0.0, s};
    auto q2 = q * q;
    // 180° yaw: [0,0,0,1] (up to sign — both represent same rotation)
    REQUIRE_THAT(std::abs(q2.w), WithinAbs(0.0, 1e-14));
    REQUIRE_THAT(std::abs(q2.x), WithinAbs(0.0, 1e-14));
    REQUIRE_THAT(std::abs(q2.y), WithinAbs(0.0, 1e-14));
    REQUIRE_THAT(std::abs(q2.z), WithinAbs(1.0, 1e-14));
}

TEST_CASE("DCM round-trip: dcm_to_quaternion inverts quaternion_to_dcm", "[quaternion]") {
    // 30° roll, 15° pitch, 45° yaw
    const auto q_in = euler_zyx_to_quaternion({30.0 * M_PI/180.0, 15.0 * M_PI/180.0, 45.0 * M_PI/180.0});
    const auto R = quaternion_to_dcm(q_in);
    const auto q_out = dcm_to_quaternion(R).normalised();
    // Quaternion double-cover: q and -q represent the same rotation
    const double dot = q_in.w*q_out.w + q_in.x*q_out.x + q_in.y*q_out.y + q_in.z*q_out.z;
    REQUIRE_THAT(std::abs(dot), WithinAbs(1.0, 1e-12));
}

TEST_CASE("Euler round-trip: identity", "[quaternion]") {
    Quaternion q{1.0, 0.0, 0.0, 0.0};
    auto e = quaternion_to_euler_zyx(q);
    REQUIRE_THAT(e.roll_rad,  WithinAbs(0.0, EPS_DEG));
    REQUIRE_THAT(e.pitch_rad, WithinAbs(0.0, EPS_DEG));
    REQUIRE_THAT(e.yaw_rad,   WithinAbs(0.0, EPS_DEG));
}

TEST_CASE("Euler round-trip: 90 degree yaw", "[quaternion]") {
    const double c = std::cos(M_PI / 4.0);
    const double s = std::sin(M_PI / 4.0);
    Quaternion q{c, 0.0, 0.0, s};
    auto e = quaternion_to_euler_zyx(q);
    REQUIRE_THAT(e.roll_rad,  WithinAbs(0.0,      1e-10));
    REQUIRE_THAT(e.pitch_rad, WithinAbs(0.0,      1e-10));
    REQUIRE_THAT(e.yaw_rad,   WithinAbs(M_PI/2.0, 1e-10));
}

TEST_CASE("euler_zyx_to_quaternion → quaternion_to_euler_zyx round-trip", "[quaternion]") {
    EulerAnglesRad e_in{0.3, -0.2, 1.1};
    auto q = euler_zyx_to_quaternion(e_in);
    auto e_out = quaternion_to_euler_zyx(q);
    REQUIRE_THAT(e_out.roll_rad,  WithinAbs(e_in.roll_rad,  1e-12));
    REQUIRE_THAT(e_out.pitch_rad, WithinAbs(e_in.pitch_rad, 1e-12));
    REQUIRE_THAT(e_out.yaw_rad,   WithinAbs(e_in.yaw_rad,   1e-12));
}

TEST_CASE("Normalised quaternion has unit norm", "[quaternion]") {
    Quaternion q{2.0, 1.0, -3.0, 0.5};
    auto qn = q.normalised();
    REQUIRE_THAT(qn.norm(), WithinAbs(1.0, EPS));
}

TEST_CASE("Conjugate of unit quaternion is its inverse", "[quaternion]") {
    const double c = std::cos(0.4);
    const double s = std::sin(0.4);
    Quaternion q{c, s*0.6, s*0.8, 0.0};  // rotation about some axis, already unit
    auto q_conj = q.conjugate();
    auto q_prod = q * q_conj;
    // q * q* = identity
    REQUIRE_THAT(q_prod.w, WithinAbs(1.0, 1e-14));
    REQUIRE_THAT(q_prod.x, WithinAbs(0.0, 1e-14));
    REQUIRE_THAT(q_prod.y, WithinAbs(0.0, 1e-14));
    REQUIRE_THAT(q_prod.z, WithinAbs(0.0, 1e-14));
}
