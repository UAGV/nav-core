"""Python-side binding tests.

These verify the numpy ↔ C++ marshalling layer and re-check the key
worked cases from the C++ headers (so a binding regression is caught
even if the C++ tests are not run).
"""

from __future__ import annotations

import math
import numpy as np
import pytest

import navcore._navcore as nc  # type: ignore[import]


# --------------------------------------------------------------------------- #
# Rotations                                                                   #
# --------------------------------------------------------------------------- #

def test_quaternion_to_dcm_identity() -> None:
    q = np.array([1.0, 0.0, 0.0, 0.0])
    R = nc.quaternion_to_dcm(q)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-14)


def test_quaternion_to_dcm_90deg_yaw() -> None:
    c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
    q = np.array([c, 0.0, 0.0, s])
    R = nc.quaternion_to_dcm(q)
    # quaternion_to_dcm(q) is the body→world (active) DCM, consistent with
    # rotate_vector: for +90° yaw, body-x maps to world +y, so column 0 = [0, 1, 0]
    # and the matrix is the active z-rotation [[0,-1,0],[1,0,0],[0,0,1]]. [NAV-021/NAV-002b]
    np.testing.assert_allclose(R @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-14)
    np.testing.assert_allclose(R[0, 1], -1.0, atol=1e-14)


def test_rotate_vector_identity() -> None:
    q = np.array([1.0, 0.0, 0.0, 0.0])
    v = np.array([3.0, -1.5, 7.2])
    np.testing.assert_allclose(nc.rotate_vector(q, v), v, atol=1e-14)


def test_rotate_vector_90deg_yaw() -> None:
    c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
    q = np.array([c, 0.0, 0.0, s])
    result = nc.rotate_vector(q, np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(result, [0.0, 1.0, 0.0], atol=1e-14)


def test_euler_zyx_round_trip() -> None:
    roll, pitch, yaw = 0.3, -0.2, 1.1
    q = nc.euler_zyx_to_quaternion(roll, pitch, yaw)
    rpy = nc.quaternion_to_euler_zyx(q)
    np.testing.assert_allclose(rpy, [roll, pitch, yaw], atol=1e-12)


def test_dcm_round_trip() -> None:
    q_in = nc.euler_zyx_to_quaternion(0.5, -0.3, 1.2)
    R = nc.quaternion_to_dcm(q_in)
    q_out = nc.dcm_to_quaternion(R)
    dot = float(np.dot(q_in, q_out))
    assert abs(abs(dot) - 1.0) < 1e-12, f"Quaternion round-trip dot={dot}"


# --------------------------------------------------------------------------- #
# Frame transforms                                                            #
# --------------------------------------------------------------------------- #

def test_llh_to_ecef_equator() -> None:
    ecef = nc.llh_to_ecef(0.0, 0.0, 0.0)
    np.testing.assert_allclose(ecef[0], nc.WGS84_A, rtol=1e-9)
    np.testing.assert_allclose(ecef[1], 0.0, atol=1e-3)
    np.testing.assert_allclose(ecef[2], 0.0, atol=1e-3)


def test_ecef_llh_round_trip() -> None:
    lat, lon, h = 51.5, -0.12, 50.0
    ecef = nc.llh_to_ecef(lat, lon, h)
    llh = nc.ecef_to_llh(ecef[0], ecef[1], ecef[2])
    np.testing.assert_allclose(llh[0], lat, atol=1e-9)
    np.testing.assert_allclose(llh[1], lon, atol=1e-9)
    np.testing.assert_allclose(llh[2], h,   atol=1e-3)


def test_llh_to_ned_same_point_is_zero() -> None:
    ref = np.array([51.5, -0.12, 50.0])
    ned = nc.llh_to_ned(ref, ref)
    np.testing.assert_allclose(ned, [0.0, 0.0, 0.0], atol=1e-6)


def test_ned_enu_round_trip() -> None:
    ned = np.array([5.0, -3.0, 2.0])
    enu = nc.ned_to_enu(ned)
    ned2 = nc.enu_to_ned(enu)
    np.testing.assert_allclose(ned2, ned, atol=1e-15)


def test_apply_lever_arm_identity_attitude() -> None:
    q = np.array([1.0, 0.0, 0.0, 0.0])
    p_ant = np.array([10.0, 5.0, 2.0])
    lever = np.array([0.5, 0.0, 0.0])
    p_ref = nc.apply_lever_arm(p_ant, q, lever)
    np.testing.assert_allclose(p_ref, [9.5, 5.0, 2.0], atol=1e-13)


def test_apply_lever_arm_90deg_yaw_rotates_lever_to_world_y() -> None:
    # Body→world is the ACTIVE rotation (NAV-021 convention): at +90° yaw the
    # body-x lever points along world-y, so p_ref = p_ant − [0, 0.5, 0]. The
    # pre-NAV-022 transposed code gave p_ant − [0, −0.5, 0] — pinned so the
    # wrong direction cannot return.
    half_sqrt2 = np.sqrt(0.5)
    q = np.array([half_sqrt2, 0.0, 0.0, half_sqrt2])  # +90° yaw about NED-down
    p_ant = np.array([10.0, 20.0, 30.0])
    lever = np.array([0.5, 0.0, 0.0])
    p_ref = nc.apply_lever_arm(p_ant, q, lever)
    np.testing.assert_allclose(p_ref, [10.0, 19.5, 30.0], atol=1e-13)


# --------------------------------------------------------------------------- #
# GNSS error budget                                                           #
# --------------------------------------------------------------------------- #

def test_uere_two_equal_components() -> None:
    uere = nc.compute_uere(0.0, 0.0, 1.0, 0.0, 0.0, 1.0)
    assert abs(uere - math.sqrt(2.0)) < 1e-12


def test_dop_to_position_sigma() -> None:
    assert abs(nc.dop_to_position_sigma(1.5, 2.0) - 3.0) < 1e-12


def test_pdop_from_hdop_vdop() -> None:
    assert abs(nc.pdop_from_hdop_vdop(1.0, 2.0) - math.sqrt(5.0)) < 1e-12


def test_epu_to_nacp() -> None:
    assert nc.epu_to_nacp(2.0)  == 11
    assert nc.epu_to_nacp(8.0)  == 10
    assert nc.epu_to_nacp(25.0) == 9
    assert nc.epu_to_nacp(3.0)  == 10   # threshold is strict <


def test_timing_to_range_sigma() -> None:
    sigma_r = nc.timing_to_range_sigma(10e-9)
    # c × 10 ns = 299792458 × 10e-9 = 2.99792458 m
    assert abs(sigma_r - 2.99792458) < 1e-6


def test_timing_to_tdoa_range_sigma_is_sqrt2_times_one_way() -> None:
    r1 = nc.timing_to_range_sigma(10e-9)
    r2 = nc.timing_to_tdoa_range_sigma(10e-9)
    assert abs(r2 - math.sqrt(2.0) * r1) < 1e-10


def test_gnss_position_covariance_ned_shape_and_values() -> None:
    P = nc.gnss_position_covariance_ned(1.0, 2.0, 1.5)
    assert P.shape == (3, 3)
    np.testing.assert_allclose(P[0, 0], 1.5 ** 2, atol=1e-12)
    np.testing.assert_allclose(P[1, 1], 1.5 ** 2, atol=1e-12)
    np.testing.assert_allclose(P[2, 2], 3.0 ** 2, atol=1e-12)
    np.testing.assert_allclose(P[0, 1], 0.0, atol=1e-12)


# --------------------------------------------------------------------------- #
# ESKF via bindings                                                           #
# --------------------------------------------------------------------------- #

def _make_eskf() -> nc.Eskf:
    nom = nc.NominalState()
    nom.position_ned_m       = np.zeros(3)
    nom.velocity_ned_m_per_s = np.zeros(3)
    nom.q_body_from_ned      = np.array([1.0, 0.0, 0.0, 0.0])
    P0 = np.diag([1.0]*3 + [0.01]*3 + [1e-4]*3 + [1e-6]*3 + [1e-4]*3).astype(np.float64)
    return nc.Eskf(nom, P0, 9.80665)


def _make_Q(dt: float) -> np.ndarray:
    diag = np.array([
        1e-6*dt, 1e-6*dt, 1e-6*dt,       # δp
        1e-4*dt, 1e-4*dt, 1e-4*dt,       # δv
        1e-6*dt, 1e-6*dt, 1e-6*dt,       # δψ
        1e-10*dt, 1e-10*dt, 1e-10*dt,    # δb_g
        1e-8*dt, 1e-8*dt, 1e-8*dt,       # δb_a
    ])
    return np.diag(diag).astype(np.float64)


def test_eskf_stationary_stays_at_origin() -> None:
    eskf = _make_eskf()
    dt = 0.01
    Q = _make_Q(dt)
    gyro  = np.zeros(3)
    accel = np.array([0.0, 0.0, -9.80665])  # physical f = a − g: 1 g up at rest

    for _ in range(100):
        eskf.predict(gyro, accel, dt, Q)

    ns = eskf.nominal_state
    np.testing.assert_allclose(ns.position_ned_m,       [0, 0, 0], atol=1e-10)
    np.testing.assert_allclose(ns.velocity_ned_m_per_s, [0, 0, 0], atol=1e-10)


def test_eskf_constant_north_accel() -> None:
    eskf = _make_eskf()
    dt = 0.01
    Q = _make_Q(dt)
    gyro  = np.zeros(3)
    accel = np.array([1.0, 0.0, -9.80665])  # f = a − g: 1 m/s² north, 1 g up

    for _ in range(100):  # 1 second
        eskf.predict(gyro, accel, dt, Q)

    ns = eskf.nominal_state
    np.testing.assert_allclose(ns.velocity_ned_m_per_s[0], 1.0, atol=1e-8)
    np.testing.assert_allclose(ns.position_ned_m[0],        0.5, atol=1e-7)


def test_eskf_gnss_update_corrects_error() -> None:
    nom = nc.NominalState()
    nom.position_ned_m       = np.array([10.0, 0.0, 0.0])
    nom.velocity_ned_m_per_s = np.zeros(3)
    nom.q_body_from_ned      = np.array([1.0, 0.0, 0.0, 0.0])
    P0 = np.diag([4.0, 1.0, 1.0] + [0.01]*3 + [1e-4]*3 + [1e-6]*3 + [1e-4]*3).astype(np.float64)
    eskf = nc.Eskf(nom, P0, 9.80665)

    eskf.update_gnss_position(
        np.array([8.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 4.0]),
    )

    # K = 4/(4+1) = 0.8; δp_N = 0.8*(8-10) = -1.6; nominal → 10-1.6 = 8.4
    ns = eskf.nominal_state
    np.testing.assert_allclose(ns.position_ned_m[0], 8.4, atol=1e-9)


def test_eskf_covariance_shape() -> None:
    eskf = _make_eskf()
    P = eskf.error_covariance
    assert P.shape == (15, 15)


def test_eskf_attitude_stays_unit_after_yaw_rate() -> None:
    eskf = _make_eskf()
    dt = 0.01
    Q = _make_Q(dt)
    gyro  = np.array([0.0, 0.0, 0.1])
    accel = np.array([0.0, 0.0, -9.80665])  # physical f = a − g
    for _ in range(200):
        eskf.predict(gyro, accel, dt, Q)
    q = eskf.nominal_state.q_body_from_ned
    assert abs(float(np.linalg.norm(q)) - 1.0) < 1e-10


# --------------------------------------------------------------------------- #
# Ranging geometry (nav-core v0.2.0)                                          #
# --------------------------------------------------------------------------- #

def test_range_m() -> None:
    assert nc.range_m(np.array([0.0, 0.0, 0.0]),
                      np.array([3.0, 4.0, 0.0])) == pytest.approx(5.0)


def test_range_diff_tdoa() -> None:
    d = nc.range_diff_m(np.array([0.0, 0.0, 0.0]),
                        np.array([5.0, 0.0, 0.0]),
                        np.array([0.0, 12.0, 0.0]))
    assert d == pytest.approx(-7.0)


def test_los_unit() -> None:
    u = nc.los_unit(np.array([0.0, 0.0, 0.0]), np.array([3.0, 4.0, 0.0]))
    np.testing.assert_allclose(u, [0.6, 0.8, 0.0], atol=1e-12)


def test_bearing_from_ned() -> None:
    az, el = nc.bearing_from_ned(np.array([1.0, 1.0, 0.0]))
    assert az == pytest.approx(math.pi / 4)
    assert el == pytest.approx(0.0)
    az_up, el_up = nc.bearing_from_ned(np.array([0.0, 0.0, -1.0]))
    assert el_up == pytest.approx(math.pi / 2)


def test_range_dop_orthogonal() -> None:
    beacons = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    dop = nc.range_dop(beacons, np.array([0.0, 0.0, 0.0]))
    assert dop["hdop"] == pytest.approx(math.sqrt(2.0), abs=1e-9)
    assert dop["vdop"] == pytest.approx(1.0, abs=1e-9)
    assert dop["pdop"] == pytest.approx(math.sqrt(3.0), abs=1e-9)


def test_range_dop_too_few_beacons_raises() -> None:
    with pytest.raises(Exception):
        nc.range_dop([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.array([0.0, 0.0, 0.0]))
