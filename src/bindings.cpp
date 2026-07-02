/**
 * @file bindings.cpp
 * @brief pybind11 bindings for nav-core.
 *
 * This is the ONLY file in the C++ tree that knows numpy/Python exists.
 * All C++ math headers are dependency-free.
 *
 * Numpy ↔ C++ conventions (matching nav-data toolkit-wide):
 *   quaternions  (4,) float64  [w, x, y, z]
 *   vectors      (3,) float64  [x, y, z]
 *   matrices     (N, N) float64 row-major (C order)
 *   timestamps   kept as int64 ns in Python; not passed to C++ core
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "navcore/quaternion.hpp"
#include "navcore/frames.hpp"
#include "navcore/gnss_error.hpp"
#include "navcore/ranging.hpp"
#include "navcore/ekf.hpp"
#include "navcore/eskf.hpp"

namespace py = pybind11;
using namespace navcore;

// -------------------------------------------------------------------------- //
// Numpy helper utilities                                                     //
// -------------------------------------------------------------------------- //

static std::array<double, 3> vec3_from_numpy(const py::array_t<double>& arr) {
    auto buf = arr.request();
    if (buf.size != 3)
        throw std::invalid_argument("navcore: expected array of size 3");
    const double* ptr = static_cast<double*>(buf.ptr);
    return {ptr[0], ptr[1], ptr[2]};
}

static std::array<double, 4> vec4_from_numpy(const py::array_t<double>& arr) {
    auto buf = arr.request();
    if (buf.size != 4)
        throw std::invalid_argument("navcore: expected array of size 4");
    const double* ptr = static_cast<double*>(buf.ptr);
    return {ptr[0], ptr[1], ptr[2], ptr[3]};
}

static py::array_t<double> vec3_to_numpy(const std::array<double, 3>& v) {
    py::array_t<double> result(3);
    auto buf = result.request();
    double* ptr = static_cast<double*>(buf.ptr);
    ptr[0] = v[0]; ptr[1] = v[1]; ptr[2] = v[2];
    return result;
}

static py::array_t<double> vec4_to_numpy(const std::array<double, 4>& v) {
    py::array_t<double> result(4);
    auto buf = result.request();
    double* ptr = static_cast<double*>(buf.ptr);
    ptr[0] = v[0]; ptr[1] = v[1]; ptr[2] = v[2]; ptr[3] = v[3];
    return result;
}

static Quaternion quat_from_numpy(const py::array_t<double>& arr) {
    auto v = vec4_from_numpy(arr);
    return {v[0], v[1], v[2], v[3]};
}

static py::array_t<double> quat_to_numpy(const Quaternion& q) {
    return vec4_to_numpy({q.w, q.x, q.y, q.z});
}

static py::array_t<double> mat33_to_numpy(const std::array<std::array<double, 3>, 3>& m) {
    py::array_t<double> result({3, 3});
    auto buf = result.request();
    double* ptr = static_cast<double*>(buf.ptr);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            ptr[i * 3 + j] = m[i][j];
    return result;
}

template <int N>
static std::array<double, N * N> mat_from_numpy(const py::array_t<double>& arr) {
    auto buf = arr.request();
    if (buf.size != N * N)
        throw std::invalid_argument("navcore: matrix size mismatch");
    const double* ptr = static_cast<double*>(buf.ptr);
    std::array<double, N * N> out;
    std::copy(ptr, ptr + N * N, out.begin());
    return out;
}

template <int N>
static py::array_t<double> mat_to_numpy(const std::array<double, N * N>& m) {
    py::array_t<double> result({N, N});
    auto buf = result.request();
    double* ptr = static_cast<double*>(buf.ptr);
    std::copy(m.begin(), m.end(), ptr);
    return result;
}

// -------------------------------------------------------------------------- //
// Module                                                                     //
// -------------------------------------------------------------------------- //

PYBIND11_MODULE(_navcore, m) {
    m.doc() = "nav-core C++ bindings — rotations, frames, GNSS error budget, EKF/ESKF";

    // ---------------------------------------------------------------------- //
    // Quaternion                                                              //
    // ---------------------------------------------------------------------- //
    py::class_<Quaternion>(m, "Quaternion",
        "Unit quaternion [w, x, y, z], body-from-world convention.")
        .def(py::init<double, double, double, double>(),
             py::arg("w"), py::arg("x"), py::arg("y"), py::arg("z"))
        .def_readwrite("w", &Quaternion::w)
        .def_readwrite("x", &Quaternion::x)
        .def_readwrite("y", &Quaternion::y)
        .def_readwrite("z", &Quaternion::z)
        .def("norm",       &Quaternion::norm)
        .def("normalised", &Quaternion::normalised)
        .def("conjugate",  &Quaternion::conjugate)
        .def("__mul__",    &Quaternion::operator*)
        .def("__repr__", [](const Quaternion& q) {
            return "Quaternion(w=" + std::to_string(q.w) + ", x=" + std::to_string(q.x)
                 + ", y=" + std::to_string(q.y) + ", z=" + std::to_string(q.z) + ")";
        });

    // ---------------------------------------------------------------------- //
    // Rotation functions                                                      //
    // ---------------------------------------------------------------------- //
    m.def("quaternion_to_dcm",
        [](const py::array_t<double>& q_arr) {
            return mat33_to_numpy(quaternion_to_dcm(quat_from_numpy(q_arr)));
        },
        py::arg("q_wxyz"),
        "Convert quaternion [w,x,y,z] → 3×3 DCM (row-major, body-from-world).");

    m.def("dcm_to_quaternion",
        [](const py::array_t<double>& R) {
            auto buf = R.request();
            if (buf.size != 9)
                throw std::invalid_argument("navcore: DCM must be 3×3");
            const double* p = static_cast<const double*>(buf.ptr);
            std::array<std::array<double,3>,3> mat{{
                {p[0],p[1],p[2]}, {p[3],p[4],p[5]}, {p[6],p[7],p[8]}
            }};
            return quat_to_numpy(dcm_to_quaternion(mat));
        },
        py::arg("R"),
        "Convert 3×3 DCM → quaternion [w,x,y,z].");

    m.def("quaternion_to_euler_zyx",
        [](const py::array_t<double>& q_arr) {
            const auto e = quaternion_to_euler_zyx(quat_from_numpy(q_arr));
            py::array_t<double> out(3);
            auto buf = out.request();
            double* ptr = static_cast<double*>(buf.ptr);
            ptr[0] = e.roll_rad; ptr[1] = e.pitch_rad; ptr[2] = e.yaw_rad;
            return out;
        },
        py::arg("q_wxyz"),
        "Quaternion → ZYX Euler [roll_rad, pitch_rad, yaw_rad].");

    m.def("euler_zyx_to_quaternion",
        [](double roll_rad, double pitch_rad, double yaw_rad) {
            return quat_to_numpy(euler_zyx_to_quaternion({roll_rad, pitch_rad, yaw_rad}));
        },
        py::arg("roll_rad"), py::arg("pitch_rad"), py::arg("yaw_rad"),
        "ZYX Euler → quaternion [w,x,y,z].");

    m.def("rotate_vector",
        [](const py::array_t<double>& q_arr, const py::array_t<double>& v_arr) {
            return vec3_to_numpy(
                rotate_vector_by_quaternion(quat_from_numpy(q_arr), vec3_from_numpy(v_arr)));
        },
        py::arg("q_wxyz"), py::arg("v_xyz"),
        "Rotate v_world → v_body using quaternion q (body-from-world).");

    // ---------------------------------------------------------------------- //
    // Frame transforms                                                        //
    // ---------------------------------------------------------------------- //
    m.def("llh_to_ecef",
        [](double lat_deg, double lon_deg, double h_m) {
            return vec3_to_numpy(llh_to_ecef(lat_deg, lon_deg, h_m));
        },
        py::arg("latitude_deg"), py::arg("longitude_deg"), py::arg("height_m"),
        "Geodetic LLH → ECEF [X, Y, Z] metres.");

    m.def("ecef_to_llh",
        [](double X, double Y, double Z) {
            return vec3_to_numpy(ecef_to_llh(X, Y, Z));
        },
        py::arg("ecef_x_m"), py::arg("ecef_y_m"), py::arg("ecef_z_m"),
        "ECEF [X, Y, Z] → geodetic [lat_deg, lon_deg, h_m].");

    m.def("llh_to_ned",
        [](const py::array_t<double>& pos, const py::array_t<double>& ref) {
            return vec3_to_numpy(llh_to_ned(
                {vec3_from_numpy(pos)[0], vec3_from_numpy(pos)[1], vec3_from_numpy(pos)[2]},
                {vec3_from_numpy(ref)[0], vec3_from_numpy(ref)[1], vec3_from_numpy(ref)[2]}));
        },
        py::arg("pos_llh"), py::arg("ref_llh"),
        "LLH position → NED offset from ref_llh [m].");

    m.def("ned_to_enu",
        [](const py::array_t<double>& ned) {
            return vec3_to_numpy(ned_to_enu(vec3_from_numpy(ned)));
        },
        py::arg("ned_xyz"),
        "NED [N,E,D] → ENU [E,N,U].");

    m.def("enu_to_ned",
        [](const py::array_t<double>& enu) {
            return vec3_to_numpy(enu_to_ned(vec3_from_numpy(enu)));
        },
        py::arg("enu_xyz"),
        "ENU [E,N,U] → NED [N,E,D].");

    m.def("apply_lever_arm",
        [](const py::array_t<double>& p_antenna,
           const py::array_t<double>& q_arr,
           const py::array_t<double>& lever) {
            return vec3_to_numpy(apply_lever_arm(
                vec3_from_numpy(p_antenna),
                quat_from_numpy(q_arr),
                vec3_from_numpy(lever)));
        },
        py::arg("p_antenna_world_m"), py::arg("q_body_from_world"), py::arg("lever_arm_body_m"),
        "Translate antenna position to reference-point position by removing lever arm.");

    // ---------------------------------------------------------------------- //
    // GNSS error budget                                                       //
    // ---------------------------------------------------------------------- //
    m.def("compute_uere",
        &compute_uere_m,
        py::arg("sigma_clock_m"), py::arg("sigma_ephemeris_m"),
        py::arg("sigma_iono_m"),  py::arg("sigma_tropo_m"),
        py::arg("sigma_multipath_m"), py::arg("sigma_noise_m"),
        "Compose UERE [m] from individual 1-sigma error sources.");

    m.def("dop_to_position_sigma",
        &dop_to_position_sigma_m,
        py::arg("dop"), py::arg("sigma_uere_m"),
        "σ_pos = DOP · σ_UERE [m].");

    m.def("pdop_from_hdop_vdop",
        &pdop_from_hdop_vdop,
        py::arg("hdop"), py::arg("vdop"),
        "PDOP from HDOP and VDOP.");

    m.def("epu_to_nacp", &epu_to_nacp,
        py::arg("epu_95_m"),
        "Minimum NACp consistent with a 95% EPU (DO-260B).");

    m.def("nacp_to_epu_threshold",
        &nacp_to_epu_threshold_m,
        py::arg("nacp"),
        "Upper bound on EPU [m] for a given NACp level.");

    m.def("epu_95_to_sigma_h",
        &epu_95_to_sigma_h_m,
        py::arg("epu_95_m"), py::arg("use_two_sigma_approximation") = true,
        "Convert 95% EPU to 1-sigma horizontal std dev [m].");

    m.def("timing_to_range_sigma",
        &timing_to_range_sigma_m,
        py::arg("sigma_time_s"),
        "σ_r = c · σ_t  [m].");

    m.def("timing_to_tdoa_range_sigma",
        &timing_to_tdoa_range_sigma_m,
        py::arg("sigma_time_s"),
        "σ_r_tdoa = √2 · c · σ_t  [m].");

    m.def("gnss_position_covariance_ned",
        [](double hdop, double vdop, double sigma_uere_m) {
            return mat33_to_numpy(gnss_position_covariance_ned_m2(hdop, vdop, sigma_uere_m));
        },
        py::arg("hdop"), py::arg("vdop"), py::arg("sigma_uere_m"),
        "3×3 NED position covariance [m²] from HDOP, VDOP, σ_UERE.");

    // ---------------------------------------------------------------------- //
    // Ranging geometry (UWB / TDOA / rangefinder / bearing)                   //
    // ---------------------------------------------------------------------- //
    m.def("range_m",
        [](const py::array_t<double>& a, const py::array_t<double>& b) {
            return range_m(vec3_from_numpy(a), vec3_from_numpy(b));
        },
        py::arg("a"), py::arg("b"),
        "Euclidean range [m] between two 3-D points.");

    m.def("range_diff_m",
        [](const py::array_t<double>& user,
           const py::array_t<double>& anchor_a,
           const py::array_t<double>& anchor_b) {
            return range_diff_m(vec3_from_numpy(user),
                                vec3_from_numpy(anchor_a),
                                vec3_from_numpy(anchor_b));
        },
        py::arg("user"), py::arg("anchor_a"), py::arg("anchor_b"),
        "TDOA range difference [m] = range(user, a) − range(user, b).");

    m.def("los_unit",
        [](const py::array_t<double>& user, const py::array_t<double>& target) {
            return vec3_to_numpy(los_unit(vec3_from_numpy(user), vec3_from_numpy(target)));
        },
        py::arg("user"), py::arg("target"),
        "Unit line-of-sight user→target (range Jacobian = −los_unit).");

    m.def("bearing_from_ned",
        [](const py::array_t<double>& delta_ned) {
            const Bearing brg = bearing_from_ned(vec3_from_numpy(delta_ned));
            return py::make_tuple(brg.azimuth_rad, brg.elevation_rad);
        },
        py::arg("delta_ned"),
        "Bearing of an NED displacement → (azimuth_rad, elevation_rad).");

    m.def("range_dop",
        [](const std::vector<std::array<double, 3>>& beacons,
           const py::array_t<double>& user) {
            const RangeDop dop = range_dop(beacons, vec3_from_numpy(user));
            py::dict out;
            out["hdop"] = dop.hdop;
            out["vdop"] = dop.vdop;
            out["pdop"] = dop.pdop;
            out["gdop"] = dop.gdop;
            return out;
        },
        py::arg("beacons"), py::arg("user"),
        "Position DOP {hdop, vdop, pdop, gdop} for ranging to local beacons (NED).");

    // ---------------------------------------------------------------------- //
    // ESKF                                                                    //
    // ---------------------------------------------------------------------- //
    py::class_<NominalState>(m, "NominalState")
        .def(py::init<>())
        .def_property("position_ned_m",
            [](const NominalState& s) { return vec3_to_numpy(s.position_ned_m); },
            [](NominalState& s, const py::array_t<double>& v) {
                s.position_ned_m = vec3_from_numpy(v); })
        .def_property("velocity_ned_m_per_s",
            [](const NominalState& s) { return vec3_to_numpy(s.velocity_ned_m_per_s); },
            [](NominalState& s, const py::array_t<double>& v) {
                s.velocity_ned_m_per_s = vec3_from_numpy(v); })
        .def_property("q_body_from_ned",
            [](const NominalState& s) { return quat_to_numpy(s.q_body_from_ned); },
            [](NominalState& s, const py::array_t<double>& q) {
                s.q_body_from_ned = quat_from_numpy(q); })
        .def_property("bias_gyro_rad_per_s",
            [](const NominalState& s) { return vec3_to_numpy(s.bias_gyro_rad_per_s); },
            [](NominalState& s, const py::array_t<double>& v) {
                s.bias_gyro_rad_per_s = vec3_from_numpy(v); })
        .def_property("bias_accel_m_per_s2",
            [](const NominalState& s) { return vec3_to_numpy(s.bias_accel_m_per_s2); },
            [](NominalState& s, const py::array_t<double>& v) {
                s.bias_accel_m_per_s2 = vec3_from_numpy(v); });

    py::class_<Eskf>(m, "Eskf",
        "15-state Error-State Kalman Filter for aided inertial navigation (INS/GNSS).\n"
        "The filter engine an INS/GNSS estimator is built on, not itself an INS.\n\n"
        "Error-state ordering (0-indexed):\n"
        "  0-2:   δp  position error [m]   (NED)\n"
        "  3-5:   δv  velocity error [m/s] (NED)\n"
        "  6-8:   δψ  attitude error [rad] (rotation vector)\n"
        "  9-11:  δb_g gyro bias error [rad/s]\n"
        "  12-14: δb_a accel bias error [m/s²]")
        .def(py::init([](const NominalState& nom,
                         const py::array_t<double>& P,
                         double g) {
                 return std::make_unique<Eskf>(
                     nom, mat_from_numpy<ESKF_STATE_DIM>(P), g);
             }),
             py::arg("initial_nominal"),
             py::arg("initial_covariance_15x15"),
             py::arg("gravity_m_per_s2") = DEFAULT_GRAVITY_M_PER_S2)
        .def("predict",
            [](Eskf& self,
               const py::array_t<double>& gyro,
               const py::array_t<double>& accel,
               double dt_s,
               const py::array_t<double>& Q) {
                self.predict(vec3_from_numpy(gyro),
                             vec3_from_numpy(accel),
                             dt_s,
                             mat_from_numpy<ESKF_STATE_DIM>(Q));
            },
            py::arg("gyro_body_rad_per_s"),
            py::arg("accel_body_m_per_s2"),
            py::arg("dt_s"),
            py::arg("Q_15x15"),
            "IMU mechanisation + error-state prediction.")
        .def("update_gnss_position",
            [](Eskf& self,
               const py::array_t<double>& pos_ned,
               const py::array_t<double>& cov_diag) {
                self.update_gnss_position(
                    vec3_from_numpy(pos_ned),
                    vec3_from_numpy(cov_diag));
            },
            py::arg("gnss_position_ned_m"),
            py::arg("position_cov_diagonal_m2"),
            "Fuse a GNSS NED position measurement (diagonal covariance).")
        .def("update",
            [](Eskf& self,
               const py::array_t<double>& z,
               const py::array_t<double>& H,
               const py::array_t<double>& R,
               int M) {
                auto z_buf = z.request();
                std::vector<double> z_vec(
                    static_cast<double*>(z_buf.ptr),
                    static_cast<double*>(z_buf.ptr) + M);
                auto H_buf = H.request();
                std::vector<double> H_vec(
                    static_cast<double*>(H_buf.ptr),
                    static_cast<double*>(H_buf.ptr) + M * ESKF_STATE_DIM);
                auto R_buf = R.request();
                std::vector<double> R_vec(
                    static_cast<double*>(R_buf.ptr),
                    static_cast<double*>(R_buf.ptr) + M * M);
                self.update(z_vec, H_vec, R_vec, M);
            },
            py::arg("innovation_z"), py::arg("H_Mx15"), py::arg("R_MxM"), py::arg("M"),
            "Generic measurement update (custom H and R).")
        .def_property_readonly("nominal_state", &Eskf::nominal_state)
        .def_property_readonly("error_covariance",
            [](const Eskf& self) {
                return mat_to_numpy<ESKF_STATE_DIM>(self.error_covariance());
            })
        .def_property_readonly("innovation",
            [](const Eskf& self) {
                const auto& v = self.innovation();
                py::array_t<double> out(v.size());
                std::copy(v.begin(), v.end(), static_cast<double*>(out.request().ptr));
                return out;
            })
        .def_property_readonly("innovation_covariance",
            [](const Eskf& self) {
                const auto& v = self.innovation_covariance();
                int M = static_cast<int>(std::sqrt(v.size()));
                py::array_t<double> out({M, M});
                std::copy(v.begin(), v.end(), static_cast<double*>(out.request().ptr));
                return out;
            });

    // ---------------------------------------------------------------------- //
    // Constants                                                               //
    // ---------------------------------------------------------------------- //
    m.attr("ESKF_STATE_DIM") = ESKF_STATE_DIM;
    m.attr("SPEED_OF_LIGHT_M_PER_S") = SPEED_OF_LIGHT_M_PER_S;
    m.attr("WGS84_A") = WGS84_A;
    m.attr("WGS84_F") = WGS84_F;
    m.attr("WGS84_E2") = WGS84_E2;
}
