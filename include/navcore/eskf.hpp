/**
 * @file eskf.hpp
 * @brief Error-State Kalman Filter (ESKF): 15-state error model. The filter
 *        engine an aided inertial navigation system (INS/GNSS) is built on —
 *        not itself an INS.
 *
 * ─── Why ESKF instead of a standard EKF on the full navigation state? ───────
 *
 *   Attitude is a rotation, which lives on SO(3) — a manifold, not a vector
 *   space.  If you track attitude directly as a quaternion you have to
 *   constrain its norm every step and deal with linearisation at a quaternion
 *   level.  The ESKF instead:
 *
 *     1. Propagates a NOMINAL state via full nonlinear kinematics (no
 *        approximation here: the IMU mechanisation equations are integrated
 *        exactly with respect to rotation).
 *     2. Tracks an ERROR state δx in a 15-D tangent space around the nominal,
 *        where attitude error δψ is a 3-vector (a "rotation vector" or Rodrigues
 *        parameter in the small-angle limit).
 *     3. Runs a linear KF on δx.
 *     4. After an update, injects the estimated error back into the nominal
 *        state (the "reset" step) and re-zeroes the error state.
 *
 *   This avoids over-parameterisation, keeps P well-conditioned, and is the
 *   industry standard for high-grade INS/GNSS (Farrell 2008, Sola 2017).
 *
 * ─── State ordering (FIXED — nav-eval must match) ───────────────────────────
 *
 *   Error-state vector δx ∈ ℝ¹⁵ (exposed in TrajectoryEstimate.meta):
 *
 *     Indices  0– 2:  δp    position error [m] in world (NED)
 *     Indices  3– 5:  δv    velocity error [m/s] in world (NED)
 *     Indices  6– 8:  δψ    attitude error [rad], small rotation vector
 *     Indices  9–11:  δb_g  gyroscope bias error [rad/s]
 *     Indices 12–14:  δb_a  accelerometer bias error [m/s²]
 *
 *   Nominal state (not directly accessible as KF state):
 *     p_ned_m       (3,) float64  NED position [m]
 *     v_ned_m_per_s (3,) float64  NED velocity [m/s]
 *     q_body_from_ned (4,) float64  attitude quaternion [w,x,y,z]
 *     bias_gyro_rad_per_s (3,) float64
 *     bias_accel_m_per_s2 (3,) float64
 *
 * ─── Worked predict example ─────────────────────────────────────────────────
 *
 *   At rest, q = identity, zero biases, zero velocity, g_ned = [0,0,9.81]:
 *
 *   IMU reads specific force f = a − g.  At rest a = 0, so the accelerometer
 *   reads the "1 g up" reaction a_body = [0, 0, −9.81] m/s² (FRD: up is −body-z),
 *   ω_body = [0, 0, 0]. This is the physical convention real IMUs (and nav-sim)
 *   emit.
 *
 *   After mechanisation with dt=0.01 s:
 *     a_ned = R_ned_from_body · (a_body − b_a) + g_ned
 *           = I · ([0,0,−9.81] − 0) + [0,0,9.81] = [0,0,0]              ✓
 *     v_new = v + a_ned · dt = 0                                         ✓
 *     p_new = p + v · dt + 0.5 · a_ned · dt² = 0                        ✓
 *
 * ─── Worked update example (GNSS position, scalar in one axis) ──────────────
 *
 *   Error state δp_N = 2.0 m,  P[0][0] = 4.0 m².
 *   GNSS measurement: z = p_nom_N + 0 (unbiased), so innovation = 0 − 2.0 = −2.0.
 *   R = 1.0 m², H selects δp_N: H[0, 0..14] = [1,0,0,...].
 *   S = 4.0 + 1.0 = 5.0.
 *   K[0] = 4.0 / 5.0 = 0.8.
 *   δp_N ← 2.0 + 0.8·(−2.0) = 0.4 m.
 *   P[0][0] ← (1 − 0.8)·4.0 = 0.8 m².                                   ✓
 */

#pragma once
#include "quaternion.hpp"
#include "ekf.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace navcore {

/// Number of error-state dimensions (fixed across the toolkit).
inline constexpr int ESKF_STATE_DIM = 15;

/** Nominal navigation state (world = NED, body = FRD). */
struct NominalState {
    std::array<double, 3> position_ned_m{};
    std::array<double, 3> velocity_ned_m_per_s{};
    Quaternion q_body_from_ned{};  // [w,x,y,z], body-from-world
    std::array<double, 3> bias_gyro_rad_per_s{};
    std::array<double, 3> bias_accel_m_per_s2{};
};

/** Gravity vector in NED [0, 0, +g_m_per_s2]. */
inline constexpr double DEFAULT_GRAVITY_M_PER_S2 = 9.80665;

/**
 * 15-state Error-State Kalman Filter for aided inertial navigation (INS/GNSS).
 * Its predict step runs the strapdown inertial mechanisation; its updates fuse
 * external aiding measurements.
 *
 * Call sequence:
 *   1. Construct with initial nominal state and error covariance.
 *   2. For each IMU sample: call predict(imu_gyro, imu_accel, dt_s, Q).
 *   3. For each external measurement: call update(z, H, R, M).
 *   4. Read nominal_state() and error_covariance() at any time.
 *   5. Call reset_error_state() after each update (done automatically by update()).
 */
class Eskf {
public:
    using ErrorStateMat = std::array<double, ESKF_STATE_DIM * ESKF_STATE_DIM>;
    using ErrorStateVec = std::array<double, ESKF_STATE_DIM>;

    /**
     * Construct an ESKF.
     *
     * @param initial_nominal   Starting nominal navigation state.
     * @param initial_P         15×15 initial error covariance, row-major.
     * @param gravity_m_per_s2  Gravity magnitude [m/s²], default WGS-84 standard.
     */
    Eskf(const NominalState& initial_nominal,
         const ErrorStateMat& initial_P,
         double gravity_m_per_s2 = DEFAULT_GRAVITY_M_PER_S2)
        : nominal_(initial_nominal),
          ekf_(ErrorStateVec{}, initial_P),
          gravity_m_per_s2_(gravity_m_per_s2) {}

    // ------------------------------------------------------------------ //
    // Accessors                                                           //
    // ------------------------------------------------------------------ //

    [[nodiscard]] const NominalState& nominal_state()       const noexcept { return nominal_; }
    [[nodiscard]] const ErrorStateMat& error_covariance()   const noexcept { return ekf_.covariance(); }
    [[nodiscard]] const std::vector<double>& innovation()   const noexcept { return ekf_.innovation(); }
    [[nodiscard]] const std::vector<double>& innovation_covariance() const noexcept {
        return ekf_.innovation_covariance();
    }

    // ------------------------------------------------------------------ //
    // Predict — IMU mechanisation + error-state F propagation            //
    // ------------------------------------------------------------------ //

    /**
     * Propagate the nominal state and error covariance by one IMU step.
     *
     * Nominal mechanisation (exact, not linearised):
     *   1. Remove biases from IMU readings.
     *   2. Integrate quaternion by angular velocity (1st-order Bortz equation).
     *   3. Rotate specific force to NED, add gravity (a = f + g), integrate velocity.
     *   4. Integrate position.
     *
     * Error-state Jacobian F (15×15) linearised around the nominal.
     * Process noise Q supplied by the caller (sensor noise parameters).
     *
     * @param gyro_body_rad_per_s   Raw gyro measurement [rad/s] in body frame.
     * @param accel_body_m_per_s2   Raw accel measurement [m/s²] in body frame.
     * @param dt_s                  Time step [seconds] — kept as double for precision.
     * @param Q_row_major           15×15 process noise covariance, row-major.
     */
    void predict(const std::array<double, 3>& gyro_body_rad_per_s,
                 const std::array<double, 3>& accel_body_m_per_s2,
                 double dt_s,
                 const ErrorStateMat& Q_row_major) {
        // --- 1. Debias ---
        const std::array<double, 3> omega{
            gyro_body_rad_per_s[0] - nominal_.bias_gyro_rad_per_s[0],
            gyro_body_rad_per_s[1] - nominal_.bias_gyro_rad_per_s[1],
            gyro_body_rad_per_s[2] - nominal_.bias_gyro_rad_per_s[2],
        };
        const std::array<double, 3> f_body{
            accel_body_m_per_s2[0] - nominal_.bias_accel_m_per_s2[0],
            accel_body_m_per_s2[1] - nominal_.bias_accel_m_per_s2[1],
            accel_body_m_per_s2[2] - nominal_.bias_accel_m_per_s2[2],
        };

        // --- 2. Quaternion integration (1st-order) ---
        // dq/dt = 0.5 * Ξ(q) * ω_body  →  q_new = q * exp(0.5·ω·dt)
        const double half_dt = 0.5 * dt_s;
        const Quaternion dq{
            1.0,
            omega[0] * half_dt,
            omega[1] * half_dt,
            omega[2] * half_dt,
        };
        const Quaternion q_new = (nominal_.q_body_from_ned * dq).normalised();

        // --- 3. Specific force in NED  ---
        // quaternion_to_dcm(q) is the body→world (NED-from-body) rotation — it maps a
        // body vector into NED: rotate_vector(q, v_body) == R · v_body. So the body
        // specific force rotates into NED as f_ned = R · f_body (NO transpose). [NAV-021]
        const auto R_ned_from_body = quaternion_to_dcm(nominal_.q_body_from_ned);
        const std::array<double, 3> f_ned{
            R_ned_from_body[0][0]*f_body[0] + R_ned_from_body[0][1]*f_body[1] + R_ned_from_body[0][2]*f_body[2],
            R_ned_from_body[1][0]*f_body[0] + R_ned_from_body[1][1]*f_body[1] + R_ned_from_body[1][2]*f_body[2],
            R_ned_from_body[2][0]*f_body[0] + R_ned_from_body[2][1]*f_body[1] + R_ned_from_body[2][2]*f_body[2],
        };
        // Specific force is the PHYSICAL accelerometer convention f = a − g (the
        // accelerometer measures non-gravitational proper acceleration), so the
        // kinematic acceleration is a = f + g, with g_ned = [0, 0, +g] (NED,
        // Down-positive). At rest f = −g (the "1 g up" reaction) and a = 0.
        const std::array<double, 3> a_ned{
            f_ned[0],
            f_ned[1],
            f_ned[2] + gravity_m_per_s2_,
        };

        // --- 4. Velocity and position integration ---
        const std::array<double, 3> v_new{
            nominal_.velocity_ned_m_per_s[0] + a_ned[0] * dt_s,
            nominal_.velocity_ned_m_per_s[1] + a_ned[1] * dt_s,
            nominal_.velocity_ned_m_per_s[2] + a_ned[2] * dt_s,
        };
        const std::array<double, 3> p_new{
            nominal_.position_ned_m[0] + nominal_.velocity_ned_m_per_s[0] * dt_s + 0.5 * a_ned[0] * dt_s * dt_s,
            nominal_.position_ned_m[1] + nominal_.velocity_ned_m_per_s[1] * dt_s + 0.5 * a_ned[1] * dt_s * dt_s,
            nominal_.position_ned_m[2] + nominal_.velocity_ned_m_per_s[2] * dt_s + 0.5 * a_ned[2] * dt_s * dt_s,
        };

        // Commit nominal state
        nominal_.position_ned_m         = p_new;
        nominal_.velocity_ned_m_per_s   = v_new;
        nominal_.q_body_from_ned         = q_new;
        // Biases are modelled as random walk: bias stays unchanged in nominal.

        // --- 5. Error-state Jacobian F ---
        // Linearisation of the 15-state error dynamics around the nominal.
        // Sub-blocks (using Sola 2017 notation):
        //   F_pp = I + 0 (position ← velocity)
        //   F_pv = I·dt
        //   F_vv = I
        //   F_vψ = −R·[f_body]× · dt  (δψ is body-side; NAV-022: was the
        //                              old-scheme +[f_ned]×)
        //   F_vb_a = −R · dt       (R = body→world; NAV-021: was wrongly Rᵀ)
        //   F_ψψ = I − [ω]× · dt
        //   F_ψb_g = −I · dt
        ErrorStateMat F_row_major{};
        // Identity blocks
        for (int i = 0; i < ESKF_STATE_DIM; ++i) F_row_major[i * ESKF_STATE_DIM + i] = 1.0;

        // F_pv = I·dt  (rows 0..2, cols 3..5)
        for (int i = 0; i < 3; ++i) F_row_major[i * ESKF_STATE_DIM + (3 + i)] = dt_s;

        // F_vψ = −R·[f_body]×·dt  (rows 3..5, cols 6..8).  δψ is the BODY-side
        // attitude error (the injection is q ⊗ δq), so R_true = R·(I + [δψ]×) and
        // δv̇ = R·[δψ]×·f_body = −R·[f_body]×·δψ  (Sola 2017 eq. 270c).
        // Under the pre-NAV-021 transposed scheme the correct block was
        // +[f_ned]×·dt; NAV-021 fixed the mechanisation and F_vb_a but left this
        // block at the old-scheme value, so velocity↔attitude cross-covariance
        // built with the wrong sign and every aiding update pushed roll/pitch
        // away from truth. [NAV-022]
        // skew([ax,ay,az]) = [[0,-az,ay],[az,0,-ax],[-ay,ax,0]]
        const double fbx = f_body[0], fby = f_body[1], fbz = f_body[2];
        const std::array<std::array<double, 3>, 3> skew_f_body{{
            {{0.0, -fbz, fby}},
            {{fbz, 0.0, -fbx}},
            {{-fby, fbx, 0.0}},
        }};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                double r_skew_ij = 0.0;
                for (int k = 0; k < 3; ++k)
                    r_skew_ij += R_ned_from_body[i][k] * skew_f_body[k][j];
                F_row_major[(3 + i) * ESKF_STATE_DIM + (6 + j)] = -r_skew_ij * dt_s;
            }

        // F_vb_a = −R · dt  (rows 3..5, cols 12..14): a body-frame accel-bias error maps
        // into NED velocity error through the same body→world rotation R.  [NAV-021]
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                F_row_major[(3 + i) * ESKF_STATE_DIM + (12 + j)] = -R_ned_from_body[i][j] * dt_s;

        // F_ψψ = I − skew(ω) · dt  (rows 6..8, cols 6..8)  — already I from above
        const double ox = omega[0], oy = omega[1], oz = omega[2];
        F_row_major[6 * ESKF_STATE_DIM + 7] =  oz * dt_s;
        F_row_major[6 * ESKF_STATE_DIM + 8] = -oy * dt_s;
        F_row_major[7 * ESKF_STATE_DIM + 6] = -oz * dt_s;
        F_row_major[7 * ESKF_STATE_DIM + 8] =  ox * dt_s;
        F_row_major[8 * ESKF_STATE_DIM + 6] =  oy * dt_s;
        F_row_major[8 * ESKF_STATE_DIM + 7] = -ox * dt_s;

        // F_ψb_g = −I·dt  (rows 6..8, cols 9..11)
        for (int i = 0; i < 3; ++i)
            F_row_major[(6 + i) * ESKF_STATE_DIM + (9 + i)] = -dt_s;

        // Propagate error covariance (error state stays zero after each reset)
        ekf_.predict(F_row_major, Q_row_major);
    }

    // ------------------------------------------------------------------ //
    // Update — generic measurement of any dimension M                    //
    // ------------------------------------------------------------------ //

    /**
     * ESKF update: fuse an external measurement with the error state.
     *
     * The innovation is formed against the nominal state.  The caller computes
     * the residual z (measurement minus nominal prediction) and linearises the
     * measurement wrt the error state as H.
     *
     * After the linear KF update the estimated error is injected into the
     * nominal state and the error state is reset to zero.
     *
     * @param innovation_z    Residual: z_measured − h(x_nominal), length M.
     * @param H_row_major     Error-state measurement Jacobian, M×15 row-major.
     * @param R_row_major     Measurement noise covariance, M×M row-major.
     * @param M               Measurement dimension.
     */
    void update(const std::vector<double>& innovation_z,
                const std::vector<double>& H_row_major,
                const std::vector<double>& R_row_major,
                int M) {
        ekf_.update(innovation_z, H_row_major, R_row_major, M);
        inject_and_reset_error_state();
    }

    /**
     * Convenience: GNSS position-only update.
     *
     * Forms the innovation as p_gnss_ned − p_nominal_ned and sets H to select
     * the position sub-block of the error state (rows selecting δp).
     *
     * @param gnss_position_ned_m  GNSS-derived NED position [m].
     * @param position_cov_ned_m2  3×3 position covariance [m²], row-major.
     */
    void update_gnss_position(const std::array<double, 3>& gnss_position_ned_m,
                               const std::array<double, 3>& position_cov_ned_m2_diag) {
        // Innovation: z = p_gnss − p_nominal
        std::vector<double> z = {
            gnss_position_ned_m[0] - nominal_.position_ned_m[0],
            gnss_position_ned_m[1] - nominal_.position_ned_m[1],
            gnss_position_ned_m[2] - nominal_.position_ned_m[2],
        };
        // H: 3×15, selects δp (cols 0,1,2)
        std::vector<double> H(3 * ESKF_STATE_DIM, 0.0);
        H[0 * ESKF_STATE_DIM + 0] = 1.0;
        H[1 * ESKF_STATE_DIM + 1] = 1.0;
        H[2 * ESKF_STATE_DIM + 2] = 1.0;
        // R: 3×3 diagonal
        std::vector<double> R(9, 0.0);
        R[0] = position_cov_ned_m2_diag[0];
        R[4] = position_cov_ned_m2_diag[1];
        R[8] = position_cov_ned_m2_diag[2];
        ekf_.update(z, H, R, 3);
        inject_and_reset_error_state();
    }

private:
    NominalState nominal_;
    Ekf<ESKF_STATE_DIM> ekf_;
    double gravity_m_per_s2_;

    /** Inject estimated error into nominal state, then re-zero error state. */
    void inject_and_reset_error_state() {
        const auto& dx = ekf_.state();

        // δp injection
        for (int i = 0; i < 3; ++i) nominal_.position_ned_m[i]       += dx[i];
        // δv injection
        for (int i = 0; i < 3; ++i) nominal_.velocity_ned_m_per_s[i] += dx[3 + i];
        // δψ injection: q_new = q * Δq where Δq ≈ [1, δψ/2]
        const Quaternion dq_err{
            1.0,
            0.5 * dx[6],
            0.5 * dx[7],
            0.5 * dx[8],
        };
        nominal_.q_body_from_ned = (nominal_.q_body_from_ned * dq_err).normalised();
        // δb_g injection
        for (int i = 0; i < 3; ++i) nominal_.bias_gyro_rad_per_s[i]  += dx[9  + i];
        // δb_a injection
        for (int i = 0; i < 3; ++i) nominal_.bias_accel_m_per_s2[i]  += dx[12 + i];

        // Reset error state to zero (P stays as is — it now represents
        // residual uncertainty around the updated nominal).
        // We do this by resetting the Ekf's internal state vector.
        // The Ekf<N> does not expose a direct reset; rebuild with same P.
        const auto P = ekf_.covariance();
        ekf_ = Ekf<ESKF_STATE_DIM>(ErrorStateVec{}, P);
    }
};

} // namespace navcore
