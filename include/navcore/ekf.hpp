/**
 * @file ekf.hpp
 * @brief Linear Extended Kalman Filter: predict / update with general process
 *        and measurement models.
 *
 * This is a generic, dimension-templated EKF.  "Linear" here means we supply
 * the Jacobians (F, H) explicitly; the caller is responsible for linearising
 * nonlinear dynamics.  For the fully-nonlinear IMU case use eskf.hpp.
 *
 * ─── Toy worked example (1-D, verify by hand) ───────────────────────────────
 *
 *   State: x = position (scalar), P = variance.
 *   Initial: x₀ = 0, P₀ = 1.
 *
 *   PREDICT (constant velocity, dt=1, Q = 0.1):
 *     F = [[1]], G = [[1]], Q = [[0.1]]
 *     x⁻ = F·x = 0
 *     P⁻ = F·P·Fᵀ + Q = 1·1·1 + 0.1 = 1.1
 *
 *   UPDATE (direct position measurement z=0.5, R=0.25):
 *     H = [[1]]
 *     S = H·P⁻·Hᵀ + R = 1.1 + 0.25 = 1.35
 *     K = P⁻·Hᵀ·S⁻¹ = 1.1 / 1.35 ≈ 0.8148
 *     ν = z − H·x⁻ = 0.5 − 0 = 0.5
 *     x⁺ = x⁻ + K·ν = 0 + 0.8148·0.5 ≈ 0.4074
 *     P⁺ = (I − K·H)·P⁻ = (1 − 0.8148)·1.1 ≈ 0.2037                   ✓
 *
 *   These numbers are what the test suite checks.
 *
 * ─── General matrix-form EKF ─────────────────────────────────────────────────
 *
 *   Predict:
 *     x⁻ = F · x
 *     P⁻ = F · P · Fᵀ + Q
 *
 *   Update:
 *     ν   = z − H · x⁻
 *     S   = H · P⁻ · Hᵀ + R
 *     K   = P⁻ · Hᵀ · S⁻¹
 *     x⁺  = x⁻ + K · ν
 *     P⁺  = (I − K·H) · P⁻    [Joseph form for numerical stability when needed]
 *
 * ─── Representation ──────────────────────────────────────────────────────────
 *
 *   All matrices are stored as 1-D row-major std::vector<double> of size N×N (or
 *   N×M).  The template parameter N is the state dimension; M is the measurement
 *   dimension (supplied per update call).
 *
 *   Caller supplies F, Q, H, R as flat row-major arrays.  The EKF does not
 *   allocate on the hot path; the caller is responsible for buffers.
 *
 *   Innovations ν and innovation covariance S are stored after each update for
 *   downstream consistency checks (NIS = νᵀ S⁻¹ ν should be χ²(M)).
 */

#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace navcore {

/**
 * General-purpose EKF with state dimension N (compile-time constant).
 *
 * Template parameter N: state vector length.
 */
template <int N>
class Ekf {
public:
    using StateVec  = std::array<double, N>;
    using StateMat  = std::array<double, N * N>;  // row-major N×N

    /** Initialise with a state vector and covariance matrix (row-major N×N). */
    Ekf(const StateVec& initial_state, const StateMat& initial_covariance)
        : state_(initial_state), covariance_(initial_covariance) {}

    // ------------------------------------------------------------------ //
    // Accessors                                                           //
    // ------------------------------------------------------------------ //

    [[nodiscard]] const StateVec& state()      const noexcept { return state_; }
    [[nodiscard]] const StateMat& covariance() const noexcept { return covariance_; }

    /** Last innovation vector (length M of most recent update call). */
    [[nodiscard]] const std::vector<double>& innovation()            const noexcept { return innovation_; }
    /** Last innovation covariance (row-major M×M). */
    [[nodiscard]] const std::vector<double>& innovation_covariance() const noexcept { return innovation_cov_; }

    // ------------------------------------------------------------------ //
    // Predict                                                             //
    // ------------------------------------------------------------------ //

    /**
     * EKF prediction step.
     *
     *   x⁻ = F · x
     *   P⁻ = F · P · Fᵀ + Q
     *
     * @param F_row_major  State transition Jacobian, N×N row-major.
     * @param Q_row_major  Process noise covariance, N×N row-major.
     */
    void predict(const StateMat& F_row_major, const StateMat& Q_row_major) {
        // x⁻ = F * x
        StateVec x_new{};
        for (int i = 0; i < N; ++i) {
            double s = 0.0;
            for (int k = 0; k < N; ++k) s += F_row_major[i * N + k] * state_[k];
            x_new[i] = s;
        }
        state_ = x_new;

        // P⁻ = F * P * Fᵀ + Q
        // Tmp = F * P
        StateMat tmp{};
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                double s = 0.0;
                for (int k = 0; k < N; ++k) s += F_row_major[i * N + k] * covariance_[k * N + j];
                tmp[i * N + j] = s;
            }
        // P⁻ = Tmp * Fᵀ + Q
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                double s = 0.0;
                for (int k = 0; k < N; ++k) s += tmp[i * N + k] * F_row_major[j * N + k];
                covariance_[i * N + j] = s + Q_row_major[i * N + j];
            }
    }

    // ------------------------------------------------------------------ //
    // Update                                                              //
    // ------------------------------------------------------------------ //

    /**
     * EKF update step for a measurement of dimension M.
     *
     *   ν   = z − H · x⁻
     *   S   = H · P⁻ · Hᵀ + R
     *   K   = P⁻ · Hᵀ · S⁻¹
     *   x⁺  = x⁻ + K · ν
     *   P⁺  = (I − K·H) · P⁻
     *
     * Innovation ν and S are stored for NIS computation.
     *
     * @param z_meas         Measurement vector, length M.
     * @param H_row_major    Measurement Jacobian, M×N row-major.
     * @param R_row_major    Measurement noise covariance, M×M row-major.
     * @param M              Measurement dimension.
     */
    void update(const std::vector<double>& z_meas,
                const std::vector<double>& H_row_major,
                const std::vector<double>& R_row_major,
                int M) {
        if (static_cast<int>(z_meas.size()) != M)
            throw std::invalid_argument("navcore::Ekf::update: z_meas size != M");

        // ν = z − H·x⁻
        innovation_.resize(M);
        for (int i = 0; i < M; ++i) {
            double Hx = 0.0;
            for (int k = 0; k < N; ++k) Hx += H_row_major[i * N + k] * state_[k];
            innovation_[i] = z_meas[i] - Hx;
        }

        // S = H · P · Hᵀ + R
        // HP = H (M×N) * P (N×N) → M×N
        std::vector<double> HP(M * N, 0.0);
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j) {
                double s = 0.0;
                for (int k = 0; k < N; ++k) s += H_row_major[i * N + k] * covariance_[k * N + j];
                HP[i * N + j] = s;
            }
        // S = HP * Hᵀ + R  (M×M)
        innovation_cov_.resize(M * M);
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < M; ++j) {
                double s = 0.0;
                for (int k = 0; k < N; ++k) s += HP[i * N + k] * H_row_major[j * N + k];
                innovation_cov_[i * M + j] = s + R_row_major[i * M + j];
            }

        // S⁻¹ via Cholesky (for symmetric positive-definite S)
        std::vector<double> S_inv = cholesky_inverse(innovation_cov_, M);

        // K = P · Hᵀ · S⁻¹
        // PHt = P (N×N) * Hᵀ (N×M) → N×M
        std::vector<double> PHt(N * M, 0.0);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < M; ++j) {
                double s = 0.0;
                for (int k = 0; k < N; ++k) s += covariance_[i * N + k] * H_row_major[j * N + k];
                PHt[i * M + j] = s;
            }
        // K = PHt * S⁻¹  (N×M)
        std::vector<double> K(N * M, 0.0);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < M; ++j) {
                double s = 0.0;
                for (int k = 0; k < M; ++k) s += PHt[i * M + k] * S_inv[k * M + j];
                K[i * M + j] = s;
            }

        // x⁺ = x⁻ + K · ν
        for (int i = 0; i < N; ++i) {
            double s = 0.0;
            for (int k = 0; k < M; ++k) s += K[i * M + k] * innovation_[k];
            state_[i] += s;
        }

        // P⁺ = (I − K·H) · P
        // KH = K (N×M) * H (M×N) → N×N
        StateMat KH{};
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                double s = 0.0;
                for (int k = 0; k < M; ++k) s += K[i * M + k] * H_row_major[k * N + j];
                KH[i * N + j] = s;
            }
        // IKH = I − KH
        StateMat IKH{};
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                IKH[i * N + j] = (i == j ? 1.0 : 0.0) - KH[i * N + j];
        // P⁺ = IKH * P
        StateMat P_new{};
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                double s = 0.0;
                for (int k = 0; k < N; ++k) s += IKH[i * N + k] * covariance_[k * N + j];
                P_new[i * N + j] = s;
            }
        covariance_ = P_new;
    }

private:
    StateVec state_;
    StateMat covariance_;
    std::vector<double> innovation_;
    std::vector<double> innovation_cov_;

    /**
     * Invert a symmetric positive-definite matrix via Cholesky decomposition.
     *
     * @param A_row_major  Row-major M×M SPD matrix.
     * @param M            Matrix dimension.
     * @returns  Row-major M×M inverse.
     */
    static std::vector<double> cholesky_inverse(const std::vector<double>& A_row_major, int M) {
        // Lower-triangular Cholesky L such that A = L Lᵀ
        std::vector<double> L(M * M, 0.0);
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j <= i; ++j) {
                double s = A_row_major[i * M + j];
                for (int k = 0; k < j; ++k) s -= L[i * M + k] * L[j * M + k];
                if (i == j) {
                    if (s < 0.0) s = 0.0;  // clamp floating-point noise
                    L[i * M + i] = std::sqrt(s);
                } else {
                    L[i * M + j] = (L[j * M + j] > 1e-300) ? s / L[j * M + j] : 0.0;
                }
            }
        }
        // Forward substitution: solve L * Y = I
        std::vector<double> Y(M * M, 0.0);
        for (int col = 0; col < M; ++col) {
            for (int i = 0; i < M; ++i) {
                double s = (i == col) ? 1.0 : 0.0;
                for (int k = 0; k < i; ++k) s -= L[i * M + k] * Y[k * M + col];
                Y[i * M + col] = (L[i * M + i] > 1e-300) ? s / L[i * M + i] : 0.0;
            }
        }
        // Backward substitution: solve Lᵀ * X = Y
        std::vector<double> X(M * M, 0.0);
        for (int col = 0; col < M; ++col) {
            for (int i = M - 1; i >= 0; --i) {
                double s = Y[i * M + col];
                for (int k = i + 1; k < M; ++k) s -= L[k * M + i] * X[k * M + col];
                X[i * M + col] = (L[i * M + i] > 1e-300) ? s / L[i * M + i] : 0.0;
            }
        }
        return X;
    }
};

} // namespace navcore
