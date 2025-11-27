/*
 * Copyright (C) 2024 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * Before/After Benchmark Comparison for Optimizations
 */

#include <albatross/Core>
#include <albatross/CovarianceFunctions>
#include <albatross/GP>
#include <chrono>
#include <iostream>
#include <iomanip>

namespace albatross {

class Timer {
public:
  using Clock = std::chrono::high_resolution_clock;

  void start() { start_ = Clock::now(); }

  double elapsed_ms() const {
    auto end = Clock::now();
    return std::chrono::duration<double, std::milli>(end - start_).count();
  }

private:
  Clock::time_point start_;
};

// OLD IMPLEMENTATION: Branching version (BEFORE optimization)
Eigen::DiagonalMatrix<double, Eigen::Dynamic> diagonal_sqrt_inverse_OLD(const Eigen::VectorXd& vectorD) {
  Eigen::VectorXd thresholded_diag_sqrt_inverse = vectorD;

  // Branching version - prevents SIMD vectorization
  for (Eigen::Index i = 0; i < thresholded_diag_sqrt_inverse.size(); ++i) {
    if (thresholded_diag_sqrt_inverse[i] > 0.) {
      thresholded_diag_sqrt_inverse[i] =
          1. / std::sqrt(thresholded_diag_sqrt_inverse[i]);
    } else {
      thresholded_diag_sqrt_inverse[i] = 0.;
    }
  }

  return thresholded_diag_sqrt_inverse.asDiagonal();
}

// NEW IMPLEMENTATION: Branchless version (AFTER optimization)
Eigen::DiagonalMatrix<double, Eigen::Dynamic> diagonal_sqrt_inverse_NEW(const Eigen::VectorXd& vectorD) {
  Eigen::VectorXd thresholded_diag_sqrt_inverse = vectorD;

  // Branchless version: enables SIMD auto-vectorization
  thresholded_diag_sqrt_inverse =
      (thresholded_diag_sqrt_inverse.array() > 0.0)
      .select(1.0 / thresholded_diag_sqrt_inverse.cwiseSqrt().array(), 0.0);

  return thresholded_diag_sqrt_inverse.asDiagonal();
}

// OLD IMPLEMENTATION: Row-major loop (BEFORE optimization)
template<typename CovFunc>
Eigen::MatrixXd compute_covariance_OLD(const CovFunc& cov_func,
                                       const std::vector<double>& xs,
                                       const std::vector<double>& ys) {
  const Eigen::Index m = static_cast<Eigen::Index>(xs.size());
  const Eigen::Index n = static_cast<Eigen::Index>(ys.size());
  Eigen::MatrixXd C(m, n);

  // Row-major loop: poor cache locality for column-major storage
  for (Eigen::Index i = 0; i < m; i++) {
    for (Eigen::Index j = 0; j < n; j++) {
      C(i, j) = cov_func(xs[static_cast<std::size_t>(i)],
                         ys[static_cast<std::size_t>(j)]);
    }
  }
  return C;
}

// NEW IMPLEMENTATION: Column-major loop (AFTER optimization)
template<typename CovFunc>
Eigen::MatrixXd compute_covariance_NEW(const CovFunc& cov_func,
                                       const std::vector<double>& xs,
                                       const std::vector<double>& ys) {
  const Eigen::Index m = static_cast<Eigen::Index>(xs.size());
  const Eigen::Index n = static_cast<Eigen::Index>(ys.size());
  Eigen::MatrixXd C(m, n);

  // Column-major loop: sequential writes optimize cache utilization
  for (Eigen::Index j = 0; j < n; j++) {
    const auto& y = ys[static_cast<std::size_t>(j)];
    for (Eigen::Index i = 0; i < m; i++) {
      C(i, j) = cov_func(xs[static_cast<std::size_t>(i)], y);
    }
  }
  return C;
}

void benchmark_comparison() {
  std::cout << "\n";
  std::cout << "================================================================\n";
  std::cout << "     BEFORE/AFTER OPTIMIZATION COMPARISON                      \n";
  std::cout << "================================================================\n";

  // ===== BENCHMARK 1: Diagonal Square Root Inverse =====
  std::cout << "\n========================================\n";
  std::cout << "Test 1: Diagonal Square Root Inverse\n";
  std::cout << "========================================\n\n";

  const int n = 5000;
  const int iterations = 1000;

  Eigen::VectorXd test_vector = Eigen::VectorXd::Random(n).array().abs() + 0.1;

  Timer timer;

  // BEFORE: Branching version
  std::cout << "BEFORE (branching loop):\n";
  timer.start();
  for (int i = 0; i < iterations; ++i) {
    auto result = diagonal_sqrt_inverse_OLD(test_vector);
    volatile double sum = result.diagonal().sum();
    (void)sum;
  }
  double time_before = timer.elapsed_ms();
  std::cout << "  Time: " << std::fixed << std::setprecision(2) << time_before << " ms\n";
  std::cout << "  Rate: " << std::fixed << std::setprecision(0)
            << (iterations * 1000.0 / time_before) << " ops/sec\n\n";

  // AFTER: Branchless version
  std::cout << "AFTER (branchless SIMD):\n";
  timer.start();
  for (int i = 0; i < iterations; ++i) {
    auto result = diagonal_sqrt_inverse_NEW(test_vector);
    volatile double sum = result.diagonal().sum();
    (void)sum;
  }
  double time_after = timer.elapsed_ms();
  std::cout << "  Time: " << std::fixed << std::setprecision(2) << time_after << " ms\n";
  std::cout << "  Rate: " << std::fixed << std::setprecision(0)
            << (iterations * 1000.0 / time_after) << " ops/sec\n\n";

  double speedup1 = time_before / time_after;
  std::cout << "*** SPEEDUP: " << std::fixed << std::setprecision(2)
            << speedup1 << "x faster ***\n";

  // ===== BENCHMARK 2: Covariance Matrix Computation =====
  std::cout << "\n========================================\n";
  std::cout << "Test 2: Covariance Matrix Computation\n";
  std::cout << "========================================\n\n";

  const int n_train = 500;
  const int n_test = 200;
  const int cov_iterations = 20;

  std::vector<double> train_features;
  std::vector<double> test_features;
  for (int i = 0; i < n_train; ++i) {
    train_features.push_back(i * 0.1);
  }
  for (int i = 0; i < n_test; ++i) {
    test_features.push_back(i * 0.1);
  }

  auto cov_func = SquaredExponential<EuclideanDistance>();

  // BEFORE: Row-major loop
  std::cout << "BEFORE (row-major loop):\n";
  timer.start();
  for (int i = 0; i < cov_iterations; ++i) {
    auto C = compute_covariance_OLD(cov_func, train_features, test_features);
    volatile double sum = C.sum();
    (void)sum;
  }
  time_before = timer.elapsed_ms();
  std::cout << "  Time: " << std::fixed << std::setprecision(2) << time_before << " ms\n";
  std::cout << "  Rate: " << std::fixed << std::setprecision(1)
            << (cov_iterations * 1000.0 / time_before) << " matrices/sec\n\n";

  // AFTER: Column-major loop
  std::cout << "AFTER (column-major loop):\n";
  timer.start();
  for (int i = 0; i < cov_iterations; ++i) {
    auto C = compute_covariance_NEW(cov_func, train_features, test_features);
    volatile double sum = C.sum();
    (void)sum;
  }
  time_after = timer.elapsed_ms();
  std::cout << "  Time: " << std::fixed << std::setprecision(2) << time_after << " ms\n";
  std::cout << "  Rate: " << std::fixed << std::setprecision(1)
            << (cov_iterations * 1000.0 / time_after) << " matrices/sec\n\n";

  double speedup2 = time_before / time_after;
  std::cout << "*** SPEEDUP: " << std::fixed << std::setprecision(2)
            << speedup2 << "x faster ***\n";

  // ===== SUMMARY =====
  std::cout << "\n================================================================\n";
  std::cout << "                    SUMMARY                                     \n";
  std::cout << "================================================================\n\n";
  std::cout << "Optimization 1 (Branchless SIMD):       " << std::fixed << std::setprecision(2)
            << speedup1 << "x speedup\n";
  std::cout << "Optimization 2 (Cache-optimized loops): " << std::fixed << std::setprecision(2)
            << speedup2 << "x speedup\n\n";
  std::cout << "Combined expected improvement: " << std::fixed << std::setprecision(1)
            << ((speedup1 - 1) * 100) << "% + " << ((speedup2 - 1) * 100) << "%\n\n";
  std::cout << "================================================================\n\n";
}

} // namespace albatross

int main() {
  albatross::benchmark_comparison();
  return 0;
}
