/*
 * Copyright (C) 2024 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * Benchmark for Direct Diagonal Inverse Optimization
 */

#include <albatross/Core>
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

void benchmark_diagonal_inverse() {
  std::cout << "\n";
  std::cout << "================================================================\n";
  std::cout << "     DIAGONAL INVERSE OPTIMIZATION BENCHMARK                   \n";
  std::cout << "================================================================\n";
  std::cout << "\nMeasuring O(n²) direct diagonal inverse algorithm performance\n";
  std::cout << "and verifying correctness against full matrix inverse.\n";

  const std::vector<int> sizes = {100, 200, 500, 1000, 2000};

  std::cout << "\n";
  std::cout << "Matrix Size | Time/Op | Ops/sec | Error vs Full Inverse\n";
  std::cout << "------------|---------|---------|----------------------\n";

  for (int n : sizes) {
    // Create a positive definite matrix via A^T A
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(n, n);
    Eigen::MatrixXd M = A.transpose() * A;
    M.diagonal().array() += 1.0;  // Ensure positive definite

    // Decompose
    Eigen::SerializableLDLT ldlt;
    ldlt.compute(M);

    Timer timer;

    // Benchmark the O(n²) algorithm
    const int iterations = (n <= 500) ? 10 : 3;
    timer.start();
    for (int i = 0; i < iterations; ++i) {
      auto diag = ldlt.inverse_diagonal();
      volatile double sum = diag.sum();
      (void)sum;
    }
    double time = timer.elapsed_ms() / iterations;

    // Verify correctness (for smaller sizes)
    double error = 0.0;
    if (n <= 500) {
      Eigen::MatrixXd M_inv = M.inverse();
      Eigen::VectorXd diag_ldlt = ldlt.inverse_diagonal();
      Eigen::VectorXd diag_direct = M_inv.diagonal();
      error = (diag_ldlt - diag_direct).norm() / diag_direct.norm();
    }

    std::cout << std::setw(11) << n << " | ";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << std::setw(7) << time << " ms | ";
    std::cout << std::fixed << std::setprecision(1);
    std::cout << std::setw(7) << (1000.0 / time) << " | ";

    if (n <= 500) {
      std::cout << std::scientific << std::setprecision(2) << error;
      if (error < 1e-6) {
        std::cout << " (PASS)";
      } else {
        std::cout << " (WARN)";
      }
    } else {
      std::cout << "not tested";
    }
    std::cout << "\n";
  }

  std::cout << "\n================================================================\n";
  std::cout << "  Algorithm Complexity: O(n²) vs O(n³) for full inverse\n";
  std::cout << "  Memory Efficiency: O(n) workspace vs O(n²) for full inverse\n";
  std::cout << "  Expected 3-5x speedup for n > 500 compared to old version\n";
  std::cout << "================================================================\n\n";
}

} // namespace albatross

int main() {
  albatross::benchmark_diagonal_inverse();
  return 0;
}
