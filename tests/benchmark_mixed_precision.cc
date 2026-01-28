/*
 * Copyright (C) 2024 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * Performance benchmarks for mixed-precision implementation.
 *
 * This benchmarks the actual speedup from using float vs double
 * in covariance function evaluations and matrix operations.
 */

#include <albatross/Core>
#include <albatross/CovarianceFunctions>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>

namespace albatross {

// Simple timing utility
class Timer {
public:
  using Clock = std::chrono::high_resolution_clock;
  using TimePoint = Clock::time_point;

  void start() { start_ = Clock::now(); }

  double elapsed_ms() const {
    auto end = Clock::now();
    return std::chrono::duration<double, std::milli>(end - start_).count();
  }

private:
  TimePoint start_;
};

// Benchmark result
struct BenchmarkResult {
  std::string name;
  double time_ms;
  size_t iterations;

  double time_per_iteration_us() const {
    return (time_ms * 1000.0) / iterations;
  }
};

void print_header() {
  std::cout << "\n================================================================\n";
  std::cout << "         ALBATROSS MIXED-PRECISION BENCHMARK RESULTS            \n";
  std::cout << "================================================================\n\n";
}

void print_result(const BenchmarkResult& result) {
  std::cout << std::left << std::setw(50) << result.name << ": "
            << std::right << std::setw(10) << std::fixed << std::setprecision(2)
            << result.time_per_iteration_us() << " us/op"
            << "  (" << result.iterations << " iterations)\n";
}

void print_speedup(const BenchmarkResult& baseline, const BenchmarkResult& optimized) {
  double speedup = baseline.time_per_iteration_us() / optimized.time_per_iteration_us();
  std::cout << "  >> Speedup: " << std::fixed << std::setprecision(2)
            << speedup << "x faster";

  if (speedup >= 1.5) {
    std::cout << " +++ EXCELLENT\n";
  } else if (speedup >= 1.3) {
    std::cout << " ++ VERY GOOD\n";
  } else if (speedup >= 1.1) {
    std::cout << " + GOOD\n";
  } else if (speedup >= 1.0) {
    std::cout << " ~ MARGINAL\n";
  } else {
    std::cout << " X SLOWER\n";
  }
  std::cout << "\n";
}

void print_section(const std::string& title) {
  std::cout << "\n" << std::string(70, '-') << "\n";
  std::cout << "  " << title << "\n";
  std::cout << std::string(70, '-') << "\n";
}

// Benchmark 1: Covariance function evaluation
BenchmarkResult benchmark_covariance_function_double(size_t n_iterations) {
  Timer timer;
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(0.1, 10.0);

  // Parameters
  double length_scale = 100.0;
  double sigma = 2.0;

  // Generate random distances
  std::vector<double> distances(1000);
  for (auto& d : distances) {
    d = dist(rng);
  }

  timer.start();
  double sum = 0.0;
  for (size_t iter = 0; iter < n_iterations; ++iter) {
    for (const auto& distance : distances) {
      sum += squared_exponential_covariance<double>(distance, length_scale, sigma);
      sum += exponential_covariance<double>(distance, length_scale, sigma);
      sum += matern_32_covariance<double>(distance, length_scale, sigma);
      sum += matern_52_covariance<double>(distance, length_scale, sigma);
    }
  }

  // Prevent optimization
  volatile double result = sum;
  (void)result;

  return {"Covariance Functions (double)", timer.elapsed_ms(), n_iterations * 1000 * 4};
}

BenchmarkResult benchmark_covariance_function_float(size_t n_iterations) {
  Timer timer;
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(0.1f, 10.0f);

  // Parameters
  float length_scale = 100.0f;
  float sigma = 2.0f;

  // Generate random distances
  std::vector<float> distances(1000);
  for (auto& d : distances) {
    d = dist(rng);
  }

  timer.start();
  float sum = 0.0f;
  for (size_t iter = 0; iter < n_iterations; ++iter) {
    for (const auto& distance : distances) {
      sum += squared_exponential_covariance<float>(distance, length_scale, sigma);
      sum += exponential_covariance<float>(distance, length_scale, sigma);
      sum += matern_32_covariance<float>(distance, length_scale, sigma);
      sum += matern_52_covariance<float>(distance, length_scale, sigma);
    }
  }

  // Prevent optimization
  volatile float result = sum;
  (void)result;

  return {"Covariance Functions (float)", timer.elapsed_ms(), n_iterations * 1000 * 4};
}

// Benchmark 2: Precision conversion overhead
BenchmarkResult benchmark_precision_conversion_to_float(size_t n_iterations) {
  Timer timer;

  // Create a double precision vector
  Eigen::VectorXd vec_d = Eigen::VectorXd::Random(1000);

  timer.start();
  for (size_t iter = 0; iter < n_iterations; ++iter) {
    Eigen::VectorXf vec_f = convert_precision<float>(vec_d);
    // Prevent optimization
    volatile float sum = vec_f.sum();
    (void)sum;
  }

  return {"Precision Conversion (double→float)", timer.elapsed_ms(), n_iterations};
}

BenchmarkResult benchmark_precision_conversion_to_double(size_t n_iterations) {
  Timer timer;

  // Create a float precision vector
  Eigen::VectorXf vec_f = Eigen::VectorXf::Random(1000);

  timer.start();
  for (size_t iter = 0; iter < n_iterations; ++iter) {
    Eigen::VectorXd vec_d = convert_precision<double>(vec_f);
    // Prevent optimization
    volatile double sum = vec_d.sum();
    (void)sum;
  }

  return {"Precision Conversion (float→double)", timer.elapsed_ms(), n_iterations};
}

// Benchmark 3: Matrix operations
BenchmarkResult benchmark_matrix_multiply_double(size_t n_iterations) {
  Timer timer;

  Eigen::MatrixXd A = Eigen::MatrixXd::Random(200, 200);
  Eigen::MatrixXd B = Eigen::MatrixXd::Random(200, 200);

  timer.start();
  for (size_t iter = 0; iter < n_iterations; ++iter) {
    Eigen::MatrixXd C = A * B;
    // Prevent optimization
    volatile double sum = C.sum();
    (void)sum;
  }

  return {"Matrix Multiply 200x200 (double)", timer.elapsed_ms(), n_iterations};
}

BenchmarkResult benchmark_matrix_multiply_float(size_t n_iterations) {
  Timer timer;

  Eigen::MatrixXf A = Eigen::MatrixXf::Random(200, 200);
  Eigen::MatrixXf B = Eigen::MatrixXf::Random(200, 200);

  timer.start();
  for (size_t iter = 0; iter < n_iterations; ++iter) {
    Eigen::MatrixXf C = A * B;
    // Prevent optimization
    volatile float sum = C.sum();
    (void)sum;
  }

  return {"Matrix Multiply 200x200 (float)", timer.elapsed_ms(), n_iterations};
}

// Benchmark 4: exp() function (core of covariance functions)
BenchmarkResult benchmark_exp_double(size_t n_iterations) {
  Timer timer;
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(-5.0, 5.0);

  std::vector<double> values(1000);
  for (auto& v : values) {
    v = dist(rng);
  }

  timer.start();
  double sum = 0.0;
  for (size_t iter = 0; iter < n_iterations; ++iter) {
    for (const auto& val : values) {
      sum += std::exp(val);
    }
  }

  volatile double result = sum;
  (void)result;

  return {"std::exp() function (double)", timer.elapsed_ms(), n_iterations * 1000};
}

BenchmarkResult benchmark_exp_float(size_t n_iterations) {
  Timer timer;
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

  std::vector<float> values(1000);
  for (auto& v : values) {
    v = dist(rng);
  }

  timer.start();
  float sum = 0.0f;
  for (size_t iter = 0; iter < n_iterations; ++iter) {
    for (const auto& val : values) {
      sum += std::exp(val);
    }
  }

  volatile float result = sum;
  (void)result;

  return {"std::exp() function (float)", timer.elapsed_ms(), n_iterations * 1000};
}

void run_benchmarks() {
  print_header();

  std::cout << "Warming up CPU...\n";
  // Warm-up runs
  benchmark_covariance_function_double(100);
  benchmark_covariance_function_float(100);

  std::cout << "Running benchmarks (this may take a minute)...\n";

  // Section 1: Covariance Functions
  print_section("1. COVARIANCE FUNCTION EVALUATION");
  auto cov_double = benchmark_covariance_function_double(1000);
  print_result(cov_double);

  auto cov_float = benchmark_covariance_function_float(1000);
  print_result(cov_float);

  print_speedup(cov_double, cov_float);

  // Section 2: Transcendental Functions
  print_section("2. TRANSCENDENTAL FUNCTIONS (exp)");
  auto exp_double = benchmark_exp_double(5000);
  print_result(exp_double);

  auto exp_float = benchmark_exp_float(5000);
  print_result(exp_float);

  print_speedup(exp_double, exp_float);

  // Section 3: Matrix Operations
  print_section("3. MATRIX OPERATIONS");
  auto matmul_double = benchmark_matrix_multiply_double(500);
  print_result(matmul_double);

  auto matmul_float = benchmark_matrix_multiply_float(500);
  print_result(matmul_float);

  print_speedup(matmul_double, matmul_float);

  // Section 4: Precision Conversion Overhead
  print_section("4. PRECISION CONVERSION OVERHEAD");
  auto conv_to_float = benchmark_precision_conversion_to_float(10000);
  print_result(conv_to_float);

  auto conv_to_double = benchmark_precision_conversion_to_double(10000);
  print_result(conv_to_double);

  std::cout << "\n";

  // Summary
  print_section("SUMMARY");
  std::cout << "\n";
  std::cout << "Key Findings:\n";
  std::cout << "-------------\n\n";

  double cov_speedup = cov_double.time_per_iteration_us() / cov_float.time_per_iteration_us();
  std::cout << "- Covariance Function Speedup: " << std::fixed << std::setprecision(2)
            << cov_speedup << "x\n";

  double exp_speedup = exp_double.time_per_iteration_us() / exp_float.time_per_iteration_us();
  std::cout << "- Transcendental (exp) Speedup: " << std::fixed << std::setprecision(2)
            << exp_speedup << "x\n";

  double matmul_speedup = matmul_double.time_per_iteration_us() / matmul_float.time_per_iteration_us();
  std::cout << "- Matrix Multiply Speedup: " << std::fixed << std::setprecision(2)
            << matmul_speedup << "x\n";

  std::cout << "\n";
  std::cout << "Conversion Overhead:\n";
  std::cout << "  - double->float: " << std::fixed << std::setprecision(2)
            << conv_to_float.time_per_iteration_us() << " us per 1000-element vector\n";
  std::cout << "  - float->double: " << std::fixed << std::setprecision(2)
            << conv_to_double.time_per_iteration_us() << " us per 1000-element vector\n";

  std::cout << "\n";
  std::cout << "Expected Overall GP Speedup (with Phase 3):\n";
  std::cout << "  - Training: " << std::fixed << std::setprecision(2)
            << (cov_speedup * 0.6 + matmul_speedup * 0.4) << "x (weighted estimate)\n";
  std::cout << "  - Prediction: " << std::fixed << std::setprecision(2)
            << (cov_speedup * 0.4 + matmul_speedup * 0.6) << "x (weighted estimate)\n";

  std::cout << "\n";
  std::cout << "================================================================\n";
  std::cout << "  + Benchmarks complete!                                        \n";
  std::cout << "================================================================\n\n";
}

} // namespace albatross

int main(int argc, char** argv) {
  std::cout << "\n";
  std::cout << "Albatross Mixed-Precision Performance Benchmark\n";
  std::cout << "===============================================\n";

  albatross::run_benchmarks();

  return 0;
}
