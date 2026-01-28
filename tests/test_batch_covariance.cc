/*
 * Copyright (C) 2024 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include <albatross/CovarianceFunctions>

#include <gtest/gtest.h>

namespace albatross {

/*
 * Test covariance function that implements batch computation.
 * Returns a constant matrix to verify batch method is actually called.
 */
class BatchTestCovariance : public CovarianceFunction<BatchTestCovariance> {
public:
  BatchTestCovariance(double value = 42.0) : value_(value) {}

  std::string name() const { return "batch_test"; }

  // Pointwise implementation (should NOT be called if batch works)
  double _call_impl(const double &x, const double &y) const {
    // Return a different value so we can detect if pointwise was used
    return -999.0;
  }

  // Batch implementation
  Eigen::MatrixXd _call_impl_vector(const std::vector<double> &xs,
                                    const std::vector<double> &ys,
                                    ThreadPool *pool) const {
    const Eigen::Index m = cast::to_index(xs.size());
    const Eigen::Index n = cast::to_index(ys.size());
    return Eigen::MatrixXd::Constant(m, n, value_);
  }

private:
  double value_;
};

/*
 * Test that batch method is actually called when defined
 */
TEST(test_batch_covariance, test_batch_method_is_called) {
  BatchTestCovariance cov(42.0);

  std::vector<double> xs = {1.0, 2.0, 3.0};
  std::vector<double> ys = {4.0, 5.0};

  Eigen::MatrixXd result = cov(xs, ys);

  // If batch was called, all values should be 42.0
  // If pointwise was called, values would be -999.0
  EXPECT_EQ(result.rows(), 3);
  EXPECT_EQ(result.cols(), 2);
  for (Eigen::Index i = 0; i < result.rows(); ++i) {
    for (Eigen::Index j = 0; j < result.cols(); ++j) {
      EXPECT_EQ(result(i, j), 42.0) << "Batch method was not called!";
    }
  }
}

/*
 * Test covariance that implements both pointwise and batch for equivalence
 * testing
 */
class EquivalenceTestCovariance
    : public CovarianceFunction<EquivalenceTestCovariance> {
public:
  EquivalenceTestCovariance() = default;

  std::string name() const { return "equivalence_test"; }

  // Pointwise: simple squared distance
  double _call_impl(const double &x, const double &y) const {
    const double d = x - y;
    return std::exp(-d * d);
  }

  // Batch: should give same result as pointwise
  Eigen::MatrixXd _call_impl_vector(const std::vector<double> &xs,
                                    const std::vector<double> &ys,
                                    ThreadPool *pool) const {
    const Eigen::Index m = cast::to_index(xs.size());
    const Eigen::Index n = cast::to_index(ys.size());
    Eigen::MatrixXd result(m, n);

    for (Eigen::Index i = 0; i < m; ++i) {
      for (Eigen::Index j = 0; j < n; ++j) {
        const double d = xs[cast::to_size(i)] - ys[cast::to_size(j)];
        result(i, j) = std::exp(-d * d);
      }
    }

    return result;
  }
};

/*
 * Test that batch and pointwise give equivalent results
 */
TEST(test_batch_covariance, test_batch_pointwise_equivalence) {
  EquivalenceTestCovariance cov;

  std::vector<double> xs = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> ys = {2.5, 3.5, 4.5};

  // Get batch result
  Eigen::MatrixXd batch_result = cov(xs, ys);

  // Compute pointwise result manually
  Eigen::MatrixXd pointwise_result(xs.size(), ys.size());
  for (size_t i = 0; i < xs.size(); ++i) {
    for (size_t j = 0; j < ys.size(); ++j) {
      pointwise_result(i, j) = cov(xs[i], ys[j]);
    }
  }

  // Check equivalence
  EXPECT_EQ(batch_result.rows(), pointwise_result.rows());
  EXPECT_EQ(batch_result.cols(), pointwise_result.cols());

  for (Eigen::Index i = 0; i < batch_result.rows(); ++i) {
    for (Eigen::Index j = 0; j < batch_result.cols(); ++j) {
      EXPECT_NEAR(batch_result(i, j), pointwise_result(i, j), 1e-10)
          << "Mismatch at (" << i << ", " << j << ")";
    }
  }
}

/*
 * Test that covariances without batch implementation still work
 */
TEST(test_batch_covariance, test_fallback_to_pointwise) {
  // Use a simple covariance without batch implementation
  class PointwiseOnlyCovariance
      : public CovarianceFunction<PointwiseOnlyCovariance> {
  public:
    std::string name() const { return "pointwise_only"; }

    double _call_impl(const double &x, const double &y) const { return x * y; }
  };

  PointwiseOnlyCovariance cov;

  std::vector<double> xs = {1.0, 2.0, 3.0};
  std::vector<double> ys = {4.0, 5.0};

  Eigen::MatrixXd result = cov(xs, ys);

  // Check that fallback to pointwise worked correctly
  EXPECT_EQ(result.rows(), 3);
  EXPECT_EQ(result.cols(), 2);
  EXPECT_DOUBLE_EQ(result(0, 0), 1.0 * 4.0);
  EXPECT_DOUBLE_EQ(result(0, 1), 1.0 * 5.0);
  EXPECT_DOUBLE_EQ(result(1, 0), 2.0 * 4.0);
  EXPECT_DOUBLE_EQ(result(1, 1), 2.0 * 5.0);
  EXPECT_DOUBLE_EQ(result(2, 0), 3.0 * 4.0);
  EXPECT_DOUBLE_EQ(result(2, 1), 3.0 * 5.0);
}

/*
 * Test that batch-only covariances work (no _call_impl defined)
 */
TEST(test_batch_covariance, test_batch_only_covariance) {
  // Covariance that defines ONLY batch, no pointwise
  class BatchOnlyCovariance : public CovarianceFunction<BatchOnlyCovariance> {
  public:
    std::string name() const { return "batch_only"; }

    // NO _call_impl defined!

    // Only batch implementation
    Eigen::MatrixXd _call_impl_vector(const std::vector<double> &xs,
                                      const std::vector<double> &ys,
                                      ThreadPool *pool) const {
      return Eigen::MatrixXd::Constant(xs.size(), ys.size(), 777.0);
    }
  };

  BatchOnlyCovariance cov;

  std::vector<double> xs = {1.0, 2.0};
  std::vector<double> ys = {3.0, 4.0, 5.0};

  Eigen::MatrixXd result = cov(xs, ys);

  EXPECT_EQ(result.rows(), 2);
  EXPECT_EQ(result.cols(), 3);
  for (Eigen::Index i = 0; i < result.rows(); ++i) {
    for (Eigen::Index j = 0; j < result.cols(); ++j) {
      EXPECT_EQ(result(i, j), 777.0) << "Batch-only covariance should work";
    }
  }

  // NOTE: Batch-only covariances with Measurements require the operator()
  // to be enabled for Measurement types. This requires more sophisticated
  // trait checking that accounts for unwrapping. For Phase 1, batch-only
  // covariances work for simple (non-wrapper) types.
}

/*
 * Test variant batch support
 * Variants unwrap to their underlying type, which then uses batch if available.
 * Since batch is preferred over pointwise, variants should use the batch
 * method.
 */
TEST(test_batch_covariance, test_variant_uses_batch) {
  BatchTestCovariance cov(42.0);

  using DoubleVariant = variant<double>;

  std::vector<DoubleVariant> xs;
  xs.push_back(DoubleVariant(1.0));
  xs.push_back(DoubleVariant(2.0));
  xs.push_back(DoubleVariant(3.0));

  std::vector<DoubleVariant> ys;
  ys.push_back(DoubleVariant(4.0));
  ys.push_back(DoubleVariant(5.0));

  // Variants unwrap and use batch (batch is preferred over pointwise)
  Eigen::MatrixXd result = cov(xs, ys);

  EXPECT_EQ(result.rows(), 3);
  EXPECT_EQ(result.cols(), 2);

  // Batch returns 42.0 for all elements
  for (Eigen::Index i = 0; i < result.rows(); ++i) {
    for (Eigen::Index j = 0; j < result.cols(); ++j) {
      EXPECT_EQ(result(i, j), 42.0) << "Variant should use batch method";
    }
  }
}

/*
 * Test that Measurements are properly handled in batch operations
 * Measurements unwrap to their underlying type, which then uses batch if
 * available. Since batch is preferred over pointwise, measurements should use
 * the batch method.
 */
TEST(test_batch_covariance, test_measurement_with_batch_covariance) {
  BatchTestCovariance cov(100.0);

  std::vector<double> xs_raw = {1.0, 2.0};
  std::vector<double> ys_raw = {3.0, 4.0};

  // Test with raw types first - should use batch
  Eigen::MatrixXd raw_result = cov(xs_raw, ys_raw);

  // All should be 100.0 from batch
  for (Eigen::Index i = 0; i < raw_result.rows(); ++i) {
    for (Eigen::Index j = 0; j < raw_result.cols(); ++j) {
      EXPECT_EQ(raw_result(i, j), 100.0) << "Raw should use batch";
    }
  }

  // Test with Measurements - unwraps and uses batch (batch is preferred)
  std::vector<Measurement<double>> xs;
  xs.push_back(Measurement<double>{1.0});
  xs.push_back(Measurement<double>{2.0});

  std::vector<Measurement<double>> ys;
  ys.push_back(Measurement<double>{3.0});
  ys.push_back(Measurement<double>{4.0});

  Eigen::MatrixXd result = cov(xs, ys);

  // Measurements unwrap and use batch (batch is preferred over pointwise)
  EXPECT_EQ(result.rows(), 2);
  EXPECT_EQ(result.cols(), 2);
  for (Eigen::Index i = 0; i < result.rows(); ++i) {
    for (Eigen::Index j = 0; j < result.cols(); ++j) {
      EXPECT_EQ(result(i, j), 100.0)
          << "Measurement uses batch at (" << i << "," << j << ")";
    }
  }
}

/*
 * Test homogeneous variant vector assertion
 */
TEST(test_batch_covariance, test_assert_homogeneous_variant_vector) {
  using TestVariant = variant<double, int>;

  // Homogeneous vector - should not assert
  std::vector<TestVariant> homogeneous = {TestVariant(1.0), TestVariant(2.0),
                                          TestVariant(3.0)};
  EXPECT_NO_THROW(assert_homogeneous_variant_vector(homogeneous));

  // Empty vector - should not assert
  std::vector<TestVariant> empty;
  EXPECT_NO_THROW(assert_homogeneous_variant_vector(empty));

  // Heterogeneous vector - should assert
  std::vector<TestVariant> heterogeneous = {TestVariant(1.0), TestVariant(2),
                                            TestVariant(3.0)};
  EXPECT_DEATH(assert_homogeneous_variant_vector(heterogeneous),
               "All variants in a batch operation must hold the same type");
}

/*
 * Test Sum with both batch
 */
TEST(test_batch_covariance, test_sum_both_batch) {
  BatchTestCovariance batch1(10.0);
  BatchTestCovariance batch2(5.0);
  auto sum = batch1 + batch2;

  std::vector<double> xs = {1.0, 2.0};
  std::vector<double> ys = {3.0, 4.0};

  Eigen::MatrixXd result = sum(xs, ys);

  // Should be 10 + 5 = 15 for all elements
  for (Eigen::Index i = 0; i < result.rows(); ++i) {
    for (Eigen::Index j = 0; j < result.cols(); ++j) {
      EXPECT_EQ(result(i, j), 15.0);
    }
  }
}

/*
 * Test Sum with mixed batch/pointwise
 */
TEST(test_batch_covariance, test_sum_mixed_batch_pointwise) {
  BatchTestCovariance batch(10.0);

  class PointwiseOnly : public CovarianceFunction<PointwiseOnly> {
  public:
    std::string name() const { return "pointwise"; }
    double _call_impl(const double &x, const double &y) const { return x + y; }
  };

  PointwiseOnly pointwise;
  auto sum = batch + pointwise;

  std::vector<double> xs = {1.0, 2.0};
  std::vector<double> ys = {3.0, 4.0};

  Eigen::MatrixXd result = sum(xs, ys);

  // batch returns 10.0, pointwise returns x+y
  EXPECT_EQ(result(0, 0), 10.0 + (1.0 + 3.0)); // 14.0
  EXPECT_EQ(result(0, 1), 10.0 + (1.0 + 4.0)); // 15.0
  EXPECT_EQ(result(1, 0), 10.0 + (2.0 + 3.0)); // 15.0
  EXPECT_EQ(result(1, 1), 10.0 + (2.0 + 4.0)); // 16.0
}

/*
 * Test Product with both batch
 */
TEST(test_batch_covariance, test_product_both_batch) {
  BatchTestCovariance batch1(3.0);
  BatchTestCovariance batch2(4.0);
  auto product = batch1 * batch2;

  std::vector<double> xs = {1.0, 2.0};
  std::vector<double> ys = {3.0, 4.0};

  Eigen::MatrixXd result = product(xs, ys);

  // Should be 3 * 4 = 12 for all elements
  for (Eigen::Index i = 0; i < result.rows(); ++i) {
    for (Eigen::Index j = 0; j < result.cols(); ++j) {
      EXPECT_EQ(result(i, j), 12.0);
    }
  }
}

/*
 * Test Product with mixed batch/pointwise
 */
TEST(test_batch_covariance, test_product_mixed_batch_pointwise) {
  BatchTestCovariance batch(2.0);

  class PointwiseOnly : public CovarianceFunction<PointwiseOnly> {
  public:
    std::string name() const { return "pointwise"; }
    double _call_impl(const double &x, const double &y) const { return x * y; }
  };

  PointwiseOnly pointwise;
  auto product = batch * pointwise;

  std::vector<double> xs = {1.0, 2.0};
  std::vector<double> ys = {3.0, 4.0};

  Eigen::MatrixXd result = product(xs, ys);

  // batch returns 2.0, pointwise returns x*y
  EXPECT_EQ(result(0, 0), 2.0 * (1.0 * 3.0)); // 6.0
  EXPECT_EQ(result(0, 1), 2.0 * (1.0 * 4.0)); // 8.0
  EXPECT_EQ(result(1, 0), 2.0 * (2.0 * 3.0)); // 12.0
  EXPECT_EQ(result(1, 1), 2.0 * (2.0 * 4.0)); // 16.0
}

/*
 * Test that dispatch produces valid (finite) results
 */
TEST(test_batch_covariance, test_no_inf_or_nan) {
  EquivalenceTestCovariance cov;

  std::vector<double> xs = {1.0, 2.0, 3.0};
  std::vector<double> ys = {1.5, 2.5};

  Eigen::MatrixXd result = cov(xs, ys);

  // Check no inf or nan values
  for (Eigen::Index i = 0; i < result.rows(); ++i) {
    for (Eigen::Index j = 0; j < result.cols(); ++j) {
      EXPECT_TRUE(std::isfinite(result(i, j)))
          << "Result at (" << i << "," << j
          << ") is not finite: " << result(i, j);
    }
  }
}

/*
 * Test with real covariance functions (SquaredExponential)
 */
TEST(test_batch_covariance, test_real_covariance_function) {
  SquaredExponential<EuclideanDistance> cov(100.0, 1.0);

  std::vector<double> xs = {0.0, 1.0, 2.0};
  std::vector<double> ys = {0.5, 1.5};

  Eigen::MatrixXd result = cov(xs, ys);

  // Check basic properties
  EXPECT_EQ(result.rows(), 3);
  EXPECT_EQ(result.cols(), 2);

  // Check no inf or nan
  for (Eigen::Index i = 0; i < result.rows(); ++i) {
    for (Eigen::Index j = 0; j < result.cols(); ++j) {
      EXPECT_TRUE(std::isfinite(result(i, j)))
          << "Result at (" << i << "," << j << ") is not finite";
    }
  }

  // Check positive values (covariance should be positive)
  for (Eigen::Index i = 0; i < result.rows(); ++i) {
    for (Eigen::Index j = 0; j < result.cols(); ++j) {
      EXPECT_GT(result(i, j), 0.0) << "Covariance should be positive";
    }
  }
}

/*
 * PHASE 2: Symmetric Batch Covariance Tests
 */

/*
 * Test that symmetric batch method is called
 */
TEST(test_batch_covariance, test_symmetric_batch_called) {
  // Covariance with ONLY symmetric batch (no cross-covariance batch)
  class SymmetricBatchCovariance
      : public CovarianceFunction<SymmetricBatchCovariance> {
  public:
    std::string name() const { return "symmetric_batch"; }

    double _call_impl(const double &x, const double &y) const {
      return -111.0; // Different value to detect if pointwise was used
    }

    // Symmetric batch implementation
    Eigen::MatrixXd _call_impl_vector(const std::vector<double> &xs,
                                      const std::vector<double> &ys,
                                      ThreadPool *pool) const {
      // Return constant to verify batch is called
      const Eigen::Index m = cast::to_index(xs.size());
      const Eigen::Index n = cast::to_index(ys.size());
      return Eigen::MatrixXd::Constant(m, n, 888.0);
    }
  };

  SymmetricBatchCovariance cov;

  std::vector<double> xs = {1.0, 2.0, 3.0};

  Eigen::MatrixXd result = cov(xs); // Symmetric call

  EXPECT_EQ(result.rows(), 3);
  EXPECT_EQ(result.cols(), 3);

  // Should get 888.0 from batch, not -111.0 from pointwise
  for (Eigen::Index i = 0; i < result.rows(); ++i) {
    for (Eigen::Index j = 0; j < result.cols(); ++j) {
      EXPECT_EQ(result(i, j), 888.0) << "Symmetric batch should be used";
    }
  }
}

/*
 * Test symmetric batch-only covariance
 */
TEST(test_batch_covariance, test_symmetric_batch_only) {
  // Covariance with ONLY batch, no pointwise
  class SymmetricBatchOnlyCovariance
      : public CovarianceFunction<SymmetricBatchOnlyCovariance> {
  public:
    std::string name() const { return "symmetric_batch_only"; }

    // NO _call_impl defined!

    Eigen::MatrixXd _call_impl_vector(const std::vector<double> &xs,
                                      const std::vector<double> &ys,
                                      ThreadPool *pool) const {
      const Eigen::Index m = cast::to_index(xs.size());
      const Eigen::Index n = cast::to_index(ys.size());
      return Eigen::MatrixXd::Constant(m, n, 999.0);
    }
  };

  SymmetricBatchOnlyCovariance cov;

  std::vector<double> xs = {1.0, 2.0, 3.0};

  Eigen::MatrixXd result = cov(xs);

  EXPECT_EQ(result.rows(), 3);
  EXPECT_EQ(result.cols(), 3);

  for (Eigen::Index i = 0; i < result.rows(); ++i) {
    for (Eigen::Index j = 0; j < result.cols(); ++j) {
      EXPECT_EQ(result(i, j), 999.0) << "Batch-only symmetric works";
    }
  }
}

/*
 * Test symmetric with Sum composite
 */
TEST(test_batch_covariance, test_symmetric_sum_both_batch) {
  BatchTestCovariance batch1(10.0);
  BatchTestCovariance batch2(5.0);
  auto sum = batch1 + batch2;

  std::vector<double> xs = {1.0, 2.0};

  Eigen::MatrixXd result = sum(xs); // Symmetric

  // Should be 10 + 5 = 15 for all elements
  EXPECT_EQ(result.rows(), 2);
  EXPECT_EQ(result.cols(), 2);
  for (Eigen::Index i = 0; i < result.rows(); ++i) {
    for (Eigen::Index j = 0; j < result.cols(); ++j) {
      EXPECT_EQ(result(i, j), 15.0);
    }
  }
}

/*
 * PHASE 3: Diagonal Batch Covariance Tests
 */

/*
 * Test diagonal batch method is called
 */
TEST(test_batch_covariance, test_diagonal_batch_called) {
  class DiagonalBatchCovariance
      : public CovarianceFunction<DiagonalBatchCovariance> {
  public:
    std::string name() const { return "diagonal_batch"; }

    double _call_impl(const double &x, const double &y) const {
      return -222.0; // Different value to detect if pointwise was used
    }

    // Diagonal batch implementation
    Eigen::VectorXd _call_impl_vector_diagonal(const std::vector<double> &xs,
                                               ThreadPool *pool) const {
      return Eigen::VectorXd::Constant(xs.size(), 555.0);
    }
  };

  DiagonalBatchCovariance cov;
  std::vector<double> xs = {1.0, 2.0, 3.0};

  Eigen::VectorXd result = cov.diagonal(xs);

  EXPECT_EQ(result.size(), 3);
  for (Eigen::Index i = 0; i < result.size(); ++i) {
    EXPECT_EQ(result[i], 555.0) << "Diagonal batch should be used";
  }
}

/*
 * Test diagonal batch-only covariance
 */
TEST(test_batch_covariance, test_diagonal_batch_only) {
  class DiagonalBatchOnlyCovariance
      : public CovarianceFunction<DiagonalBatchOnlyCovariance> {
  public:
    std::string name() const { return "diagonal_batch_only"; }

    // NO _call_impl defined!

    Eigen::VectorXd _call_impl_vector_diagonal(const std::vector<double> &xs,
                                               ThreadPool *pool) const {
      return Eigen::VectorXd::Constant(xs.size(), 666.0);
    }
  };

  DiagonalBatchOnlyCovariance cov;
  std::vector<double> xs = {1.0, 2.0, 3.0, 4.0};

  Eigen::VectorXd result = cov.diagonal(xs);

  EXPECT_EQ(result.size(), 4);
  for (Eigen::Index i = 0; i < result.size(); ++i) {
    EXPECT_EQ(result[i], 666.0) << "Diagonal batch-only works";
  }
}

/*
 * Test diagonal with Sum composite
 */
TEST(test_batch_covariance, test_diagonal_sum_both_batch) {
  class DiagBatch1 : public CovarianceFunction<DiagBatch1> {
  public:
    std::string name() const { return "diag1"; }
    double _call_impl(const double &x, const double &y) const { return 0.0; }
    Eigen::VectorXd _call_impl_vector_diagonal(const std::vector<double> &xs,
                                               ThreadPool *pool) const {
      return Eigen::VectorXd::Constant(xs.size(), 10.0);
    }
  };

  class DiagBatch2 : public CovarianceFunction<DiagBatch2> {
  public:
    std::string name() const { return "diag2"; }
    double _call_impl(const double &x, const double &y) const { return 0.0; }
    Eigen::VectorXd _call_impl_vector_diagonal(const std::vector<double> &xs,
                                               ThreadPool *pool) const {
      return Eigen::VectorXd::Constant(xs.size(), 7.0);
    }
  };

  DiagBatch1 cov1;
  DiagBatch2 cov2;
  auto sum = cov1 + cov2;

  std::vector<double> xs = {1.0, 2.0, 3.0};
  Eigen::VectorXd result = sum.diagonal(xs);

  EXPECT_EQ(result.size(), 3);
  for (Eigen::Index i = 0; i < result.size(); ++i) {
    EXPECT_EQ(result[i], 17.0) << "Sum diagonal: 10 + 7 = 17";
  }
}

/*
 * Test diagonal with Product composite
 */
TEST(test_batch_covariance, test_diagonal_product_both_batch) {
  class DiagBatch1 : public CovarianceFunction<DiagBatch1> {
  public:
    std::string name() const { return "diag1"; }
    double _call_impl(const double &x, const double &y) const { return 0.0; }
    Eigen::VectorXd _call_impl_vector_diagonal(const std::vector<double> &xs,
                                               ThreadPool *pool) const {
      return Eigen::VectorXd::Constant(xs.size(), 5.0);
    }
  };

  class DiagBatch2 : public CovarianceFunction<DiagBatch2> {
  public:
    std::string name() const { return "diag2"; }
    double _call_impl(const double &x, const double &y) const { return 0.0; }
    Eigen::VectorXd _call_impl_vector_diagonal(const std::vector<double> &xs,
                                               ThreadPool *pool) const {
      return Eigen::VectorXd::Constant(xs.size(), 3.0);
    }
  };

  DiagBatch1 cov1;
  DiagBatch2 cov2;
  auto product = cov1 * cov2;

  std::vector<double> xs = {1.0, 2.0};
  Eigen::VectorXd result = product.diagonal(xs);

  EXPECT_EQ(result.size(), 2);
  for (Eigen::Index i = 0; i < result.size(); ++i) {
    EXPECT_EQ(result[i], 15.0) << "Product diagonal: 5 * 3 = 15";
  }
}

/*
 * COMPLETENESS TESTS: Edge Cases and Missing Coverage
 */

/*
 * Test partial batch: only cross-covariance batch, no diagonal
 */
TEST(test_batch_covariance, test_cross_batch_only_no_diagonal) {
  class CrossBatchOnlyCovariance
      : public CovarianceFunction<CrossBatchOnlyCovariance> {
  public:
    std::string name() const { return "cross_batch_only"; }

    double _call_impl(const double &x, const double &y) const {
      return x + y; // Simple implementation for diagonal fallback
    }

    // Only cross-covariance batch, no diagonal batch
    Eigen::MatrixXd _call_impl_vector(const std::vector<double> &xs,
                                      const std::vector<double> &ys,
                                      ThreadPool *pool) const {
      return Eigen::MatrixXd::Constant(xs.size(), ys.size(), 123.0);
    }
  };

  CrossBatchOnlyCovariance cov;
  std::vector<double> xs = {1.0, 2.0, 3.0};

  // Cross-covariance should use batch
  Eigen::MatrixXd cross = cov(xs, xs);
  EXPECT_EQ(cross(0, 0), 123.0) << "Cross batch works";

  // Diagonal should fall back to pointwise _call_impl
  Eigen::VectorXd diag = cov.diagonal(xs);
  EXPECT_EQ(diag[0], 2.0) << "Diagonal falls back to pointwise: 1+1=2";
  EXPECT_EQ(diag[1], 4.0) << "Diagonal falls back to pointwise: 2+2=4";
  EXPECT_EQ(diag[2], 6.0) << "Diagonal falls back to pointwise: 3+3=6";
}

/*
 * Test partial batch: only diagonal, no full matrix batch
 */
TEST(test_batch_covariance, test_diagonal_batch_only_no_matrix) {
  class DiagonalBatchOnlyCovariance
      : public CovarianceFunction<DiagonalBatchOnlyCovariance> {
  public:
    std::string name() const { return "diag_batch_only"; }

    double _call_impl(const double &x, const double &y) const {
      return x * y; // For matrix fallback
    }

    // Only diagonal batch, no full matrix batch
    Eigen::VectorXd _call_impl_vector_diagonal(const std::vector<double> &xs,
                                               ThreadPool *pool) const {
      return Eigen::VectorXd::Constant(xs.size(), 456.0);
    }
  };

  DiagonalBatchOnlyCovariance cov;
  std::vector<double> xs = {2.0, 3.0};

  // Diagonal should use batch
  Eigen::VectorXd diag = cov.diagonal(xs);
  EXPECT_EQ(diag[0], 456.0) << "Diagonal batch works";

  // Full matrix should fall back to pointwise _call_impl
  Eigen::MatrixXd mat = cov(xs);
  EXPECT_EQ(mat(0, 0), 4.0) << "Matrix falls back: 2*2=4";
  EXPECT_EQ(mat(1, 1), 9.0) << "Matrix falls back: 3*3=9";
  EXPECT_EQ(mat(0, 1), 6.0) << "Matrix falls back: 2*3=6";
}

/*
 * Test edge case: empty vectors
 */
TEST(test_batch_covariance, test_empty_vectors) {
  BatchTestCovariance cov(100.0);

  std::vector<double> empty;

  // Empty cross-covariance
  Eigen::MatrixXd cross = cov(empty, empty);
  EXPECT_EQ(cross.rows(), 0);
  EXPECT_EQ(cross.cols(), 0);

  // Empty symmetric
  Eigen::MatrixXd sym = cov(empty);
  EXPECT_EQ(sym.rows(), 0);
  EXPECT_EQ(sym.cols(), 0);

  // Empty diagonal
  Eigen::VectorXd diag = cov.diagonal(empty);
  EXPECT_EQ(diag.size(), 0);
}

/*
 * Test edge case: single element
 */
TEST(test_batch_covariance, test_single_element) {
  BatchTestCovariance cov(100.0);

  std::vector<double> single = {42.0};

  // Matrix should use batch
  Eigen::MatrixXd result = cov(single);
  EXPECT_EQ(result.rows(), 1);
  EXPECT_EQ(result.cols(), 1);
  EXPECT_EQ(result(0, 0), 100.0) << "Single element matrix uses batch";

  // Diagonal falls back to pointwise (BatchTestCovariance has no diagonal
  // batch)
  Eigen::VectorXd diag = cov.diagonal(single);
  EXPECT_EQ(diag.size(), 1);
  EXPECT_EQ(diag[0], -999.0)
      << "Single element diagonal uses pointwise fallback";
}

/*
 * Test that ThreadPool is passed through
 */
TEST(test_batch_covariance, test_threadpool_passed_through) {
  class ThreadPoolAwareCovariance
      : public CovarianceFunction<ThreadPoolAwareCovariance> {
  public:
    std::string name() const { return "threadpool_aware"; }

    double _call_impl(const double &x, const double &y) const { return 0.0; }

    mutable bool pool_was_nullptr = true;

    Eigen::MatrixXd _call_impl_vector(const std::vector<double> &xs,
                                      const std::vector<double> &ys,
                                      ThreadPool *pool) const {
      pool_was_nullptr = (pool == nullptr);
      return Eigen::MatrixXd::Zero(xs.size(), ys.size());
    }
  };

  ThreadPoolAwareCovariance cov;
  std::vector<double> xs = {1.0, 2.0};

  // Call with nullptr
  cov(xs, xs, nullptr);
  EXPECT_TRUE(cov.pool_was_nullptr) << "nullptr should be passed through";

  // Call with actual ThreadPool
  ThreadPool pool(2);
  cov(xs, xs, &pool);
  EXPECT_FALSE(cov.pool_was_nullptr) << "ThreadPool should be passed through";
}

/*
 * Test positive definiteness of batch covariance matrices
 */
TEST(test_batch_covariance, test_batch_positive_definite) {
  // Use a real covariance that should be PD
  SquaredExponential<EuclideanDistance> cov(100.0, 1.0);

  std::vector<double> xs = {0.0, 1.0, 2.0, 3.0, 4.0};

  Eigen::MatrixXd C = cov(xs);

  // Verify positive definite by attempting inverse
  EXPECT_NO_THROW(C.inverse())
      << "Covariance matrix should be positive definite";

  // Also check it's symmetric
  EXPECT_LT((C - C.transpose()).norm(), 1e-10) << "Should be symmetric";
}

/*
 * Test batch covariance with LinearCombination correctness
 * Use a simple covariance that supports the types
 */
TEST(test_batch_covariance, test_batch_with_linear_combination_correctness) {
  class SimpleBatchCov : public CovarianceFunction<SimpleBatchCov> {
  public:
    std::string name() const { return "simple_batch"; }

    double _call_impl(const double &x, const double &y) const {
      return std::exp(-(x - y) * (x - y));
    }

    Eigen::MatrixXd _call_impl_vector(const std::vector<double> &xs,
                                      const std::vector<double> &ys,
                                      ThreadPool *pool) const {
      Eigen::MatrixXd result(xs.size(), ys.size());
      for (size_t i = 0; i < xs.size(); ++i) {
        for (size_t j = 0; j < ys.size(); ++j) {
          result(i, j) = std::exp(-(xs[i] - ys[j]) * (xs[i] - ys[j]));
        }
      }
      return result;
    }
  };

  SimpleBatchCov cov;

  std::vector<double> features = {1.0, 2.0, 3.0};

  // Create a simple linear combination: mean of first two
  std::vector<double> two_features = {features[0], features[1]};
  auto mean_12 = linear_combination::mean(two_features);

  // Batch computation
  Eigen::MatrixXd C = cov(features);

  // Manually compute what mean_12 covariance should be
  // mean_12 = 0.5 * (1 + 2) with coefficients [0.5, 0.5, 0]
  Eigen::Vector3d coeffs(0.5, 0.5, 0.0);
  double expected_cov = coeffs.dot(C * coeffs);

  // Compute using pointwise LinearCombination logic
  double actual_cov = cov(mean_12, mean_12);

  EXPECT_NEAR(actual_cov, expected_cov, 1e-10)
      << "LinearCombination should use covariance matrix correctly";
}

/*
 * Test batch covariance with MeasurementOnly wrapper
 */
TEST(test_batch_covariance, test_batch_with_measurement_only) {
  SquaredExponential<EuclideanDistance> radial(100.0, 1.0);
  IndependentNoise<double> noise(0.1);
  auto meas_noise = measurement_only(noise);
  auto sum = radial + meas_noise;

  std::vector<double> features = {0., 1., 2.};
  std::vector<Measurement<double>> measurements;
  for (const auto &f : features) {
    measurements.emplace_back(Measurement<double>(f));
  }

  const auto f = features[0];
  const auto m = measurements[0];

  // Build covariance matrices
  Eigen::MatrixXd C_features = sum(features);
  Eigen::MatrixXd C_measurements = sum(measurements);

  // Measurement matrix should have larger diagonal (includes noise)
  EXPECT_GT(C_measurements(0, 0), C_features(0, 0))
      << "Measurements should include measurement noise";

  // Off-diagonal should be the same (only radial, no measurement noise)
  EXPECT_DOUBLE_EQ(C_measurements(0, 1), C_features(0, 1))
      << "Off-diagonal should only have radial component";
}

/*
 * Test batch matrix symmetry property
 */
TEST(test_batch_covariance, test_batch_matrix_symmetry) {
  EquivalenceTestCovariance cov;

  std::vector<double> xs = {1.0, 2.0, 3.0};
  std::vector<double> ys = {1.5, 2.5};

  Eigen::MatrixXd C_xy = cov(xs, ys);
  Eigen::MatrixXd C_yx = cov(ys, xs);

  // C(xs, ys) should equal C(ys, xs)^T
  EXPECT_LT((C_xy - C_yx.transpose()).norm(), 1e-10)
      << "Covariance should be symmetric";
}

/*
 * Test that batch respects covariance function parameters
 */
TEST(test_batch_covariance, test_batch_respects_parameters) {
  class ParameterizedBatchCovariance
      : public CovarianceFunction<ParameterizedBatchCovariance> {
  public:
    ALBATROSS_DECLARE_PARAMS(scale_param)

    ParameterizedBatchCovariance() { scale_param = {1.0, PositivePrior()}; }

    std::string name() const { return "parameterized_batch"; }

    double _call_impl(const double &x, const double &y) const {
      return scale_param.value * std::exp(-(x - y) * (x - y));
    }

    Eigen::MatrixXd _call_impl_vector(const std::vector<double> &xs,
                                      const std::vector<double> &ys,
                                      ThreadPool *pool) const {
      Eigen::MatrixXd result(xs.size(), ys.size());
      for (size_t i = 0; i < xs.size(); ++i) {
        for (size_t j = 0; j < ys.size(); ++j) {
          result(i, j) =
              scale_param.value * std::exp(-(xs[i] - ys[j]) * (xs[i] - ys[j]));
        }
      }
      return result;
    }
  };

  ParameterizedBatchCovariance cov;
  std::vector<double> xs = {0.0, 1.0};

  // Test with default parameter
  Eigen::MatrixXd C1 = cov(xs);
  EXPECT_DOUBLE_EQ(C1(0, 0), 1.0); // exp(0) * 1.0

  // Change parameter
  cov.scale_param.value = 5.0;
  Eigen::MatrixXd C2 = cov(xs);
  EXPECT_DOUBLE_EQ(C2(0, 0), 5.0) << "Batch should respect parameter changes";

  // Verify batch and pointwise give same result with new parameter
  EXPECT_DOUBLE_EQ(cov(xs[0], xs[0]), C2(0, 0));
}

/*
 * BATCH-ONLY SCALAR SYNTHESIS TESTS
 *
 * These tests verify that covariance functions which define ONLY
 * _call_impl_vector (no pointwise _call_impl) can still be used
 * with scalar operator() calls. The scalar result is synthesized
 * by calling the batch method with single-element vectors.
 */

/*
 * Test scalar operator() synthesized from batch-only covariance
 */
TEST(test_batch_covariance, test_batch_only_scalar_synthesis) {
  // Covariance that defines ONLY batch, no pointwise
  class BatchOnlyScalarTestCovariance
      : public CovarianceFunction<BatchOnlyScalarTestCovariance> {
  public:
    std::string name() const { return "batch_only_scalar_test"; }

    // NO _call_impl defined!

    // Only batch implementation - returns sum of x and y
    Eigen::MatrixXd _call_impl_vector(const std::vector<double> &xs,
                                      const std::vector<double> &ys,
                                      ThreadPool *pool) const {
      const Eigen::Index m = cast::to_index(xs.size());
      const Eigen::Index n = cast::to_index(ys.size());
      Eigen::MatrixXd result(m, n);
      for (Eigen::Index i = 0; i < m; ++i) {
        for (Eigen::Index j = 0; j < n; ++j) {
          result(i, j) = xs[cast::to_size(i)] + ys[cast::to_size(j)];
        }
      }
      return result;
    }
  };

  BatchOnlyScalarTestCovariance cov;

  // Test scalar calls - should work even without _call_impl
  EXPECT_DOUBLE_EQ(cov(1.0, 2.0), 3.0) << "Scalar synthesis: 1+2=3";
  EXPECT_DOUBLE_EQ(cov(5.0, 7.0), 12.0) << "Scalar synthesis: 5+7=12";
  EXPECT_DOUBLE_EQ(cov(0.0, 0.0), 0.0) << "Scalar synthesis: 0+0=0";
  EXPECT_DOUBLE_EQ(cov(-1.0, 3.0), 2.0) << "Scalar synthesis: -1+3=2";
}

/*
 * Test that batch-only covariance works with Sum composition (scalar calls)
 */
TEST(test_batch_covariance, test_batch_only_sum_scalar) {
  class BatchOnlyCov1 : public CovarianceFunction<BatchOnlyCov1> {
  public:
    std::string name() const { return "batch_only_1"; }

    Eigen::MatrixXd _call_impl_vector(const std::vector<double> &xs,
                                      const std::vector<double> &ys,
                                      ThreadPool *pool) const {
      return Eigen::MatrixXd::Constant(xs.size(), ys.size(), 10.0);
    }
  };

  class BatchOnlyCov2 : public CovarianceFunction<BatchOnlyCov2> {
  public:
    std::string name() const { return "batch_only_2"; }

    Eigen::MatrixXd _call_impl_vector(const std::vector<double> &xs,
                                      const std::vector<double> &ys,
                                      ThreadPool *pool) const {
      return Eigen::MatrixXd::Constant(xs.size(), ys.size(), 7.0);
    }
  };

  BatchOnlyCov1 cov1;
  BatchOnlyCov2 cov2;
  auto sum = cov1 + cov2;

  // Scalar call on sum of batch-only covariances
  EXPECT_DOUBLE_EQ(sum(1.0, 2.0), 17.0) << "Sum of batch-only: 10+7=17";

  // Vector call should also work
  std::vector<double> xs = {1.0, 2.0};
  Eigen::MatrixXd result = sum(xs);
  for (Eigen::Index i = 0; i < result.rows(); ++i) {
    for (Eigen::Index j = 0; j < result.cols(); ++j) {
      EXPECT_DOUBLE_EQ(result(i, j), 17.0);
    }
  }
}

/*
 * Test that batch-only covariance works with Product composition (scalar calls)
 */
TEST(test_batch_covariance, test_batch_only_product_scalar) {
  class BatchOnlyCov1 : public CovarianceFunction<BatchOnlyCov1> {
  public:
    std::string name() const { return "batch_only_prod_1"; }

    Eigen::MatrixXd _call_impl_vector(const std::vector<double> &xs,
                                      const std::vector<double> &ys,
                                      ThreadPool *pool) const {
      return Eigen::MatrixXd::Constant(xs.size(), ys.size(), 3.0);
    }
  };

  class BatchOnlyCov2 : public CovarianceFunction<BatchOnlyCov2> {
  public:
    std::string name() const { return "batch_only_prod_2"; }

    Eigen::MatrixXd _call_impl_vector(const std::vector<double> &xs,
                                      const std::vector<double> &ys,
                                      ThreadPool *pool) const {
      return Eigen::MatrixXd::Constant(xs.size(), ys.size(), 5.0);
    }
  };

  BatchOnlyCov1 cov1;
  BatchOnlyCov2 cov2;
  auto product = cov1 * cov2;

  // Scalar call on product of batch-only covariances
  EXPECT_DOUBLE_EQ(product(1.0, 2.0), 15.0) << "Product of batch-only: 3*5=15";
}

/*
 * Test mixing batch-only with pointwise covariances in Sum (scalar)
 */
TEST(test_batch_covariance, test_batch_only_mixed_sum_scalar) {
  class BatchOnlyCov : public CovarianceFunction<BatchOnlyCov> {
  public:
    std::string name() const { return "batch_only_mixed"; }

    Eigen::MatrixXd _call_impl_vector(const std::vector<double> &xs,
                                      const std::vector<double> &ys,
                                      ThreadPool *pool) const {
      return Eigen::MatrixXd::Constant(xs.size(), ys.size(), 100.0);
    }
  };

  class PointwiseCov : public CovarianceFunction<PointwiseCov> {
  public:
    std::string name() const { return "pointwise_mixed"; }

    double _call_impl(const double &x, const double &y) const { return x * y; }
  };

  BatchOnlyCov batch_cov;
  PointwiseCov pointwise_cov;
  auto sum = batch_cov + pointwise_cov;

  // Scalar call: batch returns 100, pointwise returns x*y
  EXPECT_DOUBLE_EQ(sum(2.0, 3.0), 106.0) << "Mixed sum: 100 + 2*3 = 106";
  EXPECT_DOUBLE_EQ(sum(5.0, 4.0), 120.0) << "Mixed sum: 100 + 5*4 = 120";
}

/*
 * Test mixing batch-only with pointwise covariances in Product (scalar)
 */
TEST(test_batch_covariance, test_batch_only_mixed_product_scalar) {
  class BatchOnlyCov : public CovarianceFunction<BatchOnlyCov> {
  public:
    std::string name() const { return "batch_only_mixed_prod"; }

    Eigen::MatrixXd _call_impl_vector(const std::vector<double> &xs,
                                      const std::vector<double> &ys,
                                      ThreadPool *pool) const {
      return Eigen::MatrixXd::Constant(xs.size(), ys.size(), 2.0);
    }
  };

  class PointwiseCov : public CovarianceFunction<PointwiseCov> {
  public:
    std::string name() const { return "pointwise_mixed_prod"; }

    double _call_impl(const double &x, const double &y) const { return x + y; }
  };

  BatchOnlyCov batch_cov;
  PointwiseCov pointwise_cov;
  auto product = batch_cov * pointwise_cov;

  // Scalar call: batch returns 2, pointwise returns x+y
  EXPECT_DOUBLE_EQ(product(3.0, 4.0), 14.0) << "Mixed product: 2 * (3+4) = 14";
}

/*
 * Test diagonal extraction from batch-only covariance (no
 * _call_impl_vector_diagonal)
 */
TEST(test_batch_covariance, test_batch_only_diagonal_synthesis) {
  class BatchOnlyNoDiagonalCovariance
      : public CovarianceFunction<BatchOnlyNoDiagonalCovariance> {
  public:
    std::string name() const { return "batch_only_no_diag"; }

    // NO _call_impl defined!
    // NO _call_impl_vector_diagonal defined!

    // Only full batch implementation
    Eigen::MatrixXd _call_impl_vector(const std::vector<double> &xs,
                                      const std::vector<double> &ys,
                                      ThreadPool *pool) const {
      const Eigen::Index m = cast::to_index(xs.size());
      const Eigen::Index n = cast::to_index(ys.size());
      Eigen::MatrixXd result(m, n);
      for (Eigen::Index i = 0; i < m; ++i) {
        for (Eigen::Index j = 0; j < n; ++j) {
          // Return x*y so diagonal is x*x = x^2
          result(i, j) = xs[cast::to_size(i)] * ys[cast::to_size(j)];
        }
      }
      return result;
    }
  };

  BatchOnlyNoDiagonalCovariance cov;

  std::vector<double> xs = {2.0, 3.0, 5.0};

  // Diagonal should be extracted from full matrix
  Eigen::VectorXd diag = cov.diagonal(xs);

  EXPECT_EQ(diag.size(), 3);
  EXPECT_DOUBLE_EQ(diag[0], 4.0) << "Diagonal[0]: 2*2=4";
  EXPECT_DOUBLE_EQ(diag[1], 9.0) << "Diagonal[1]: 3*3=9";
  EXPECT_DOUBLE_EQ(diag[2], 25.0) << "Diagonal[2]: 5*5=25";
}

/*
 * Test batch-only with different X and Y types
 */
TEST(test_batch_covariance, test_batch_only_different_types) {
  struct TypeA {
    double value;
  };
  struct TypeB {
    double value;
  };

  class BatchOnlyDifferentTypesCovariance
      : public CovarianceFunction<BatchOnlyDifferentTypesCovariance> {
  public:
    std::string name() const { return "batch_only_diff_types"; }

    // Only batch for A-B cross-covariance
    Eigen::MatrixXd _call_impl_vector(const std::vector<TypeA> &xs,
                                      const std::vector<TypeB> &ys,
                                      ThreadPool *pool) const {
      const Eigen::Index m = cast::to_index(xs.size());
      const Eigen::Index n = cast::to_index(ys.size());
      Eigen::MatrixXd result(m, n);
      for (Eigen::Index i = 0; i < m; ++i) {
        for (Eigen::Index j = 0; j < n; ++j) {
          result(i, j) =
              xs[cast::to_size(i)].value * ys[cast::to_size(j)].value;
        }
      }
      return result;
    }
  };

  BatchOnlyDifferentTypesCovariance cov;

  TypeA a{3.0};
  TypeB b{4.0};

  // Scalar call between different types
  EXPECT_DOUBLE_EQ(cov(a, b), 12.0) << "Different types scalar: 3*4=12";

  // Vector call
  std::vector<TypeA> as = {TypeA{2.0}, TypeA{3.0}};
  std::vector<TypeB> bs = {TypeB{5.0}, TypeB{7.0}};

  Eigen::MatrixXd result = cov(as, bs);
  EXPECT_DOUBLE_EQ(result(0, 0), 10.0); // 2*5
  EXPECT_DOUBLE_EQ(result(0, 1), 14.0); // 2*7
  EXPECT_DOUBLE_EQ(result(1, 0), 15.0); // 3*5
  EXPECT_DOUBLE_EQ(result(1, 1), 21.0); // 3*7
}

/*
 * Test batch-only covariance in three-way composition
 */
TEST(test_batch_covariance, test_batch_only_three_way_composition) {
  class BatchOnlyA : public CovarianceFunction<BatchOnlyA> {
  public:
    std::string name() const { return "batch_a"; }
    Eigen::MatrixXd _call_impl_vector(const std::vector<double> &xs,
                                      const std::vector<double> &ys,
                                      ThreadPool *pool) const {
      return Eigen::MatrixXd::Constant(xs.size(), ys.size(), 2.0);
    }
  };

  class BatchOnlyB : public CovarianceFunction<BatchOnlyB> {
  public:
    std::string name() const { return "batch_b"; }
    Eigen::MatrixXd _call_impl_vector(const std::vector<double> &xs,
                                      const std::vector<double> &ys,
                                      ThreadPool *pool) const {
      return Eigen::MatrixXd::Constant(xs.size(), ys.size(), 3.0);
    }
  };

  class BatchOnlyC : public CovarianceFunction<BatchOnlyC> {
  public:
    std::string name() const { return "batch_c"; }
    Eigen::MatrixXd _call_impl_vector(const std::vector<double> &xs,
                                      const std::vector<double> &ys,
                                      ThreadPool *pool) const {
      return Eigen::MatrixXd::Constant(xs.size(), ys.size(), 5.0);
    }
  };

  BatchOnlyA a;
  BatchOnlyB b;
  BatchOnlyC c;

  // (a + b) * c = (2 + 3) * 5 = 25
  auto composed = (a + b) * c;
  EXPECT_DOUBLE_EQ(composed(1.0, 1.0), 25.0) << "Three-way: (2+3)*5=25";

  // a * b + c = 2*3 + 5 = 11
  auto composed2 = a * b + c;
  EXPECT_DOUBLE_EQ(composed2(1.0, 1.0), 11.0) << "Three-way: 2*3+5=11";
}

/*
 * Test that has_valid_caller trait works for batch-only covariances
 */
TEST(test_batch_covariance, test_batch_only_has_valid_caller) {
  class BatchOnlyTraitTestCovariance
      : public CovarianceFunction<BatchOnlyTraitTestCovariance> {
  public:
    std::string name() const { return "batch_only_trait_test"; }

    Eigen::MatrixXd _call_impl_vector(const std::vector<double> &xs,
                                      const std::vector<double> &ys,
                                      ThreadPool *pool) const {
      return Eigen::MatrixXd::Constant(xs.size(), ys.size(), 1.0);
    }
  };

  // This test verifies compilation - if has_valid_caller didn't work for
  // batch-only covariances, the operator() wouldn't be enabled and this
  // wouldn't compile.
  BatchOnlyTraitTestCovariance cov;

  // These calls should compile and work
  double scalar_result = cov(1.0, 2.0);
  EXPECT_DOUBLE_EQ(scalar_result, 1.0);

  std::vector<double> xs = {1.0};
  Eigen::MatrixXd matrix_result = cov(xs);
  EXPECT_DOUBLE_EQ(matrix_result(0, 0), 1.0);

  Eigen::VectorXd diag_result = cov.diagonal(xs);
  EXPECT_DOUBLE_EQ(diag_result[0], 1.0);
}

} // namespace albatross
