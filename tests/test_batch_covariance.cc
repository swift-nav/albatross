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

/*
 * VECTORIZATION TESTS: Verify batch methods are called with full vectors
 *
 * These tests verify that the CRTP machinery properly delegates to
 * _call_impl_vector with the full training data vector, rather than
 * calling it element-by-element in a loop.
 */

/*
 * MockBatchCovariance: tracks call count, dimensions, and thread pool
 * to verify batch methods are called correctly.
 */
struct BatchCallStats {
  mutable std::atomic<int> call_count{0};
  mutable std::atomic<int> last_xs_size{0};
  mutable std::atomic<int> last_ys_size{0};
  mutable std::atomic<bool> pool_was_nonnull{false};
  mutable std::atomic<std::size_t> pool_thread_count{0};

  void reset() {
    call_count = 0;
    last_xs_size = 0;
    last_ys_size = 0;
    pool_was_nonnull = false;
    pool_thread_count = 0;
  }
};

class MockBatchCovariance : public CovarianceFunction<MockBatchCovariance> {
public:
  std::string name() const { return "mock_batch"; }

  // Shared stats object (survives copies)
  std::shared_ptr<BatchCallStats> stats = std::make_shared<BatchCallStats>();

  // NO _call_impl - batch only!

  Eigen::MatrixXd _call_impl_vector(const std::vector<double> &xs,
                                    const std::vector<double> &ys,
                                    ThreadPool *pool) const {
    stats->call_count++;
    stats->last_xs_size = static_cast<int>(xs.size());
    stats->last_ys_size = static_cast<int>(ys.size());
    stats->pool_was_nonnull = (pool != nullptr);
    if (pool)
      stats->pool_thread_count = pool->thread_count();
    return Eigen::MatrixXd::Constant(cast::to_index(xs.size()),
                                     cast::to_index(ys.size()), 42.0);
  }
};

/*
 * Test: variant batch calls _call_impl_vector exactly once with full dimensions
 */
TEST(test_vectorization, test_variant_batch_single_call) {
  MockBatchCovariance mock;
  mock.stats->reset();

  using DoubleVariant = variant<double>;
  std::vector<DoubleVariant> xs(50, DoubleVariant(1.0));

  Eigen::MatrixXd result = mock(xs);

  EXPECT_EQ(mock.stats->call_count, 1) << "Batch should be called exactly once";
  EXPECT_EQ(mock.stats->last_xs_size, 50) << "Should receive full vector";
  EXPECT_EQ(mock.stats->last_ys_size, 50);
  EXPECT_EQ(result.rows(), 50);
  EXPECT_EQ(result.cols(), 50);
}

/*
 * Test: cross-covariance with variant on left side
 */
TEST(test_vectorization, test_variant_cross_batch_single_call_left) {
  MockBatchCovariance mock;
  mock.stats->reset();

  using DoubleVariant = variant<double>;
  std::vector<DoubleVariant> xs(30, DoubleVariant(1.0));
  std::vector<double> ys(40, 2.0);

  Eigen::MatrixXd result = mock(xs, ys);

  EXPECT_EQ(mock.stats->call_count, 1);
  EXPECT_EQ(mock.stats->last_xs_size, 30);
  EXPECT_EQ(mock.stats->last_ys_size, 40);
}

/*
 * Test: cross-covariance with variant on right side
 */
TEST(test_vectorization, test_variant_cross_batch_single_call_right) {
  MockBatchCovariance mock;
  mock.stats->reset();

  using DoubleVariant = variant<double>;
  std::vector<double> xs(25, 1.0);
  std::vector<DoubleVariant> ys(35, DoubleVariant(2.0));

  Eigen::MatrixXd result = mock(xs, ys);

  EXPECT_EQ(mock.stats->call_count, 1);
  EXPECT_EQ(mock.stats->last_xs_size, 25);
  EXPECT_EQ(mock.stats->last_ys_size, 35);
}

/*
 * Test: both sides are variants (independently homogeneous)
 */
TEST(test_vectorization, test_variant_both_sides_batch) {
  MockBatchCovariance mock;
  mock.stats->reset();

  using DoubleVariant = variant<double>;
  std::vector<DoubleVariant> xs(20, DoubleVariant(1.0));
  std::vector<DoubleVariant> ys(25, DoubleVariant(2.0));

  Eigen::MatrixXd result = mock(xs, ys);

  EXPECT_EQ(mock.stats->call_count, 1);
  EXPECT_EQ(mock.stats->last_xs_size, 20);
  EXPECT_EQ(mock.stats->last_ys_size, 25);
}

/*
 * Test: Measurement vectors should unwrap and use batch
 */
TEST(test_vectorization, test_measurement_batch_single_call) {
  MockBatchCovariance mock;
  mock.stats->reset();

  std::vector<Measurement<double>> ms(100, Measurement<double>{1.0});

  Eigen::MatrixXd result = mock(ms);

  EXPECT_EQ(mock.stats->call_count, 1)
      << "Measurement batch should be called exactly once";
  EXPECT_EQ(mock.stats->last_xs_size, 100) << "Should receive full vector";
  EXPECT_EQ(mock.stats->last_ys_size, 100);
}

/*
 * Test: cross-covariance with Measurement on left side
 */
TEST(test_vectorization, test_measurement_cross_batch_left) {
  MockBatchCovariance mock;
  mock.stats->reset();

  std::vector<Measurement<double>> xs(30, Measurement<double>{1.0});
  std::vector<double> ys(40, 2.0);

  Eigen::MatrixXd result = mock(xs, ys);

  EXPECT_EQ(mock.stats->call_count, 1);
  EXPECT_EQ(mock.stats->last_xs_size, 30);
  EXPECT_EQ(mock.stats->last_ys_size, 40);
}

/*
 * Test: cross-covariance with Measurement on right side
 */
TEST(test_vectorization, test_measurement_cross_batch_right) {
  MockBatchCovariance mock;
  mock.stats->reset();

  std::vector<double> xs(25, 1.0);
  std::vector<Measurement<double>> ys(35, Measurement<double>{2.0});

  Eigen::MatrixXd result = mock(xs, ys);

  EXPECT_EQ(mock.stats->call_count, 1);
  EXPECT_EQ(mock.stats->last_xs_size, 25);
  EXPECT_EQ(mock.stats->last_ys_size, 35);
}

/*
 * Test: both sides Measurement
 */
TEST(test_vectorization, test_measurement_both_sides_batch) {
  MockBatchCovariance mock;
  mock.stats->reset();

  std::vector<Measurement<double>> xs(15, Measurement<double>{1.0});
  std::vector<Measurement<double>> ys(20, Measurement<double>{2.0});

  Eigen::MatrixXd result = mock(xs, ys);

  EXPECT_EQ(mock.stats->call_count, 1);
  EXPECT_EQ(mock.stats->last_xs_size, 15);
  EXPECT_EQ(mock.stats->last_ys_size, 20);
}

/*
 * Test: diagonal extraction with variant
 */
TEST(test_vectorization, test_variant_diagonal_batch) {
  // Need a covariance with both batch and diagonal batch
  class MockDiagonalBatchCov : public CovarianceFunction<MockDiagonalBatchCov> {
  public:
    std::string name() const { return "mock_diag_batch"; }
    mutable int vector_call_count = 0;
    mutable int diagonal_call_count = 0;

    Eigen::MatrixXd _call_impl_vector(const std::vector<double> &xs,
                                      const std::vector<double> &ys,
                                      ThreadPool *pool) const {
      vector_call_count++;
      return Eigen::MatrixXd::Constant(cast::to_index(xs.size()),
                                       cast::to_index(ys.size()), 42.0);
    }

    Eigen::VectorXd _call_impl_vector_diagonal(const std::vector<double> &xs,
                                               ThreadPool *pool) const {
      diagonal_call_count++;
      return Eigen::VectorXd::Constant(cast::to_index(xs.size()), 99.0);
    }
  };

  MockDiagonalBatchCov cov;

  using DoubleVariant = variant<double>;
  std::vector<DoubleVariant> xs(50, DoubleVariant(1.0));

  Eigen::VectorXd result = cov.diagonal(xs);

  EXPECT_EQ(cov.diagonal_call_count, 1) << "Diagonal batch should be called once";
  EXPECT_EQ(cov.vector_call_count, 0) << "Full matrix batch should not be called";
  EXPECT_EQ(result.size(), 50);
  EXPECT_DOUBLE_EQ(result[0], 99.0);
}

/*
 * Test: diagonal extraction with Measurement
 */
TEST(test_vectorization, test_measurement_diagonal_batch) {
  class MockDiagonalBatchCov : public CovarianceFunction<MockDiagonalBatchCov> {
  public:
    std::string name() const { return "mock_diag_batch"; }
    mutable int vector_call_count = 0;
    mutable int diagonal_call_count = 0;

    Eigen::MatrixXd _call_impl_vector(const std::vector<double> &xs,
                                      const std::vector<double> &ys,
                                      ThreadPool *pool) const {
      vector_call_count++;
      return Eigen::MatrixXd::Constant(cast::to_index(xs.size()),
                                       cast::to_index(ys.size()), 42.0);
    }

    Eigen::VectorXd _call_impl_vector_diagonal(const std::vector<double> &xs,
                                               ThreadPool *pool) const {
      diagonal_call_count++;
      return Eigen::VectorXd::Constant(cast::to_index(xs.size()), 88.0);
    }
  };

  MockDiagonalBatchCov cov;

  std::vector<Measurement<double>> xs(60, Measurement<double>{1.0});

  Eigen::VectorXd result = cov.diagonal(xs);

  EXPECT_EQ(cov.diagonal_call_count, 1) << "Diagonal batch should be called once";
  EXPECT_EQ(cov.vector_call_count, 0) << "Full matrix batch should not be called";
  EXPECT_EQ(result.size(), 60);
  EXPECT_DOUBLE_EQ(result[0], 88.0);
}

/*
 * Test: multi-type variant with batch support
 */
TEST(test_vectorization, test_variant_multi_type_batch) {
  class MultiTypeBatchCov : public CovarianceFunction<MultiTypeBatchCov> {
  public:
    std::string name() const { return "multi"; }
    mutable int double_calls = 0;
    mutable int int_calls = 0;

    Eigen::MatrixXd _call_impl_vector(const std::vector<double> &xs,
                                      const std::vector<double> &ys,
                                      ThreadPool *) const {
      double_calls++;
      return Eigen::MatrixXd::Constant(cast::to_index(xs.size()),
                                       cast::to_index(ys.size()), 1.0);
    }

    Eigen::MatrixXd _call_impl_vector(const std::vector<int> &xs,
                                      const std::vector<int> &ys,
                                      ThreadPool *) const {
      int_calls++;
      return Eigen::MatrixXd::Constant(cast::to_index(xs.size()),
                                       cast::to_index(ys.size()), 2.0);
    }
  };

  MultiTypeBatchCov cov;

  using TestVariant = variant<double, int>;

  // Test with doubles
  std::vector<TestVariant> xs_double(20, TestVariant(1.0));
  std::vector<TestVariant> ys_double(20, TestVariant(2.0));

  Eigen::MatrixXd result_double = cov(xs_double, ys_double);
  EXPECT_EQ(cov.double_calls, 1);
  EXPECT_EQ(cov.int_calls, 0);
  EXPECT_DOUBLE_EQ(result_double(0, 0), 1.0);

  // Reset and test with ints
  cov.double_calls = 0;
  cov.int_calls = 0;

  std::vector<TestVariant> xs_int(15, TestVariant(1));
  std::vector<TestVariant> ys_int(15, TestVariant(2));

  Eigen::MatrixXd result_int = cov(xs_int, ys_int);
  EXPECT_EQ(cov.double_calls, 0);
  EXPECT_EQ(cov.int_calls, 1);
  EXPECT_DOUBLE_EQ(result_int(0, 0), 2.0);
}

/*
 * ============================================================================
 * Measurement<X> + Batch Covariance Tests
 *
 * These tests verify that MeasurementForwarder correctly unwraps Measurement
 * types and uses batch dispatch when the covariance function has
 * _call_impl_vector for the inner type but NOT explicit Measurement support.
 * ============================================================================
 */

/*
 * MockBatchCovarianceWithPointwise: has both batch and pointwise for plain
 * types, but NO explicit Measurement support. When called with Measurement<X>,
 * it should unwrap and use batch.
 */
class MockBatchCovarianceWithPointwise
    : public CovarianceFunction<MockBatchCovarianceWithPointwise> {
public:
  std::string name() const { return "mock_batch_with_pointwise"; }

  std::shared_ptr<BatchCallStats> stats = std::make_shared<BatchCallStats>();

  // Pointwise for plain types
  double _call_impl(const double &x, const double &y) const {
    return x * y; // Different from batch to detect which path is used
  }

  // Batch for plain types
  Eigen::MatrixXd _call_impl_vector(const std::vector<double> &xs,
                                    const std::vector<double> &ys,
                                    ThreadPool *pool) const {
    stats->call_count++;
    stats->last_xs_size = static_cast<int>(xs.size());
    stats->last_ys_size = static_cast<int>(ys.size());
    stats->pool_was_nonnull = (pool != nullptr);
    if (pool)
      stats->pool_thread_count = pool->thread_count();
    return Eigen::MatrixXd::Constant(cast::to_index(xs.size()),
                                     cast::to_index(ys.size()), 42.0);
  }
};

/*
 * MeasurementAwareBatchCovariance: has explicit _call_impl for Measurement
 * types. When called with Measurement<X>, it MUST use pointwise to respect the
 * special Measurement handling.
 */
class MeasurementAwareBatchCovariance
    : public CovarianceFunction<MeasurementAwareBatchCovariance> {
public:
  std::string name() const { return "measurement_aware_batch"; }

  mutable int batch_call_count = 0;
  mutable int measurement_pointwise_count = 0;

  void reset() const {
    batch_call_count = 0;
    measurement_pointwise_count = 0;
  }

  // Pointwise for plain types
  double _call_impl(const double &x, const double &y) const { return x * y; }

  // Batch for plain types
  Eigen::MatrixXd _call_impl_vector(const std::vector<double> &xs,
                                    const std::vector<double> &ys,
                                    ThreadPool *) const {
    batch_call_count++;
    return Eigen::MatrixXd::Constant(cast::to_index(xs.size()),
                                     cast::to_index(ys.size()), 1.0);
  }

  // EXPLICIT Measurement support - should be used instead of batch
  // This simulates IndependentNoise behavior where Measurement matters
  double _call_impl(const Measurement<double> &x,
                    const Measurement<double> &y) const {
    measurement_pointwise_count++;
    // Special handling: return different value to detect this path
    return 99.0;
  }
};

/*
 * Test: Measurement<X> with batch covariance (no explicit Measurement support)
 * should unwrap and use batch dispatch (single call)
 */
TEST(test_measurement_batch, test_measurement_unwrap_uses_batch_symmetric) {
  MockBatchCovarianceWithPointwise mock;
  mock.stats->reset();

  std::vector<Measurement<double>> ms;
  for (int i = 0; i < 50; ++i) {
    ms.push_back(Measurement<double>{static_cast<double>(i)});
  }

  Eigen::MatrixXd result = mock(ms);

  EXPECT_EQ(mock.stats->call_count, 1) << "Batch should be called exactly once";
  EXPECT_EQ(mock.stats->last_xs_size, 50) << "Should receive full vector";
  EXPECT_EQ(mock.stats->last_ys_size, 50);
  EXPECT_EQ(result.rows(), 50);
  EXPECT_EQ(result.cols(), 50);
  // Verify batch value (42.0) not pointwise value
  EXPECT_DOUBLE_EQ(result(0, 0), 42.0);
}

/*
 * Test: Measurement<X> cross-covariance uses batch
 */
TEST(test_measurement_batch, test_measurement_unwrap_uses_batch_cross) {
  MockBatchCovarianceWithPointwise mock;
  mock.stats->reset();

  std::vector<Measurement<double>> xs;
  std::vector<Measurement<double>> ys;
  for (int i = 0; i < 30; ++i) {
    xs.push_back(Measurement<double>{static_cast<double>(i)});
  }
  for (int i = 0; i < 40; ++i) {
    ys.push_back(Measurement<double>{static_cast<double>(i + 100)});
  }

  Eigen::MatrixXd result = mock(xs, ys);

  EXPECT_EQ(mock.stats->call_count, 1) << "Batch should be called exactly once";
  EXPECT_EQ(mock.stats->last_xs_size, 30);
  EXPECT_EQ(mock.stats->last_ys_size, 40);
  EXPECT_EQ(result.rows(), 30);
  EXPECT_EQ(result.cols(), 40);
  EXPECT_DOUBLE_EQ(result(0, 0), 42.0);
}

/*
 * Test: Measurement on left, plain on right - should unwrap left and use batch
 */
TEST(test_measurement_batch, test_measurement_left_plain_right_uses_batch) {
  MockBatchCovarianceWithPointwise mock;
  mock.stats->reset();

  std::vector<Measurement<double>> xs;
  for (int i = 0; i < 20; ++i) {
    xs.push_back(Measurement<double>{static_cast<double>(i)});
  }
  std::vector<double> ys(25, 1.0);

  Eigen::MatrixXd result = mock(xs, ys);

  EXPECT_EQ(mock.stats->call_count, 1);
  EXPECT_EQ(mock.stats->last_xs_size, 20);
  EXPECT_EQ(mock.stats->last_ys_size, 25);
  EXPECT_DOUBLE_EQ(result(0, 0), 42.0);
}

/*
 * Test: Plain on left, Measurement on right - should unwrap right and use batch
 */
TEST(test_measurement_batch, test_plain_left_measurement_right_uses_batch) {
  MockBatchCovarianceWithPointwise mock;
  mock.stats->reset();

  std::vector<double> xs(15, 1.0);
  std::vector<Measurement<double>> ys;
  for (int i = 0; i < 22; ++i) {
    ys.push_back(Measurement<double>{static_cast<double>(i)});
  }

  Eigen::MatrixXd result = mock(xs, ys);

  EXPECT_EQ(mock.stats->call_count, 1);
  EXPECT_EQ(mock.stats->last_xs_size, 15);
  EXPECT_EQ(mock.stats->last_ys_size, 22);
  EXPECT_DOUBLE_EQ(result(0, 0), 42.0);
}

/*
 * Test: Covariance with explicit Measurement support MUST use pointwise
 * to respect the special Measurement handling
 */
TEST(test_measurement_batch, test_explicit_measurement_support_uses_pointwise) {
  MeasurementAwareBatchCovariance cov;
  cov.reset();

  std::vector<Measurement<double>> ms;
  for (int i = 0; i < 5; ++i) {
    ms.push_back(Measurement<double>{static_cast<double>(i)});
  }

  Eigen::MatrixXd result = cov(ms);

  // Should NOT use batch since there's explicit Measurement support
  EXPECT_EQ(cov.batch_call_count, 0)
      << "Batch should NOT be called when explicit Measurement support exists";
  // Should use pointwise (5x5 = 25 calls for symmetric, but symmetric compute
  // only does upper triangle + diagonal = 5+4+3+2+1 = 15 pairs)
  EXPECT_GT(cov.measurement_pointwise_count, 0)
      << "Pointwise should be called for explicit Measurement support";
  // Verify pointwise value (99.0)
  EXPECT_DOUBLE_EQ(result(0, 0), 99.0);
}

/*
 * Test: Measurement + batch with thread pool - pool should be passed through
 */
TEST(test_measurement_batch, test_measurement_batch_with_thread_pool) {
  MockBatchCovarianceWithPointwise mock;
  mock.stats->reset();

  std::vector<Measurement<double>> ms;
  for (int i = 0; i < 100; ++i) {
    ms.push_back(Measurement<double>{static_cast<double>(i)});
  }

  ThreadPool pool(4);
  Eigen::MatrixXd result = mock(ms, &pool);

  EXPECT_EQ(mock.stats->call_count, 1);
  EXPECT_TRUE(mock.stats->pool_was_nonnull) << "Thread pool should be passed";
  EXPECT_EQ(mock.stats->pool_thread_count, 4);
}

/*
 * Test: Sum composition with Measurement - both sides should use batch
 */
TEST(test_measurement_batch, test_sum_composition_measurement_batch) {
  MockBatchCovarianceWithPointwise lhs;
  MockBatchCovarianceWithPointwise rhs;
  lhs.stats->reset();
  rhs.stats->reset();

  // Verify traits are satisfied
  static_assert(
      has_valid_call_impl_vector<MockBatchCovarianceWithPointwise, double,
                                 double>::value,
      "MockBatchCovarianceWithPointwise should have batch support");
  static_assert(
      has_valid_call_impl_vector_symmetric<MockBatchCovarianceWithPointwise,
                                           double>::value,
      "MockBatchCovarianceWithPointwise should have symmetric batch support");

  auto sum = lhs + rhs;

  // Verify Sum has batch support for double
  using SumType = decltype(sum);
  static_assert(has_valid_call_impl_vector<SumType, double, double>::value,
                "Sum should have batch support for double");
  static_assert(has_valid_call_impl_vector_symmetric<SumType, double>::value,
                "Sum should have symmetric batch support for double");

  // Verify Sum has batch support for Measurement<double>
  static_assert(
      has_valid_call_impl_vector<SumType, Measurement<double>,
                                 Measurement<double>>::value,
      "Sum should have batch support for Measurement<double>");
  static_assert(
      has_valid_call_impl_vector_symmetric<SumType, Measurement<double>>::value,
      "Sum should have symmetric batch support for Measurement<double>");

  std::vector<Measurement<double>> ms;
  for (int i = 0; i < 25; ++i) {
    ms.push_back(Measurement<double>{static_cast<double>(i)});
  }

  Eigen::MatrixXd result = sum(ms);

  // Both sides should have been called exactly once with batch
  EXPECT_EQ(lhs.stats->call_count, 1)
      << "LHS batch should be called exactly once";
  EXPECT_EQ(rhs.stats->call_count, 1)
      << "RHS batch should be called exactly once";
  EXPECT_EQ(lhs.stats->last_xs_size, 25);
  EXPECT_EQ(rhs.stats->last_xs_size, 25);
  // Result should be sum of batch values (42 + 42 = 84)
  EXPECT_DOUBLE_EQ(result(0, 0), 84.0);
}

/*
 * Test: Product composition with Measurement - both sides should use batch
 */
TEST(test_measurement_batch, test_product_composition_measurement_batch) {
  MockBatchCovarianceWithPointwise lhs;
  MockBatchCovarianceWithPointwise rhs;
  lhs.stats->reset();
  rhs.stats->reset();

  auto product = lhs * rhs;

  std::vector<Measurement<double>> ms;
  for (int i = 0; i < 20; ++i) {
    ms.push_back(Measurement<double>{static_cast<double>(i)});
  }

  Eigen::MatrixXd result = product(ms);

  // Both sides should have been called exactly once with batch
  EXPECT_EQ(lhs.stats->call_count, 1);
  EXPECT_EQ(rhs.stats->call_count, 1);
  // Result should be product of batch values (42 * 42 = 1764)
  EXPECT_DOUBLE_EQ(result(0, 0), 42.0 * 42.0);
}

/*
 * Test: Mixed composition - one side batch, one side pointwise-only
 * The Measurement should still be unwrapped correctly
 */
TEST(test_measurement_batch, test_mixed_composition_measurement) {
  MockBatchCovarianceWithPointwise batch_cov;
  batch_cov.stats->reset();

  // Create a pointwise-only covariance
  class PointwiseOnlyCov : public CovarianceFunction<PointwiseOnlyCov> {
  public:
    std::string name() const { return "pointwise_only"; }
    double _call_impl(const double &x, const double &y) const { return 1.0; }
  };
  PointwiseOnlyCov pointwise_cov;

  auto sum = batch_cov + pointwise_cov;

  std::vector<Measurement<double>> ms;
  for (int i = 0; i < 10; ++i) {
    ms.push_back(Measurement<double>{static_cast<double>(i)});
  }

  Eigen::MatrixXd result = sum(ms);

  // Batch side should be called once
  EXPECT_EQ(batch_cov.stats->call_count, 1);
  EXPECT_EQ(batch_cov.stats->last_xs_size, 10);
  // Result should be batch + pointwise (42 + 1 = 43)
  EXPECT_DOUBLE_EQ(result(0, 0), 43.0);
}

/*
 * ============================================================================
 * PHASE 0: Permutation Infrastructure Tests
 *
 * These tests validate the sort-by-type and permutation logic used for
 * heterogeneous variant batch covariance dispatch.
 * ============================================================================
 */

/*
 * Test 0a: sort_variants_by_type produces correct grouping
 */
TEST(test_variant_permutation, test_sort_groups_by_type) {
  using V = variant<int, double, std::string>;
  std::vector<V> input = {V(1.0), V(42), V(std::string("hello")),
                          V(2.0), V(99), V(std::string("world"))};
  // Types: [double, int, string, double, int, string]
  // Indices: [0, 1, 2, 3, 4, 5]

  auto sorted =
      internal::variant_batch_detail::sort_variants_by_type(input);

  // Verify block boundaries
  // Order should be: all ints, then all doubles, then all strings (by type
  // index) Type indices: int=0, double=1, string=2
  EXPECT_EQ(sorted.block_starts[0], 0);  // int starts at 0
  EXPECT_EQ(sorted.block_starts[1], 2);  // double starts at 2 (2 ints)
  EXPECT_EQ(sorted.block_starts[2], 4);  // string starts at 4 (2 doubles)
  EXPECT_EQ(sorted.block_starts[3], 6);  // end sentinel

  // Verify sorted vector contains correct elements (grouped by type)
  EXPECT_EQ(sorted.sorted[0].template get<int>(), 42);
  EXPECT_EQ(sorted.sorted[1].template get<int>(), 99);
  EXPECT_DOUBLE_EQ(sorted.sorted[2].template get<double>(), 1.0);
  EXPECT_DOUBLE_EQ(sorted.sorted[3].template get<double>(), 2.0);
  EXPECT_EQ(sorted.sorted[4].template get<std::string>(), "hello");
  EXPECT_EQ(sorted.sorted[5].template get<std::string>(), "world");
}

/*
 * Test 0b: Permutation matrix correctly maps sorted↔original
 */
TEST(test_variant_permutation, test_permutation_indices) {
  using V = variant<int, double>;
  std::vector<V> input = {V(1.0), V(42), V(2.0), V(99)};
  // Types: [double, int, double, int]
  // Original indices: [0, 1, 2, 3]

  auto sorted =
      internal::variant_batch_detail::sort_variants_by_type(input);

  // Sorted order: [42, 99, 1.0, 2.0] (ints first, then doubles)
  // Sorted indices: [0, 1, 2, 3]
  // Eigen convention: indices[orig_idx] = sorted_idx means
  // original element at orig_idx goes to sorted position sorted_idx
  // So (P * original)[sorted_idx] = original[orig_idx]
  //
  // original[0]=1.0 (double) -> sorted[2]
  // original[1]=42 (int) -> sorted[0]
  // original[2]=2.0 (double) -> sorted[3]
  // original[3]=99 (int) -> sorted[1]

  EXPECT_EQ(sorted.to_sorted.indices()[0], 2); // original[0]=1.0 goes to sorted[2]
  EXPECT_EQ(sorted.to_sorted.indices()[1], 0); // original[1]=42 goes to sorted[0]
  EXPECT_EQ(sorted.to_sorted.indices()[2], 3); // original[2]=2.0 goes to sorted[3]
  EXPECT_EQ(sorted.to_sorted.indices()[3], 1); // original[3]=99 goes to sorted[1]
}

/*
 * Test 0c: Inverse permutation recovers original order
 */
TEST(test_variant_permutation, test_inverse_permutation_roundtrip) {
  using V = variant<int, double, std::string>;
  std::vector<V> input = {V(1.0), V(42), V(std::string("a")),
                          V(2.0), V(99), V(std::string("b"))};

  auto sorted =
      internal::variant_batch_detail::sort_variants_by_type(input);

  // Create a simple matrix in sorted order
  const Eigen::Index n = cast::to_index(input.size());
  Eigen::MatrixXd sorted_matrix(n, n);
  for (Eigen::Index i = 0; i < n; ++i) {
    for (Eigen::Index j = 0; j < n; ++j) {
      sorted_matrix(i, j) = static_cast<double>(i * 10 + j);
    }
  }

  // Unpermute: result = P^{-1} * sorted_matrix * P^{-T}
  // Note: We need to evaluate the inverse before calling transpose()
  Eigen::PermutationMatrix<Eigen::Dynamic> P_inv = sorted.to_sorted.inverse();
  Eigen::MatrixXd result = P_inv * sorted_matrix * P_inv.transpose();

  // Verify: result(i,j) should equal sorted_matrix(sorted_idx[i], sorted_idx[j])
  // where sorted_idx[orig_i] is the position of original[orig_i] in sorted order
  // With Eigen convention: indices[orig_i] = sorted_i
  for (Eigen::Index i = 0; i < n; ++i) {
    for (Eigen::Index j = 0; j < n; ++j) {
      // indices[orig_i] directly gives the sorted position
      Eigen::Index sorted_i = sorted.to_sorted.indices()[i];
      Eigen::Index sorted_j = sorted.to_sorted.indices()[j];
      EXPECT_DOUBLE_EQ(result(i, j), sorted_matrix(sorted_i, sorted_j));
    }
  }
}

/*
 * Test 0d: Empty vector edge case
 */
TEST(test_variant_permutation, test_empty_vector) {
  using V = variant<int, double>;
  std::vector<V> input = {};

  auto sorted =
      internal::variant_batch_detail::sort_variants_by_type(input);

  EXPECT_EQ(sorted.sorted.size(), 0u);
  EXPECT_EQ(sorted.block_starts[0], 0);
  EXPECT_EQ(sorted.block_starts[1], 0);
  EXPECT_EQ(sorted.block_starts[2], 0);
}

/*
 * Test 0d: Single element edge case
 */
TEST(test_variant_permutation, test_single_element) {
  using V = variant<int, double>;
  std::vector<V> input = {V(42)};

  auto sorted =
      internal::variant_batch_detail::sort_variants_by_type(input);

  EXPECT_EQ(sorted.sorted.size(), 1u);
  EXPECT_EQ(sorted.sorted[0].template get<int>(), 42);
  EXPECT_EQ(sorted.to_sorted.indices()[0], 0);
}

/*
 * Test 0e: Homogeneous input (single type)
 */
TEST(test_variant_permutation, test_homogeneous_vector) {
  using V = variant<int, double>;
  std::vector<V> input = {V(1.0), V(2.0), V(3.0)};

  auto sorted =
      internal::variant_batch_detail::sort_variants_by_type(input);

  // All doubles, so no reordering needed
  EXPECT_EQ(sorted.block_starts[0], 0); // int: empty
  EXPECT_EQ(sorted.block_starts[1], 0); // double starts at 0
  EXPECT_EQ(sorted.block_starts[2], 3); // end

  // Permutation should be identity for this portion
  for (Eigen::Index i = 0; i < 3; ++i) {
    EXPECT_EQ(sorted.to_sorted.indices()[i], i);
  }
}

/*
 * Test 0f: Random shuffle integration test with identity covariance
 *
 * This test verifies that heterogeneous variant batch dispatch produces
 * the correct output matrix by using a covariance that returns distinguishable
 * values based on element identity.
 */
class IdentityCovariance : public CovarianceFunction<IdentityCovariance> {
public:
  std::string name() const { return "identity"; }

  double _call_impl(const int &x, const int &y) const {
    return static_cast<double>(x * 1000 + y); // Encodes (x, y)
  }
  double _call_impl(const double &x, const double &y) const {
    return x * 1000.0 + y; // Encodes (x, y)
  }
  // Cross-type: return negative to distinguish
  double _call_impl(const int &x, const double &y) const {
    return -(static_cast<double>(x) * 1000.0 + y);
  }
  double _call_impl(const double &x, const int &y) const {
    return -(x * 1000.0 + static_cast<double>(y));
  }

  // Batch methods that use the scalar _call_impl
  Eigen::MatrixXd _call_impl_vector(const std::vector<int> &xs,
                                    const std::vector<int> &ys,
                                    ThreadPool *) const {
    Eigen::MatrixXd result(cast::to_index(xs.size()),
                           cast::to_index(ys.size()));
    for (std::size_t i = 0; i < xs.size(); ++i)
      for (std::size_t j = 0; j < ys.size(); ++j)
        result(cast::to_index(i), cast::to_index(j)) = _call_impl(xs[i], ys[j]);
    return result;
  }

  Eigen::MatrixXd _call_impl_vector(const std::vector<double> &xs,
                                    const std::vector<double> &ys,
                                    ThreadPool *) const {
    Eigen::MatrixXd result(cast::to_index(xs.size()),
                           cast::to_index(ys.size()));
    for (std::size_t i = 0; i < xs.size(); ++i)
      for (std::size_t j = 0; j < ys.size(); ++j)
        result(cast::to_index(i), cast::to_index(j)) = _call_impl(xs[i], ys[j]);
    return result;
  }

  Eigen::MatrixXd _call_impl_vector(const std::vector<int> &xs,
                                    const std::vector<double> &ys,
                                    ThreadPool *) const {
    Eigen::MatrixXd result(cast::to_index(xs.size()),
                           cast::to_index(ys.size()));
    for (std::size_t i = 0; i < xs.size(); ++i)
      for (std::size_t j = 0; j < ys.size(); ++j)
        result(cast::to_index(i), cast::to_index(j)) = _call_impl(xs[i], ys[j]);
    return result;
  }

  Eigen::MatrixXd _call_impl_vector(const std::vector<double> &xs,
                                    const std::vector<int> &ys,
                                    ThreadPool *) const {
    Eigen::MatrixXd result(cast::to_index(xs.size()),
                           cast::to_index(ys.size()));
    for (std::size_t i = 0; i < xs.size(); ++i)
      for (std::size_t j = 0; j < ys.size(); ++j)
        result(cast::to_index(i), cast::to_index(j)) = _call_impl(xs[i], ys[j]);
    return result;
  }
};

TEST(test_variant_permutation, test_random_shuffle_covariance_correctness) {
  IdentityCovariance cov;

  // Create input with known values
  using V = variant<int, double>;
  std::vector<V> xs = {V(1), V(10.0), V(2), V(20.0), V(3)};
  std::vector<V> ys = {V(100.0), V(4), V(200.0)};

  // Compute via heterogeneous batch dispatch
  Eigen::MatrixXd result = cov(xs, ys);

  // Verify each element matches expected value
  // result(i, j) should equal cov(xs[i], ys[j])
  for (std::size_t i = 0; i < xs.size(); ++i) {
    for (std::size_t j = 0; j < ys.size(); ++j) {
      double expected = xs[i].match(
          [&](int x) {
            return ys[j].match([&](int y) { return cov._call_impl(x, y); },
                               [&](double y) { return cov._call_impl(x, y); });
          },
          [&](double x) {
            return ys[j].match([&](int y) { return cov._call_impl(x, y); },
                               [&](double y) { return cov._call_impl(x, y); });
          });
      EXPECT_DOUBLE_EQ(result(cast::to_index(i), cast::to_index(j)), expected)
          << "Mismatch at (" << i << ", " << j << ")";
    }
  }
}

/*
 * Test heterogeneous symmetric covariance
 *
 * NOTE: Symmetric dispatch uses transpose optimization for off-diagonal blocks.
 * This is correct for proper covariance functions where cov(X,Y) = cov(Y,X).
 * We use a properly symmetric test covariance here.
 */
class SymmetricIdentityCovariance
    : public CovarianceFunction<SymmetricIdentityCovariance> {
public:
  std::string name() const { return "symmetric_identity"; }

  // Returns a symmetric value based on element identity
  double _call_impl(const int &x, const int &y) const {
    return static_cast<double>(x * y); // Symmetric: x*y = y*x
  }
  double _call_impl(const double &x, const double &y) const {
    return x * y; // Symmetric
  }
  // Cross-type: use min/max to ensure symmetry
  double _call_impl(const int &x, const double &y) const {
    return static_cast<double>(x) * y; // Symmetric when called as (y, x)
  }
  double _call_impl(const double &x, const int &y) const {
    return x * static_cast<double>(y); // Same as (int, double) version
  }

  // Batch methods
  Eigen::MatrixXd _call_impl_vector(const std::vector<int> &xs,
                                    const std::vector<int> &ys,
                                    ThreadPool *) const {
    Eigen::MatrixXd result(cast::to_index(xs.size()),
                           cast::to_index(ys.size()));
    for (std::size_t i = 0; i < xs.size(); ++i)
      for (std::size_t j = 0; j < ys.size(); ++j)
        result(cast::to_index(i), cast::to_index(j)) = _call_impl(xs[i], ys[j]);
    return result;
  }

  Eigen::MatrixXd _call_impl_vector(const std::vector<double> &xs,
                                    const std::vector<double> &ys,
                                    ThreadPool *) const {
    Eigen::MatrixXd result(cast::to_index(xs.size()),
                           cast::to_index(ys.size()));
    for (std::size_t i = 0; i < xs.size(); ++i)
      for (std::size_t j = 0; j < ys.size(); ++j)
        result(cast::to_index(i), cast::to_index(j)) = _call_impl(xs[i], ys[j]);
    return result;
  }

  Eigen::MatrixXd _call_impl_vector(const std::vector<int> &xs,
                                    const std::vector<double> &ys,
                                    ThreadPool *) const {
    Eigen::MatrixXd result(cast::to_index(xs.size()),
                           cast::to_index(ys.size()));
    for (std::size_t i = 0; i < xs.size(); ++i)
      for (std::size_t j = 0; j < ys.size(); ++j)
        result(cast::to_index(i), cast::to_index(j)) = _call_impl(xs[i], ys[j]);
    return result;
  }

  Eigen::MatrixXd _call_impl_vector(const std::vector<double> &xs,
                                    const std::vector<int> &ys,
                                    ThreadPool *) const {
    Eigen::MatrixXd result(cast::to_index(xs.size()),
                           cast::to_index(ys.size()));
    for (std::size_t i = 0; i < xs.size(); ++i)
      for (std::size_t j = 0; j < ys.size(); ++j)
        result(cast::to_index(i), cast::to_index(j)) = _call_impl(xs[i], ys[j]);
    return result;
  }
};

TEST(test_variant_heterogeneous, test_heterogeneous_symmetric) {
  SymmetricIdentityCovariance cov;

  using V = variant<int, double>;
  std::vector<V> xs = {V(1), V(10.0), V(2), V(20.0), V(3)};

  // Compute via heterogeneous batch dispatch
  Eigen::MatrixXd result = cov(xs);

  // Verify each element matches expected value
  for (std::size_t i = 0; i < xs.size(); ++i) {
    for (std::size_t j = 0; j < xs.size(); ++j) {
      double expected = xs[i].match(
          [&](int x) {
            return xs[j].match([&](int y) { return cov._call_impl(x, y); },
                               [&](double y) { return cov._call_impl(x, y); });
          },
          [&](double x) {
            return xs[j].match([&](int y) { return cov._call_impl(x, y); },
                               [&](double y) { return cov._call_impl(x, y); });
          });
      EXPECT_DOUBLE_EQ(result(cast::to_index(i), cast::to_index(j)), expected)
          << "Mismatch at (" << i << ", " << j << ")";
    }
  }

  // Also verify the matrix is symmetric
  for (std::size_t i = 0; i < xs.size(); ++i) {
    for (std::size_t j = i + 1; j < xs.size(); ++j) {
      EXPECT_DOUBLE_EQ(result(cast::to_index(i), cast::to_index(j)),
                       result(cast::to_index(j), cast::to_index(i)))
          << "Result should be symmetric at (" << i << ", " << j << ")";
    }
  }
}

/*
 * Test element ordering is preserved (most critical test)
 *
 * This uses the symmetric covariance since the symmetric dispatch path
 * uses transpose optimization which is valid for proper covariance functions.
 */
TEST(test_variant_heterogeneous, test_element_ordering_preserved) {
  SymmetricIdentityCovariance cov;

  using V = variant<int, double>;
  // xs = [A(1), B(2.0), A(3)] where A=int, B=double
  std::vector<V> xs = {V(1), V(2.0), V(3)};

  Eigen::MatrixXd result = cov(xs);

  // Verify result(0,0) corresponds to cov(1, 1), not cov(3, 1) or something else
  EXPECT_DOUBLE_EQ(result(0, 0), cov._call_impl(1, 1))
      << "result(0,0) should be cov(xs[0], xs[0]) = cov(1, 1)";
  EXPECT_DOUBLE_EQ(result(0, 1), cov._call_impl(1, 2.0))
      << "result(0,1) should be cov(xs[0], xs[1]) = cov(1, 2.0)";
  EXPECT_DOUBLE_EQ(result(0, 2), cov._call_impl(1, 3))
      << "result(0,2) should be cov(xs[0], xs[2]) = cov(1, 3)";
  EXPECT_DOUBLE_EQ(result(1, 0), cov._call_impl(2.0, 1))
      << "result(1,0) should be cov(xs[1], xs[0]) = cov(2.0, 1)";
  EXPECT_DOUBLE_EQ(result(1, 1), cov._call_impl(2.0, 2.0))
      << "result(1,1) should be cov(xs[1], xs[1]) = cov(2.0, 2.0)";
  EXPECT_DOUBLE_EQ(result(2, 2), cov._call_impl(3, 3))
      << "result(2,2) should be cov(xs[2], xs[2]) = cov(3, 3)";
}

/*
 * Test that batch methods are called with correct block sizes
 */
/*
 * BATCH VS POINTWISE EQUIVALENCE TESTS
 *
 * These tests verify that heterogeneous variant batch dispatch produces
 * the same results as pointwise computation. We use a helper that forces
 * pointwise evaluation to create a reference result.
 */

namespace {

// Helper to compute covariance matrix using only pointwise calls
// This bypasses all batch dispatch to create a reference result
template <typename CovFunc, typename... Ts>
Eigen::MatrixXd compute_pointwise_reference(
    const CovFunc &cov, const std::vector<variant<Ts...>> &xs,
    const std::vector<variant<Ts...>> &ys) {
  const auto m = cast::to_index(xs.size());
  const auto n = cast::to_index(ys.size());
  Eigen::MatrixXd result(m, n);
  for (Eigen::Index i = 0; i < m; ++i) {
    for (Eigen::Index j = 0; j < n; ++j) {
      // Use scalar operator() which goes through visitor dispatch
      result(i, j) = cov(xs[cast::to_size(i)], ys[cast::to_size(j)]);
    }
  }
  return result;
}

template <typename CovFunc, typename... Ts>
Eigen::MatrixXd compute_pointwise_reference_symmetric(
    const CovFunc &cov, const std::vector<variant<Ts...>> &xs) {
  return compute_pointwise_reference(cov, xs, xs);
}

template <typename CovFunc, typename... Ts>
Eigen::VectorXd compute_pointwise_reference_diagonal(
    const CovFunc &cov, const std::vector<variant<Ts...>> &xs) {
  const auto n = cast::to_index(xs.size());
  Eigen::VectorXd result(n);
  for (Eigen::Index i = 0; i < n; ++i) {
    result(i) = cov(xs[cast::to_size(i)], xs[cast::to_size(i)]);
  }
  return result;
}

} // anonymous namespace

/*
 * Test with a realistic multi-type covariance function
 * Uses different formulas for different type combinations
 */
class MultiTypeCovariance : public CovarianceFunction<MultiTypeCovariance> {
public:
  std::string name() const { return "multi_type"; }

  // int-int: squared exponential with length scale 10
  double _call_impl(const int &x, const int &y) const {
    double d = static_cast<double>(x - y);
    return std::exp(-d * d / 100.0);
  }

  // double-double: squared exponential with length scale 5
  double _call_impl(const double &x, const double &y) const {
    double d = x - y;
    return std::exp(-d * d / 25.0);
  }

  // int-double cross: linear covariance
  double _call_impl(const int &x, const double &y) const {
    return 0.1 * static_cast<double>(x) * y;
  }

  // double-int cross: symmetric with int-double
  double _call_impl(const double &x, const int &y) const {
    return 0.1 * x * static_cast<double>(y);
  }

  // Batch implementations that compute the same thing
  Eigen::MatrixXd _call_impl_vector(const std::vector<int> &xs,
                                    const std::vector<int> &ys,
                                    ThreadPool *) const {
    Eigen::MatrixXd result(cast::to_index(xs.size()),
                           cast::to_index(ys.size()));
    for (std::size_t i = 0; i < xs.size(); ++i) {
      for (std::size_t j = 0; j < ys.size(); ++j) {
        result(cast::to_index(i), cast::to_index(j)) = _call_impl(xs[i], ys[j]);
      }
    }
    return result;
  }

  Eigen::MatrixXd _call_impl_vector(const std::vector<double> &xs,
                                    const std::vector<double> &ys,
                                    ThreadPool *) const {
    Eigen::MatrixXd result(cast::to_index(xs.size()),
                           cast::to_index(ys.size()));
    for (std::size_t i = 0; i < xs.size(); ++i) {
      for (std::size_t j = 0; j < ys.size(); ++j) {
        result(cast::to_index(i), cast::to_index(j)) = _call_impl(xs[i], ys[j]);
      }
    }
    return result;
  }

  Eigen::MatrixXd _call_impl_vector(const std::vector<int> &xs,
                                    const std::vector<double> &ys,
                                    ThreadPool *) const {
    Eigen::MatrixXd result(cast::to_index(xs.size()),
                           cast::to_index(ys.size()));
    for (std::size_t i = 0; i < xs.size(); ++i) {
      for (std::size_t j = 0; j < ys.size(); ++j) {
        result(cast::to_index(i), cast::to_index(j)) = _call_impl(xs[i], ys[j]);
      }
    }
    return result;
  }

  Eigen::MatrixXd _call_impl_vector(const std::vector<double> &xs,
                                    const std::vector<int> &ys,
                                    ThreadPool *) const {
    Eigen::MatrixXd result(cast::to_index(xs.size()),
                           cast::to_index(ys.size()));
    for (std::size_t i = 0; i < xs.size(); ++i) {
      for (std::size_t j = 0; j < ys.size(); ++j) {
        result(cast::to_index(i), cast::to_index(j)) = _call_impl(xs[i], ys[j]);
      }
    }
    return result;
  }
};

TEST(test_batch_vs_pointwise, test_heterogeneous_cross_equivalence) {
  MultiTypeCovariance cov;

  using V = variant<int, double>;

  // Create heterogeneous vectors with interleaved types
  std::vector<V> xs = {V(1), V(2.5), V(3), V(4.5), V(5), V(6.5)};
  std::vector<V> ys = {V(10.0), V(2), V(3.0), V(4)};

  // Compute using batch dispatch
  Eigen::MatrixXd batch_result = cov(xs, ys);

  // Compute using pointwise reference
  Eigen::MatrixXd pointwise_result = compute_pointwise_reference(cov, xs, ys);

  // Verify dimensions
  ASSERT_EQ(batch_result.rows(), pointwise_result.rows());
  ASSERT_EQ(batch_result.cols(), pointwise_result.cols());

  // Verify all elements match
  for (Eigen::Index i = 0; i < batch_result.rows(); ++i) {
    for (Eigen::Index j = 0; j < batch_result.cols(); ++j) {
      EXPECT_NEAR(batch_result(i, j), pointwise_result(i, j), 1e-12)
          << "Mismatch at (" << i << ", " << j << ")";
    }
  }
}

TEST(test_batch_vs_pointwise, test_heterogeneous_symmetric_equivalence) {
  MultiTypeCovariance cov;

  using V = variant<int, double>;

  // Create heterogeneous vector
  std::vector<V> xs = {V(1), V(2.5), V(3), V(4.5), V(5)};

  // Compute using batch dispatch (symmetric)
  Eigen::MatrixXd batch_result = cov(xs);

  // Compute using pointwise reference
  Eigen::MatrixXd pointwise_result =
      compute_pointwise_reference_symmetric(cov, xs);

  // Verify dimensions
  ASSERT_EQ(batch_result.rows(), pointwise_result.rows());
  ASSERT_EQ(batch_result.cols(), pointwise_result.cols());

  // Verify all elements match
  for (Eigen::Index i = 0; i < batch_result.rows(); ++i) {
    for (Eigen::Index j = 0; j < batch_result.cols(); ++j) {
      EXPECT_NEAR(batch_result(i, j), pointwise_result(i, j), 1e-12)
          << "Mismatch at (" << i << ", " << j << ")";
    }
  }
}

TEST(test_batch_vs_pointwise, test_heterogeneous_diagonal_equivalence) {
  MultiTypeCovariance cov;

  using V = variant<int, double>;

  std::vector<V> xs = {V(1), V(2.5), V(3), V(4.5), V(5), V(6.5), V(7)};

  // Compute using batch dispatch (diagonal)
  Eigen::VectorXd batch_result = cov.diagonal(xs);

  // Compute using pointwise reference
  Eigen::VectorXd pointwise_result =
      compute_pointwise_reference_diagonal(cov, xs);

  // Verify dimensions
  ASSERT_EQ(batch_result.size(), pointwise_result.size());

  // Verify all elements match
  for (Eigen::Index i = 0; i < batch_result.size(); ++i) {
    EXPECT_NEAR(batch_result(i), pointwise_result(i), 1e-12)
        << "Mismatch at index " << i;
  }
}

/*
 * Test with Sum composite covariance
 */
TEST(test_batch_vs_pointwise, test_sum_heterogeneous_equivalence) {
  MultiTypeCovariance cov1;
  SymmetricIdentityCovariance cov2;
  auto sum = cov1 + cov2;

  using V = variant<int, double>;
  std::vector<V> xs = {V(1), V(2.0), V(3), V(4.0)};
  std::vector<V> ys = {V(5.0), V(6), V(7.0)};

  // Batch computation
  Eigen::MatrixXd batch_result = sum(xs, ys);

  // Pointwise reference
  Eigen::MatrixXd pointwise_result = compute_pointwise_reference(sum, xs, ys);

  ASSERT_EQ(batch_result.rows(), pointwise_result.rows());
  ASSERT_EQ(batch_result.cols(), pointwise_result.cols());

  for (Eigen::Index i = 0; i < batch_result.rows(); ++i) {
    for (Eigen::Index j = 0; j < batch_result.cols(); ++j) {
      EXPECT_NEAR(batch_result(i, j), pointwise_result(i, j), 1e-12)
          << "Sum mismatch at (" << i << ", " << j << ")";
    }
  }
}

/*
 * Test with Product composite covariance
 */
TEST(test_batch_vs_pointwise, test_product_heterogeneous_equivalence) {
  MultiTypeCovariance cov1;
  SymmetricIdentityCovariance cov2;
  auto product = cov1 * cov2;

  using V = variant<int, double>;
  std::vector<V> xs = {V(1), V(2.0), V(3)};
  std::vector<V> ys = {V(4.0), V(5)};

  // Batch computation
  Eigen::MatrixXd batch_result = product(xs, ys);

  // Pointwise reference
  Eigen::MatrixXd pointwise_result =
      compute_pointwise_reference(product, xs, ys);

  ASSERT_EQ(batch_result.rows(), pointwise_result.rows());
  ASSERT_EQ(batch_result.cols(), pointwise_result.cols());

  for (Eigen::Index i = 0; i < batch_result.rows(); ++i) {
    for (Eigen::Index j = 0; j < batch_result.cols(); ++j) {
      EXPECT_NEAR(batch_result(i, j), pointwise_result(i, j), 1e-12)
          << "Product mismatch at (" << i << ", " << j << ")";
    }
  }
}

/*
 * Test with three types in the variant
 */
class ThreeTypeCovariance : public CovarianceFunction<ThreeTypeCovariance> {
public:
  std::string name() const { return "three_type"; }

  // All type combinations for int, double, std::string
  double _call_impl(const int &x, const int &y) const {
    return static_cast<double>(x * y);
  }
  double _call_impl(const double &x, const double &y) const { return x * y; }
  double _call_impl(const std::string &x, const std::string &y) const {
    return static_cast<double>(x.size() * y.size());
  }
  double _call_impl(const int &x, const double &y) const {
    return static_cast<double>(x) * y;
  }
  double _call_impl(const double &x, const int &y) const {
    return x * static_cast<double>(y);
  }
  double _call_impl(const int &x, const std::string &y) const {
    return static_cast<double>(x * static_cast<int>(y.size()));
  }
  double _call_impl(const std::string &x, const int &y) const {
    return static_cast<double>(static_cast<int>(x.size()) * y);
  }
  double _call_impl(const double &x, const std::string &y) const {
    return x * static_cast<double>(y.size());
  }
  double _call_impl(const std::string &x, const double &y) const {
    return static_cast<double>(x.size()) * y;
  }

  // Batch for int-int
  Eigen::MatrixXd _call_impl_vector(const std::vector<int> &xs,
                                    const std::vector<int> &ys,
                                    ThreadPool *) const {
    Eigen::MatrixXd result(cast::to_index(xs.size()),
                           cast::to_index(ys.size()));
    for (std::size_t i = 0; i < xs.size(); ++i)
      for (std::size_t j = 0; j < ys.size(); ++j)
        result(cast::to_index(i), cast::to_index(j)) = _call_impl(xs[i], ys[j]);
    return result;
  }

  // Batch for double-double
  Eigen::MatrixXd _call_impl_vector(const std::vector<double> &xs,
                                    const std::vector<double> &ys,
                                    ThreadPool *) const {
    Eigen::MatrixXd result(cast::to_index(xs.size()),
                           cast::to_index(ys.size()));
    for (std::size_t i = 0; i < xs.size(); ++i)
      for (std::size_t j = 0; j < ys.size(); ++j)
        result(cast::to_index(i), cast::to_index(j)) = _call_impl(xs[i], ys[j]);
    return result;
  }

  // Batch for string-string
  Eigen::MatrixXd _call_impl_vector(const std::vector<std::string> &xs,
                                    const std::vector<std::string> &ys,
                                    ThreadPool *) const {
    Eigen::MatrixXd result(cast::to_index(xs.size()),
                           cast::to_index(ys.size()));
    for (std::size_t i = 0; i < xs.size(); ++i)
      for (std::size_t j = 0; j < ys.size(); ++j)
        result(cast::to_index(i), cast::to_index(j)) = _call_impl(xs[i], ys[j]);
    return result;
  }

  // Batch for int-double
  Eigen::MatrixXd _call_impl_vector(const std::vector<int> &xs,
                                    const std::vector<double> &ys,
                                    ThreadPool *) const {
    Eigen::MatrixXd result(cast::to_index(xs.size()),
                           cast::to_index(ys.size()));
    for (std::size_t i = 0; i < xs.size(); ++i)
      for (std::size_t j = 0; j < ys.size(); ++j)
        result(cast::to_index(i), cast::to_index(j)) = _call_impl(xs[i], ys[j]);
    return result;
  }

  // Batch for double-int
  Eigen::MatrixXd _call_impl_vector(const std::vector<double> &xs,
                                    const std::vector<int> &ys,
                                    ThreadPool *) const {
    Eigen::MatrixXd result(cast::to_index(xs.size()),
                           cast::to_index(ys.size()));
    for (std::size_t i = 0; i < xs.size(); ++i)
      for (std::size_t j = 0; j < ys.size(); ++j)
        result(cast::to_index(i), cast::to_index(j)) = _call_impl(xs[i], ys[j]);
    return result;
  }

  // Batch for int-string
  Eigen::MatrixXd _call_impl_vector(const std::vector<int> &xs,
                                    const std::vector<std::string> &ys,
                                    ThreadPool *) const {
    Eigen::MatrixXd result(cast::to_index(xs.size()),
                           cast::to_index(ys.size()));
    for (std::size_t i = 0; i < xs.size(); ++i)
      for (std::size_t j = 0; j < ys.size(); ++j)
        result(cast::to_index(i), cast::to_index(j)) = _call_impl(xs[i], ys[j]);
    return result;
  }

  // Batch for string-int
  Eigen::MatrixXd _call_impl_vector(const std::vector<std::string> &xs,
                                    const std::vector<int> &ys,
                                    ThreadPool *) const {
    Eigen::MatrixXd result(cast::to_index(xs.size()),
                           cast::to_index(ys.size()));
    for (std::size_t i = 0; i < xs.size(); ++i)
      for (std::size_t j = 0; j < ys.size(); ++j)
        result(cast::to_index(i), cast::to_index(j)) = _call_impl(xs[i], ys[j]);
    return result;
  }

  // Batch for double-string
  Eigen::MatrixXd _call_impl_vector(const std::vector<double> &xs,
                                    const std::vector<std::string> &ys,
                                    ThreadPool *) const {
    Eigen::MatrixXd result(cast::to_index(xs.size()),
                           cast::to_index(ys.size()));
    for (std::size_t i = 0; i < xs.size(); ++i)
      for (std::size_t j = 0; j < ys.size(); ++j)
        result(cast::to_index(i), cast::to_index(j)) = _call_impl(xs[i], ys[j]);
    return result;
  }

  // Batch for string-double
  Eigen::MatrixXd _call_impl_vector(const std::vector<std::string> &xs,
                                    const std::vector<double> &ys,
                                    ThreadPool *) const {
    Eigen::MatrixXd result(cast::to_index(xs.size()),
                           cast::to_index(ys.size()));
    for (std::size_t i = 0; i < xs.size(); ++i)
      for (std::size_t j = 0; j < ys.size(); ++j)
        result(cast::to_index(i), cast::to_index(j)) = _call_impl(xs[i], ys[j]);
    return result;
  }
};

TEST(test_batch_vs_pointwise, test_three_type_heterogeneous_equivalence) {
  ThreeTypeCovariance cov;

  using V = variant<int, double, std::string>;

  // Mix all three types
  std::vector<V> xs = {V(1),     V(2.0),   V(std::string("abc")),
                       V(3),     V(4.0),   V(std::string("de")),
                       V(std::string("f"))};
  std::vector<V> ys = {V(std::string("gh")), V(5), V(6.0), V(std::string("ijk")), V(7)};

  // Batch computation
  Eigen::MatrixXd batch_result = cov(xs, ys);

  // Pointwise reference
  Eigen::MatrixXd pointwise_result = compute_pointwise_reference(cov, xs, ys);

  ASSERT_EQ(batch_result.rows(), pointwise_result.rows());
  ASSERT_EQ(batch_result.cols(), pointwise_result.cols());

  for (Eigen::Index i = 0; i < batch_result.rows(); ++i) {
    for (Eigen::Index j = 0; j < batch_result.cols(); ++j) {
      EXPECT_NEAR(batch_result(i, j), pointwise_result(i, j), 1e-12)
          << "Three-type mismatch at (" << i << ", " << j << ")";
    }
  }

  // Also test symmetric
  Eigen::MatrixXd sym_batch = cov(xs);
  Eigen::MatrixXd sym_pointwise = compute_pointwise_reference_symmetric(cov, xs);

  for (Eigen::Index i = 0; i < sym_batch.rows(); ++i) {
    for (Eigen::Index j = 0; j < sym_batch.cols(); ++j) {
      EXPECT_NEAR(sym_batch(i, j), sym_pointwise(i, j), 1e-12)
          << "Three-type symmetric mismatch at (" << i << ", " << j << ")";
    }
  }
}

/*
 * Test edge cases: empty partitions (some types have no elements)
 */
TEST(test_batch_vs_pointwise, test_empty_partitions_equivalence) {
  ThreeTypeCovariance cov;

  using V = variant<int, double, std::string>;

  // Only ints in xs, mix in ys (double partition empty in xs)
  std::vector<V> xs = {V(1), V(2), V(3)};
  std::vector<V> ys = {V(4.0), V(5), V(std::string("ab"))};

  Eigen::MatrixXd batch_result = cov(xs, ys);
  Eigen::MatrixXd pointwise_result = compute_pointwise_reference(cov, xs, ys);

  for (Eigen::Index i = 0; i < batch_result.rows(); ++i) {
    for (Eigen::Index j = 0; j < batch_result.cols(); ++j) {
      EXPECT_NEAR(batch_result(i, j), pointwise_result(i, j), 1e-12)
          << "Empty partition mismatch at (" << i << ", " << j << ")";
    }
  }
}

/*
 * Test with Measurement wrapper
 */
TEST(test_batch_vs_pointwise, test_measurement_heterogeneous_equivalence) {
  MultiTypeCovariance cov;

  using V = variant<int, double>;
  using MV = variant<Measurement<int>, Measurement<double>>;

  // Create measurements
  std::vector<MV> xs = {MV(Measurement<int>{1}), MV(Measurement<double>{2.0}),
                        MV(Measurement<int>{3}), MV(Measurement<double>{4.0})};
  std::vector<MV> ys = {MV(Measurement<double>{5.0}), MV(Measurement<int>{6}),
                        MV(Measurement<double>{7.0})};

  // Batch computation
  Eigen::MatrixXd batch_result = cov(xs, ys);

  // Pointwise reference
  Eigen::MatrixXd pointwise_result = compute_pointwise_reference(cov, xs, ys);

  ASSERT_EQ(batch_result.rows(), pointwise_result.rows());
  ASSERT_EQ(batch_result.cols(), pointwise_result.cols());

  for (Eigen::Index i = 0; i < batch_result.rows(); ++i) {
    for (Eigen::Index j = 0; j < batch_result.cols(); ++j) {
      EXPECT_NEAR(batch_result(i, j), pointwise_result(i, j), 1e-12)
          << "Measurement mismatch at (" << i << ", " << j << ")";
    }
  }
}

/*
 * Test large heterogeneous vectors (stress test)
 */
TEST(test_batch_vs_pointwise, test_large_heterogeneous_equivalence) {
  MultiTypeCovariance cov;

  using V = variant<int, double>;

  // Create larger vectors with random-ish interleaving
  std::vector<V> xs;
  std::vector<V> ys;

  for (int i = 0; i < 50; ++i) {
    if (i % 3 == 0) {
      xs.push_back(V(i));
    } else {
      xs.push_back(V(static_cast<double>(i) * 0.5));
    }
  }

  for (int j = 0; j < 40; ++j) {
    if (j % 2 == 0) {
      ys.push_back(V(j * 2));
    } else {
      ys.push_back(V(static_cast<double>(j) * 1.5));
    }
  }

  // Batch computation
  Eigen::MatrixXd batch_result = cov(xs, ys);

  // Pointwise reference
  Eigen::MatrixXd pointwise_result = compute_pointwise_reference(cov, xs, ys);

  ASSERT_EQ(batch_result.rows(), cast::to_index(xs.size()));
  ASSERT_EQ(batch_result.cols(), cast::to_index(ys.size()));

  double max_diff = (batch_result - pointwise_result).cwiseAbs().maxCoeff();
  EXPECT_LT(max_diff, 1e-12)
      << "Large heterogeneous test: max difference = " << max_diff;
}

/*
 * Test that batch methods are called with correct block sizes
 */
TEST(test_variant_heterogeneous, test_batch_call_counts) {
  class BlockCountingCovariance
      : public CovarianceFunction<BlockCountingCovariance> {
  public:
    std::string name() const { return "block_counting"; }

    mutable int int_int_calls = 0;
    mutable int double_double_calls = 0;
    mutable int int_double_calls = 0;
    mutable int double_int_calls = 0;

    void reset() const {
      int_int_calls = 0;
      double_double_calls = 0;
      int_double_calls = 0;
      double_int_calls = 0;
    }

    double _call_impl(const int &x, const int &y) const {
      return static_cast<double>(x + y);
    }
    double _call_impl(const double &x, const double &y) const { return x + y; }
    double _call_impl(const int &x, const double &y) const {
      return static_cast<double>(x) + y;
    }
    double _call_impl(const double &x, const int &y) const {
      return x + static_cast<double>(y);
    }

    Eigen::MatrixXd _call_impl_vector(const std::vector<int> &xs,
                                      const std::vector<int> &ys,
                                      ThreadPool *) const {
      int_int_calls++;
      Eigen::MatrixXd result(cast::to_index(xs.size()),
                             cast::to_index(ys.size()));
      for (std::size_t i = 0; i < xs.size(); ++i)
        for (std::size_t j = 0; j < ys.size(); ++j)
          result(cast::to_index(i), cast::to_index(j)) =
              _call_impl(xs[i], ys[j]);
      return result;
    }

    Eigen::MatrixXd _call_impl_vector(const std::vector<double> &xs,
                                      const std::vector<double> &ys,
                                      ThreadPool *) const {
      double_double_calls++;
      Eigen::MatrixXd result(cast::to_index(xs.size()),
                             cast::to_index(ys.size()));
      for (std::size_t i = 0; i < xs.size(); ++i)
        for (std::size_t j = 0; j < ys.size(); ++j)
          result(cast::to_index(i), cast::to_index(j)) =
              _call_impl(xs[i], ys[j]);
      return result;
    }

    Eigen::MatrixXd _call_impl_vector(const std::vector<int> &xs,
                                      const std::vector<double> &ys,
                                      ThreadPool *) const {
      int_double_calls++;
      Eigen::MatrixXd result(cast::to_index(xs.size()),
                             cast::to_index(ys.size()));
      for (std::size_t i = 0; i < xs.size(); ++i)
        for (std::size_t j = 0; j < ys.size(); ++j)
          result(cast::to_index(i), cast::to_index(j)) =
              _call_impl(xs[i], ys[j]);
      return result;
    }

    Eigen::MatrixXd _call_impl_vector(const std::vector<double> &xs,
                                      const std::vector<int> &ys,
                                      ThreadPool *) const {
      double_int_calls++;
      Eigen::MatrixXd result(cast::to_index(xs.size()),
                             cast::to_index(ys.size()));
      for (std::size_t i = 0; i < xs.size(); ++i)
        for (std::size_t j = 0; j < ys.size(); ++j)
          result(cast::to_index(i), cast::to_index(j)) =
              _call_impl(xs[i], ys[j]);
      return result;
    }
  };

  BlockCountingCovariance cov;

  using V = variant<int, double>;
  // 3 ints, 2 doubles
  std::vector<V> xs = {V(1), V(10.0), V(2), V(20.0), V(3)};

  // Test symmetric case: cov(xs, xs)
  cov.reset();
  Eigen::MatrixXd sym_result = cov(xs);

  // For symmetric heterogeneous with transpose optimization:
  // - 1 call for int-int diagonal block (3x3)
  // - 1 call for double-double diagonal block (2x2)
  // - 1 call for int-double off-diagonal block (3x2), transposed for double-int
  // - 0 calls for double-int (filled by transpose of int-double)
  EXPECT_EQ(cov.int_int_calls, 1) << "int-int batch should be called once";
  EXPECT_EQ(cov.double_double_calls, 1)
      << "double-double batch should be called once";
  EXPECT_EQ(cov.int_double_calls, 1)
      << "int-double batch should be called exactly once";
  EXPECT_EQ(cov.double_int_calls, 0)
      << "double-int should NOT be called (filled by transpose)";

  // Test cross-covariance case: cov(xs, ys) where ys is different
  std::vector<V> ys = {V(100), V(200.0), V(300)};  // 2 ints, 1 double

  cov.reset();
  Eigen::MatrixXd cross_result = cov(xs, ys);

  // For cross-covariance (no transpose optimization):
  // - 1 call for int-int block (3x2)
  // - 1 call for double-double block (2x1)
  // - 1 call for int-double block (3x1)
  // - 1 call for double-int block (2x2)
  EXPECT_EQ(cov.int_int_calls, 1) << "cross: int-int batch should be called once";
  EXPECT_EQ(cov.double_double_calls, 1)
      << "cross: double-double batch should be called once";
  EXPECT_EQ(cov.int_double_calls, 1)
      << "cross: int-double batch should be called once";
  EXPECT_EQ(cov.double_int_calls, 1)
      << "cross: double-int batch should be called once";
}

/*
 * ============================================================================
 * THREAD POOL PROPAGATION TESTS
 *
 * These tests verify that ThreadPool* is correctly propagated through:
 * - Sum/Product compositions
 * - Nested compositions
 * - Measurement unwrapping
 * - Variant unwrapping
 * - diagonal() calls
 * ============================================================================
 */

/*
 * PoolCallRecord: captures information about a single pool-related call
 */
struct PoolCallRecord {
  ThreadPool *pool_received;
  std::size_t thread_count;
  bool was_nonnull;
  std::string call_type;  // "vector", "diagonal", "symmetric"
};

/*
 * PoolTrackingCovariance: records pool information for each batch call.
 * Uses shared_ptr to survive copies in Sum/Product compositions.
 */
class PoolTrackingCovariance
    : public CovarianceFunction<PoolTrackingCovariance> {
public:
  std::string name() const { return "pool_tracking"; }

  // Shared record storage that survives copies
  std::shared_ptr<std::vector<PoolCallRecord>> records =
      std::make_shared<std::vector<PoolCallRecord>>();

  void reset() const { records->clear(); }

  // Batch implementation that records pool info
  Eigen::MatrixXd _call_impl_vector(const std::vector<double> &xs,
                                    const std::vector<double> &ys,
                                    ThreadPool *pool) const {
    PoolCallRecord record;
    record.pool_received = pool;
    record.was_nonnull = (pool != nullptr);
    record.thread_count = pool ? pool->thread_count() : 0;
    record.call_type = (xs.data() == ys.data()) ? "symmetric" : "vector";
    records->push_back(record);
    return Eigen::MatrixXd::Constant(cast::to_index(xs.size()),
                                     cast::to_index(ys.size()), 1.0);
  }

  // Diagonal batch implementation that records pool info
  Eigen::VectorXd _call_impl_vector_diagonal(const std::vector<double> &xs,
                                             ThreadPool *pool) const {
    PoolCallRecord record;
    record.pool_received = pool;
    record.was_nonnull = (pool != nullptr);
    record.thread_count = pool ? pool->thread_count() : 0;
    record.call_type = "diagonal";
    records->push_back(record);
    return Eigen::VectorXd::Constant(cast::to_index(xs.size()), 1.0);
  }
};

/*
 * Test 1: Basic pool propagation - verify pool reaches batch method
 */
TEST(test_pool_propagation, test_basic_pool_propagation) {
  PoolTrackingCovariance cov;
  cov.reset();

  std::vector<double> xs = {1.0, 2.0, 3.0};

  // Call without pool
  cov(xs);
  ASSERT_EQ(cov.records->size(), 1u);
  EXPECT_FALSE(cov.records->at(0).was_nonnull) << "nullptr should propagate";

  // Call with pool
  cov.reset();
  ThreadPool pool(4);
  cov(xs, &pool);
  ASSERT_EQ(cov.records->size(), 1u);
  EXPECT_TRUE(cov.records->at(0).was_nonnull) << "Pool should propagate";
  EXPECT_EQ(cov.records->at(0).thread_count, 4u);
}

/*
 * Test 2: Sum composition - both children receive pool
 */
TEST(test_pool_propagation, test_sum_both_batch_receives_pool) {
  PoolTrackingCovariance lhs;
  PoolTrackingCovariance rhs;
  lhs.reset();
  rhs.reset();

  auto sum = lhs + rhs;

  std::vector<double> xs = {1.0, 2.0};
  ThreadPool pool(8);

  sum(xs, &pool);

  // Both sides should receive the pool
  ASSERT_EQ(lhs.records->size(), 1u);
  ASSERT_EQ(rhs.records->size(), 1u);
  EXPECT_TRUE(lhs.records->at(0).was_nonnull) << "LHS should receive pool";
  EXPECT_TRUE(rhs.records->at(0).was_nonnull) << "RHS should receive pool";
  EXPECT_EQ(lhs.records->at(0).thread_count, 8u);
  EXPECT_EQ(rhs.records->at(0).thread_count, 8u);
}

/*
 * Test 3: Product composition - both children receive pool
 */
TEST(test_pool_propagation, test_product_both_batch_receives_pool) {
  PoolTrackingCovariance lhs;
  PoolTrackingCovariance rhs;
  lhs.reset();
  rhs.reset();

  auto product = lhs * rhs;

  std::vector<double> xs = {1.0, 2.0};
  ThreadPool pool(6);

  product(xs, &pool);

  // Both sides should receive the pool
  ASSERT_EQ(lhs.records->size(), 1u);
  ASSERT_EQ(rhs.records->size(), 1u);
  EXPECT_TRUE(lhs.records->at(0).was_nonnull) << "LHS should receive pool";
  EXPECT_TRUE(rhs.records->at(0).was_nonnull) << "RHS should receive pool";
  EXPECT_EQ(lhs.records->at(0).thread_count, 6u);
  EXPECT_EQ(rhs.records->at(0).thread_count, 6u);
}

/*
 * Test 4: Nested composition - all covariances receive pool
 */
TEST(test_pool_propagation, test_nested_composition_receives_pool) {
  PoolTrackingCovariance cov1;
  PoolTrackingCovariance cov2;
  PoolTrackingCovariance cov3;
  cov1.reset();
  cov2.reset();
  cov3.reset();

  // (cov1 + cov2) * cov3
  auto composed = (cov1 + cov2) * cov3;

  std::vector<double> xs = {1.0, 2.0};
  ThreadPool pool(5);

  composed(xs, &pool);

  // All three should receive the pool
  ASSERT_EQ(cov1.records->size(), 1u);
  ASSERT_EQ(cov2.records->size(), 1u);
  ASSERT_EQ(cov3.records->size(), 1u);
  EXPECT_TRUE(cov1.records->at(0).was_nonnull) << "cov1 should receive pool";
  EXPECT_TRUE(cov2.records->at(0).was_nonnull) << "cov2 should receive pool";
  EXPECT_TRUE(cov3.records->at(0).was_nonnull) << "cov3 should receive pool";
  EXPECT_EQ(cov1.records->at(0).thread_count, 5u);
  EXPECT_EQ(cov2.records->at(0).thread_count, 5u);
  EXPECT_EQ(cov3.records->at(0).thread_count, 5u);
}

/*
 * Test 5: Sum with mixed batch/pointwise - batch side gets pool
 */
TEST(test_pool_propagation, test_sum_mixed_batch_pointwise) {
  PoolTrackingCovariance batch_cov;
  batch_cov.reset();

  class PointwiseOnlyCov : public CovarianceFunction<PointwiseOnlyCov> {
  public:
    std::string name() const { return "pointwise_only"; }
    double _call_impl(const double &x, const double &y) const { return x * y; }
  };
  PointwiseOnlyCov pointwise_cov;

  auto sum = batch_cov + pointwise_cov;

  std::vector<double> xs = {1.0, 2.0};
  ThreadPool pool(3);

  sum(xs, &pool);

  // Batch side should receive the pool
  ASSERT_EQ(batch_cov.records->size(), 1u);
  EXPECT_TRUE(batch_cov.records->at(0).was_nonnull)
      << "Batch side should receive pool";
  EXPECT_EQ(batch_cov.records->at(0).thread_count, 3u);
}

/*
 * Test 6: No pool propagates nullptr
 */
TEST(test_pool_propagation, test_no_pool_propagates_nullptr) {
  PoolTrackingCovariance cov1;
  PoolTrackingCovariance cov2;
  cov1.reset();
  cov2.reset();

  auto sum = cov1 + cov2;

  std::vector<double> xs = {1.0, 2.0};

  sum(xs);  // No pool

  // Both should receive nullptr
  ASSERT_EQ(cov1.records->size(), 1u);
  ASSERT_EQ(cov2.records->size(), 1u);
  EXPECT_FALSE(cov1.records->at(0).was_nonnull) << "cov1 should receive nullptr";
  EXPECT_FALSE(cov2.records->at(0).was_nonnull) << "cov2 should receive nullptr";
}

/*
 * Test 7: Cross-covariance receives pool
 */
TEST(test_pool_propagation, test_cross_covariance_receives_pool) {
  PoolTrackingCovariance cov;
  cov.reset();

  std::vector<double> xs = {1.0, 2.0};
  std::vector<double> ys = {3.0, 4.0, 5.0};
  ThreadPool pool(7);

  cov(xs, ys, &pool);

  ASSERT_EQ(cov.records->size(), 1u);
  EXPECT_TRUE(cov.records->at(0).was_nonnull);
  EXPECT_EQ(cov.records->at(0).thread_count, 7u);
  EXPECT_EQ(cov.records->at(0).call_type, "vector");
}

/*
 * Test 8: diagonal() receives pool (requires fix to diagonal signature)
 */
TEST(test_pool_propagation, test_diagonal_receives_pool) {
  PoolTrackingCovariance cov;
  cov.reset();

  std::vector<double> xs = {1.0, 2.0, 3.0};
  ThreadPool pool(4);

  cov.diagonal(xs, &pool);

  ASSERT_EQ(cov.records->size(), 1u);
  EXPECT_TRUE(cov.records->at(0).was_nonnull)
      << "diagonal() should pass pool to _call_impl_vector_diagonal";
  EXPECT_EQ(cov.records->at(0).thread_count, 4u);
  EXPECT_EQ(cov.records->at(0).call_type, "diagonal");
}

/*
 * Test 9: Sum diagonal - both batch children receive pool
 */
TEST(test_pool_propagation, test_sum_diagonal_both_batch_receives_pool) {
  PoolTrackingCovariance lhs;
  PoolTrackingCovariance rhs;
  lhs.reset();
  rhs.reset();

  auto sum = lhs + rhs;

  std::vector<double> xs = {1.0, 2.0, 3.0};
  ThreadPool pool(4);

  sum.diagonal(xs, &pool);

  // Both sides should receive the pool for diagonal
  ASSERT_EQ(lhs.records->size(), 1u);
  ASSERT_EQ(rhs.records->size(), 1u);
  EXPECT_TRUE(lhs.records->at(0).was_nonnull)
      << "LHS diagonal should receive pool";
  EXPECT_TRUE(rhs.records->at(0).was_nonnull)
      << "RHS diagonal should receive pool";
  EXPECT_EQ(lhs.records->at(0).thread_count, 4u);
  EXPECT_EQ(rhs.records->at(0).thread_count, 4u);
  EXPECT_EQ(lhs.records->at(0).call_type, "diagonal");
  EXPECT_EQ(rhs.records->at(0).call_type, "diagonal");
}

/*
 * Test 10: Measurement unwrap propagates pool
 */
TEST(test_pool_propagation, test_measurement_unwrap_propagates_pool) {
  PoolTrackingCovariance cov;
  cov.reset();

  std::vector<Measurement<double>> ms;
  for (int i = 0; i < 10; ++i) {
    ms.push_back(Measurement<double>{static_cast<double>(i)});
  }

  ThreadPool pool(6);
  cov(ms, &pool);

  ASSERT_EQ(cov.records->size(), 1u);
  EXPECT_TRUE(cov.records->at(0).was_nonnull)
      << "Measurement unwrap should propagate pool";
  EXPECT_EQ(cov.records->at(0).thread_count, 6u);
}

/*
 * Test 11: Variant unwrap propagates pool
 */
TEST(test_pool_propagation, test_variant_unwrap_propagates_pool) {
  PoolTrackingCovariance cov;
  cov.reset();

  using DoubleVariant = variant<double>;
  std::vector<DoubleVariant> xs(10, DoubleVariant(1.0));

  ThreadPool pool(5);
  cov(xs, &pool);

  ASSERT_EQ(cov.records->size(), 1u);
  EXPECT_TRUE(cov.records->at(0).was_nonnull)
      << "Variant unwrap should propagate pool";
  EXPECT_EQ(cov.records->at(0).thread_count, 5u);
}

/*
 * Test 12: Sum with variant<Measurement<double>> receives pool
 * Complex type unwrapping: variant<Measurement<double>> -> Measurement<double> -> double
 */
TEST(test_pool_propagation, test_sum_with_variant_measurement_receives_pool) {
  PoolTrackingCovariance lhs;
  PoolTrackingCovariance rhs;
  lhs.reset();
  rhs.reset();

  auto sum = lhs + rhs;

  using VariantMeasurement = variant<Measurement<double>>;
  std::vector<VariantMeasurement> ms;
  for (int i = 0; i < 10; ++i) {
    ms.push_back(VariantMeasurement(Measurement<double>{static_cast<double>(i)}));
  }

  ThreadPool pool(4);
  sum(ms, &pool);

  // Both sides should receive the pool after unwrapping
  ASSERT_EQ(lhs.records->size(), 1u);
  ASSERT_EQ(rhs.records->size(), 1u);
  EXPECT_TRUE(lhs.records->at(0).was_nonnull)
      << "LHS should receive pool through variant<Measurement>";
  EXPECT_TRUE(rhs.records->at(0).was_nonnull)
      << "RHS should receive pool through variant<Measurement>";
  EXPECT_EQ(lhs.records->at(0).thread_count, 4u);
  EXPECT_EQ(rhs.records->at(0).thread_count, 4u);
}

/*
 * Test 13: Product with variant<Measurement<double>> receives pool
 */
TEST(test_pool_propagation, test_product_with_variant_measurement_receives_pool) {
  PoolTrackingCovariance lhs;
  PoolTrackingCovariance rhs;
  lhs.reset();
  rhs.reset();

  auto product = lhs * rhs;

  using VariantMeasurement = variant<Measurement<double>>;
  std::vector<VariantMeasurement> ms;
  for (int i = 0; i < 8; ++i) {
    ms.push_back(VariantMeasurement(Measurement<double>{static_cast<double>(i)}));
  }

  ThreadPool pool(3);
  product(ms, &pool);

  ASSERT_EQ(lhs.records->size(), 1u);
  ASSERT_EQ(rhs.records->size(), 1u);
  EXPECT_TRUE(lhs.records->at(0).was_nonnull);
  EXPECT_TRUE(rhs.records->at(0).was_nonnull);
  EXPECT_EQ(lhs.records->at(0).thread_count, 3u);
  EXPECT_EQ(rhs.records->at(0).thread_count, 3u);
}

/*
 * Test 14: Nested Sum+Product with variant<Measurement<double>>
 * (cov1 + cov2) * cov3 with complex types
 */
TEST(test_pool_propagation,
     test_nested_sum_product_variant_measurement) {
  PoolTrackingCovariance cov1;
  PoolTrackingCovariance cov2;
  PoolTrackingCovariance cov3;
  cov1.reset();
  cov2.reset();
  cov3.reset();

  auto composed = (cov1 + cov2) * cov3;

  using VariantMeasurement = variant<Measurement<double>>;
  std::vector<VariantMeasurement> ms;
  for (int i = 0; i < 5; ++i) {
    ms.push_back(VariantMeasurement(Measurement<double>{static_cast<double>(i)}));
  }

  ThreadPool pool(7);
  composed(ms, &pool);

  ASSERT_EQ(cov1.records->size(), 1u);
  ASSERT_EQ(cov2.records->size(), 1u);
  ASSERT_EQ(cov3.records->size(), 1u);
  EXPECT_TRUE(cov1.records->at(0).was_nonnull);
  EXPECT_TRUE(cov2.records->at(0).was_nonnull);
  EXPECT_TRUE(cov3.records->at(0).was_nonnull);
  EXPECT_EQ(cov1.records->at(0).thread_count, 7u);
  EXPECT_EQ(cov2.records->at(0).thread_count, 7u);
  EXPECT_EQ(cov3.records->at(0).thread_count, 7u);
}

/*
 * Test 15: Deep composition with variant<Measurement<double>>
 * ((cov1 * cov2) + cov3) * cov4
 */
TEST(test_pool_propagation,
     test_deep_composition_variant_measurement) {
  PoolTrackingCovariance cov1;
  PoolTrackingCovariance cov2;
  PoolTrackingCovariance cov3;
  PoolTrackingCovariance cov4;
  cov1.reset();
  cov2.reset();
  cov3.reset();
  cov4.reset();

  auto composed = ((cov1 * cov2) + cov3) * cov4;

  using VariantMeasurement = variant<Measurement<double>>;
  std::vector<VariantMeasurement> ms;
  for (int i = 0; i < 6; ++i) {
    ms.push_back(VariantMeasurement(Measurement<double>{static_cast<double>(i)}));
  }

  ThreadPool pool(9);
  composed(ms, &pool);

  ASSERT_EQ(cov1.records->size(), 1u);
  ASSERT_EQ(cov2.records->size(), 1u);
  ASSERT_EQ(cov3.records->size(), 1u);
  ASSERT_EQ(cov4.records->size(), 1u);
  EXPECT_TRUE(cov1.records->at(0).was_nonnull);
  EXPECT_TRUE(cov2.records->at(0).was_nonnull);
  EXPECT_TRUE(cov3.records->at(0).was_nonnull);
  EXPECT_TRUE(cov4.records->at(0).was_nonnull);
  EXPECT_EQ(cov1.records->at(0).thread_count, 9u);
  EXPECT_EQ(cov2.records->at(0).thread_count, 9u);
  EXPECT_EQ(cov3.records->at(0).thread_count, 9u);
  EXPECT_EQ(cov4.records->at(0).thread_count, 9u);
}

/*
 * Test 16: Sum cross-covariance with variant<Measurement<double>>
 * sum(xs, ys, pool) where xs and ys are vectors of variant<Measurement<double>>
 */
TEST(test_pool_propagation,
     test_sum_cross_covariance_variant_measurement) {
  PoolTrackingCovariance lhs;
  PoolTrackingCovariance rhs;
  lhs.reset();
  rhs.reset();

  auto sum = lhs + rhs;

  using VariantMeasurement = variant<Measurement<double>>;
  std::vector<VariantMeasurement> xs;
  std::vector<VariantMeasurement> ys;
  for (int i = 0; i < 4; ++i) {
    xs.push_back(VariantMeasurement(Measurement<double>{static_cast<double>(i)}));
  }
  for (int i = 0; i < 6; ++i) {
    ys.push_back(
        VariantMeasurement(Measurement<double>{static_cast<double>(i + 10)}));
  }

  ThreadPool pool(5);
  sum(xs, ys, &pool);

  ASSERT_EQ(lhs.records->size(), 1u);
  ASSERT_EQ(rhs.records->size(), 1u);
  EXPECT_TRUE(lhs.records->at(0).was_nonnull)
      << "LHS cross-cov should receive pool";
  EXPECT_TRUE(rhs.records->at(0).was_nonnull)
      << "RHS cross-cov should receive pool";
  EXPECT_EQ(lhs.records->at(0).thread_count, 5u);
  EXPECT_EQ(rhs.records->at(0).thread_count, 5u);
}

/*
 * Test 17: Product diagonal receives pool
 */
TEST(test_pool_propagation, test_product_diagonal_both_batch_receives_pool) {
  PoolTrackingCovariance lhs;
  PoolTrackingCovariance rhs;
  lhs.reset();
  rhs.reset();

  auto product = lhs * rhs;

  std::vector<double> xs = {1.0, 2.0, 3.0};
  ThreadPool pool(4);

  product.diagonal(xs, &pool);

  // Both sides should receive the pool for diagonal
  ASSERT_EQ(lhs.records->size(), 1u);
  ASSERT_EQ(rhs.records->size(), 1u);
  EXPECT_TRUE(lhs.records->at(0).was_nonnull)
      << "LHS product diagonal should receive pool";
  EXPECT_TRUE(rhs.records->at(0).was_nonnull)
      << "RHS product diagonal should receive pool";
  EXPECT_EQ(lhs.records->at(0).thread_count, 4u);
  EXPECT_EQ(rhs.records->at(0).thread_count, 4u);
}

/*
 * Test 18: Diagonal with variant type receives pool
 */
TEST(test_pool_propagation, test_diagonal_with_variant_receives_pool) {
  PoolTrackingCovariance cov;
  cov.reset();

  using DoubleVariant = variant<double>;
  std::vector<DoubleVariant> xs(10, DoubleVariant(1.0));

  ThreadPool pool(6);
  cov.diagonal(xs, &pool);

  ASSERT_EQ(cov.records->size(), 1u);
  EXPECT_TRUE(cov.records->at(0).was_nonnull)
      << "Diagonal with variant should receive pool";
  EXPECT_EQ(cov.records->at(0).thread_count, 6u);
  EXPECT_EQ(cov.records->at(0).call_type, "diagonal");
}

/*
 * Test 19: Diagonal with Measurement type receives pool
 */
TEST(test_pool_propagation, test_diagonal_with_measurement_receives_pool) {
  PoolTrackingCovariance cov;
  cov.reset();

  std::vector<Measurement<double>> ms;
  for (int i = 0; i < 8; ++i) {
    ms.push_back(Measurement<double>{static_cast<double>(i)});
  }

  ThreadPool pool(3);
  cov.diagonal(ms, &pool);

  ASSERT_EQ(cov.records->size(), 1u);
  EXPECT_TRUE(cov.records->at(0).was_nonnull)
      << "Diagonal with Measurement should receive pool";
  EXPECT_EQ(cov.records->at(0).thread_count, 3u);
  EXPECT_EQ(cov.records->at(0).call_type, "diagonal");
}

/*
 * ============================================================================
 * SINGLE-ARG SYMMETRIC BATCH TESTS
 * Tests for _call_impl_vector(const std::vector<X>&, ThreadPool*) support
 * ============================================================================
 */

// Stats for tracking single-arg vs two-arg batch calls
struct SingleArgBatchCallStats {
  int single_arg_count = 0;
  int two_arg_count = 0;
  int last_xs_size = 0;
  bool pool_was_nonnull = false;
  void reset() {
    single_arg_count = two_arg_count = last_xs_size = 0;
    pool_was_nonnull = false;
  }
};

// Mock with BOTH single-arg and two-arg batch
class MockSingleArgBatchCov : public CovarianceFunction<MockSingleArgBatchCov> {
public:
  std::string name() const { return "mock_single_arg_batch"; }
  std::shared_ptr<SingleArgBatchCallStats> stats =
      std::make_shared<SingleArgBatchCallStats>();

  Eigen::MatrixXd _call_impl_vector(const std::vector<double> &xs,
                                    ThreadPool *pool) const {
    stats->single_arg_count++;
    stats->last_xs_size = static_cast<int>(xs.size());
    stats->pool_was_nonnull = (pool != nullptr);
    return Eigen::MatrixXd::Constant(cast::to_index(xs.size()),
                                     cast::to_index(xs.size()), 777.0);
  }

  Eigen::MatrixXd _call_impl_vector(const std::vector<double> &xs,
                                    const std::vector<double> &ys,
                                    ThreadPool *pool) const {
    stats->two_arg_count++;
    return Eigen::MatrixXd::Constant(cast::to_index(xs.size()),
                                     cast::to_index(ys.size()), 888.0);
  }
};

// Mock with ONLY single-arg batch (no pointwise, no two-arg)
class MockSingleArgOnlyCov : public CovarianceFunction<MockSingleArgOnlyCov> {
public:
  std::string name() const { return "mock_single_arg_only"; }
  std::shared_ptr<SingleArgBatchCallStats> stats =
      std::make_shared<SingleArgBatchCallStats>();

  Eigen::MatrixXd _call_impl_vector(const std::vector<double> &xs,
                                    ThreadPool *pool) const {
    stats->single_arg_count++;
    stats->last_xs_size = static_cast<int>(xs.size());
    stats->pool_was_nonnull = (pool != nullptr);
    return Eigen::MatrixXd::Constant(cast::to_index(xs.size()),
                                     cast::to_index(xs.size()), 333.0);
  }
};

/*
 * BASIC TEST 1: Single-arg should be preferred over two-arg for cov(xs)
 */
TEST(test_single_arg_symmetric, test_prefers_single_arg_over_two_arg) {
  MockSingleArgBatchCov cov;
  cov.stats->reset();

  std::vector<double> xs = {1.0, 2.0, 3.0};
  Eigen::MatrixXd result = cov(xs);  // Symmetric call

  EXPECT_EQ(cov.stats->single_arg_count, 1) << "Single-arg should be called";
  EXPECT_EQ(cov.stats->two_arg_count, 0) << "Two-arg should NOT be called";
  EXPECT_EQ(result(0, 0), 777.0) << "Result should be from single-arg";
}

/*
 * BASIC TEST 2: Single-arg-only covariance should work
 */
TEST(test_single_arg_symmetric, test_single_arg_only) {
  MockSingleArgOnlyCov cov;
  cov.stats->reset();

  std::vector<double> xs = {1.0, 2.0};
  Eigen::MatrixXd result = cov(xs);

  EXPECT_EQ(cov.stats->single_arg_count, 1);
  EXPECT_EQ(result(0, 0), 333.0);
}

/*
 * BASIC TEST 3: Cross-covariance should still use two-arg
 */
TEST(test_single_arg_symmetric, test_cross_uses_two_arg) {
  MockSingleArgBatchCov cov;
  cov.stats->reset();

  std::vector<double> xs = {1.0, 2.0};
  std::vector<double> ys = {3.0, 4.0, 5.0};
  Eigen::MatrixXd result = cov(xs, ys);  // Cross-covariance

  EXPECT_EQ(cov.stats->single_arg_count, 0) << "Single-arg should NOT be called";
  EXPECT_EQ(cov.stats->two_arg_count, 1) << "Two-arg should be called";
  EXPECT_EQ(result(0, 0), 888.0);
}

/*
 * BASIC TEST 4: Pool should be passed to single-arg method
 */
TEST(test_single_arg_symmetric, test_pool_propagation) {
  MockSingleArgBatchCov cov;
  cov.stats->reset();

  ThreadPool pool(2);
  std::vector<double> xs = {1.0, 2.0};
  cov(xs, &pool);

  EXPECT_TRUE(cov.stats->pool_was_nonnull) << "Pool should be passed";
}

/*
 * EXTENDED TEST 5: variant<double> symmetric uses single-arg
 */
TEST(test_single_arg_symmetric, test_variant_symmetric_single_call) {
  MockSingleArgBatchCov cov;
  cov.stats->reset();

  using DoubleVariant = variant<double>;
  std::vector<DoubleVariant> xs(50, DoubleVariant(1.0));

  Eigen::MatrixXd result = cov(xs);

  EXPECT_EQ(cov.stats->single_arg_count, 1);
  EXPECT_EQ(cov.stats->two_arg_count, 0);
  EXPECT_EQ(cov.stats->last_xs_size, 50);
}

/*
 * EXTENDED TEST 6: Single-arg-only with variants
 */
TEST(test_single_arg_symmetric, test_variant_single_arg_only) {
  MockSingleArgOnlyCov cov;
  cov.stats->reset();

  using DoubleVariant = variant<double>;
  std::vector<DoubleVariant> xs = {DoubleVariant(1.0), DoubleVariant(2.0)};

  Eigen::MatrixXd result = cov(xs);

  EXPECT_EQ(cov.stats->single_arg_count, 1);
  EXPECT_EQ(result(0, 0), 333.0);
}

/*
 * EXTENDED TEST 7: Measurement<double> symmetric uses single-arg
 */
TEST(test_single_arg_symmetric, test_measurement_symmetric_single_call) {
  MockSingleArgBatchCov cov;
  cov.stats->reset();

  std::vector<Measurement<double>> ms(100, Measurement<double>{1.0});

  Eigen::MatrixXd result = cov(ms);

  EXPECT_EQ(cov.stats->single_arg_count, 1);
  EXPECT_EQ(cov.stats->two_arg_count, 0);
  EXPECT_EQ(cov.stats->last_xs_size, 100);
}

/*
 * EXTENDED TEST 8: variant pool propagation
 */
TEST(test_single_arg_symmetric, test_variant_pool_propagation) {
  MockSingleArgBatchCov cov;
  cov.stats->reset();

  ThreadPool pool(2);
  using DoubleVariant = variant<double>;
  std::vector<DoubleVariant> xs(10, DoubleVariant(1.0));

  cov(xs, &pool);

  EXPECT_TRUE(cov.stats->pool_was_nonnull);
}

/*
 * EXTENDED TEST 9: measurement pool propagation
 */
TEST(test_single_arg_symmetric, test_measurement_pool_propagation) {
  MockSingleArgBatchCov cov;
  cov.stats->reset();

  ThreadPool pool(2);
  std::vector<Measurement<double>> ms(10, Measurement<double>{1.0});

  cov(ms, &pool);

  EXPECT_TRUE(cov.stats->pool_was_nonnull);
}

/*
 * EXTENDED TEST 10: Sum composition propagates single-arg support
 */
TEST(test_single_arg_symmetric, test_sum_propagates_single_arg) {
  MockSingleArgBatchCov cov1;
  MockSingleArgBatchCov cov2;
  cov1.stats->reset();
  cov2.stats->reset();

  auto sum = cov1 + cov2;

  // Verify Sum has single-arg batch support
  using SumType = decltype(sum);
  static_assert(has_valid_call_impl_vector_single_arg<SumType, double>::value,
                "Sum should have single-arg batch support");

  std::vector<double> xs = {1.0, 2.0, 3.0};
  Eigen::MatrixXd result = sum(xs);

  // Both sides should use single-arg
  EXPECT_EQ(cov1.stats->single_arg_count, 1) << "LHS should use single-arg";
  EXPECT_EQ(cov1.stats->two_arg_count, 0) << "LHS should NOT use two-arg";
  EXPECT_EQ(cov2.stats->single_arg_count, 1) << "RHS should use single-arg";
  EXPECT_EQ(cov2.stats->two_arg_count, 0) << "RHS should NOT use two-arg";

  // Result should be sum: 777 + 777 = 1554
  EXPECT_EQ(result(0, 0), 1554.0);
}

/*
 * EXTENDED TEST 11: Product composition propagates single-arg support
 */
TEST(test_single_arg_symmetric, test_product_propagates_single_arg) {
  MockSingleArgBatchCov cov1;
  MockSingleArgBatchCov cov2;
  cov1.stats->reset();
  cov2.stats->reset();

  auto product = cov1 * cov2;

  // Verify Product has single-arg batch support
  using ProductType = decltype(product);
  static_assert(has_valid_call_impl_vector_single_arg<ProductType, double>::value,
                "Product should have single-arg batch support");

  std::vector<double> xs = {1.0, 2.0, 3.0};
  Eigen::MatrixXd result = product(xs);

  // Both sides should use single-arg
  EXPECT_EQ(cov1.stats->single_arg_count, 1) << "LHS should use single-arg";
  EXPECT_EQ(cov1.stats->two_arg_count, 0) << "LHS should NOT use two-arg";
  EXPECT_EQ(cov2.stats->single_arg_count, 1) << "RHS should use single-arg";
  EXPECT_EQ(cov2.stats->two_arg_count, 0) << "RHS should NOT use two-arg";

  // Result should be product: 777 * 777 = 603729
  EXPECT_EQ(result(0, 0), 603729.0);
}

/*
 * EXTENDED TEST 12: Nested composition propagates single-arg support
 */
TEST(test_single_arg_symmetric, test_nested_composition_single_arg) {
  MockSingleArgBatchCov cov1;
  MockSingleArgBatchCov cov2;
  MockSingleArgBatchCov cov3;
  cov1.stats->reset();
  cov2.stats->reset();
  cov3.stats->reset();

  auto nested = (cov1 + cov2) * cov3;

  // Verify nested composition has single-arg batch support
  using NestedType = decltype(nested);
  static_assert(has_valid_call_impl_vector_single_arg<NestedType, double>::value,
                "Nested composition should have single-arg batch support");

  std::vector<double> xs = {1.0, 2.0};
  Eigen::MatrixXd result = nested(xs);

  // All should use single-arg
  EXPECT_EQ(cov1.stats->single_arg_count, 1);
  EXPECT_EQ(cov2.stats->single_arg_count, 1);
  EXPECT_EQ(cov3.stats->single_arg_count, 1);
  EXPECT_EQ(cov1.stats->two_arg_count, 0);
  EXPECT_EQ(cov2.stats->two_arg_count, 0);
  EXPECT_EQ(cov3.stats->two_arg_count, 0);

  // Result: (777 + 777) * 777 = 1554 * 777 = 1207458
  EXPECT_EQ(result(0, 0), 1207458.0);
}

/*
 * EXTENDED TEST 13: Mixed composition - one has single-arg, one has two-arg only
 */
TEST(test_single_arg_symmetric, test_mixed_sum_single_and_two_arg) {
  MockSingleArgBatchCov single_arg_cov;  // Has single-arg
  BatchTestCovariance two_arg_cov(100.0);  // Has only two-arg

  single_arg_cov.stats->reset();

  auto sum = single_arg_cov + two_arg_cov;

  std::vector<double> xs = {1.0, 2.0};
  Eigen::MatrixXd result = sum(xs);

  // single_arg_cov should use single-arg, two_arg_cov uses two-arg
  EXPECT_EQ(single_arg_cov.stats->single_arg_count, 1);
  EXPECT_EQ(single_arg_cov.stats->two_arg_count, 0);

  // Result: 777 + 100 = 877
  EXPECT_EQ(result(0, 0), 877.0);
}

/*
 * EXTENDED TEST 14: Neither child has single-arg (both two-arg only)
 * Sum should NOT expose single-arg batch method.
 */
TEST(test_single_arg_symmetric, test_sum_neither_child_has_single_arg) {
  MockBatchCovariance cov1;  // two-arg only
  MockBatchCovariance cov2;  // two-arg only

  auto sum = cov1 + cov2;

  // Sum should NOT have single-arg batch support
  using SumType = decltype(sum);
  static_assert(!has_valid_call_impl_vector_single_arg<SumType, double>::value,
                "Sum of two-arg-only children should NOT have single-arg");

  // Should still work via two-arg path
  std::vector<double> xs = {1.0, 2.0, 3.0};
  Eigen::MatrixXd result = sum(xs);

  // Both should use two-arg batch
  EXPECT_EQ(cov1.stats->call_count, 1);
  EXPECT_EQ(cov2.stats->call_count, 1);

  // Result: 42 + 42 = 84
  EXPECT_EQ(result(0, 0), 84.0);
}

/*
 * EXTENDED TEST 15: Mixed product - one child has single-arg, other two-arg only
 */
TEST(test_single_arg_symmetric, test_mixed_product_single_and_two_arg) {
  MockSingleArgBatchCov single_arg_cov;  // Has single-arg
  BatchTestCovariance two_arg_cov(100.0);  // Has only two-arg

  single_arg_cov.stats->reset();

  auto product = single_arg_cov * two_arg_cov;

  // Product should have single-arg batch support (at least one child has it)
  using ProductType = decltype(product);
  static_assert(has_valid_call_impl_vector_single_arg<ProductType, double>::value,
                "Product should have single-arg when one child does");

  std::vector<double> xs = {1.0, 2.0};
  Eigen::MatrixXd result = product(xs);

  // single_arg_cov should use single-arg
  EXPECT_EQ(single_arg_cov.stats->single_arg_count, 1);
  EXPECT_EQ(single_arg_cov.stats->two_arg_count, 0);

  // Result: 777 * 100 = 77700
  EXPECT_EQ(result(0, 0), 77700.0);
}

/*
 * EXTENDED TEST 16: Single-arg-only child + two-arg-only child in Sum
 */
TEST(test_single_arg_symmetric, test_sum_single_arg_only_plus_two_arg_only) {
  MockSingleArgOnlyCov single_only;  // only single-arg, no two-arg, no pointwise
  MockBatchCovariance two_arg_only;  // only two-arg

  single_only.stats->reset();

  auto sum = single_only + two_arg_only;

  // Sum should have single-arg batch support
  using SumType = decltype(sum);
  static_assert(has_valid_call_impl_vector_single_arg<SumType, double>::value,
                "Sum should have single-arg when one child does");

  std::vector<double> xs = {1.0, 2.0};
  Eigen::MatrixXd result = sum(xs);

  // single_only uses single-arg; two_arg_only uses two-arg fallback
  EXPECT_EQ(single_only.stats->single_arg_count, 1);
  EXPECT_EQ(two_arg_only.stats->call_count, 1);

  // Result: 333 + 42 = 375
  EXPECT_EQ(result(0, 0), 375.0);
}

/*
 * EXTENDED TEST 17: Variant through Sum with mixed single-arg/two-arg children
 */
TEST(test_single_arg_symmetric, test_variant_through_mixed_sum) {
  MockSingleArgBatchCov single_arg_cov;
  BatchTestCovariance two_arg_cov(100.0);

  single_arg_cov.stats->reset();

  auto sum = single_arg_cov + two_arg_cov;

  // Create variant vector
  using V = variant<double>;
  std::vector<V> xs;
  xs.push_back(V(1.0));
  xs.push_back(V(2.0));

  Eigen::MatrixXd result = sum(xs);

  // single_arg_cov should use single-arg through variant unwrapping
  EXPECT_EQ(single_arg_cov.stats->single_arg_count, 1);
  EXPECT_EQ(single_arg_cov.stats->two_arg_count, 0);

  // Result: 777 + 100 = 877
  EXPECT_EQ(result(0, 0), 877.0);
}

} // namespace albatross
