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

#include <albatross/Core>
#include <albatross/CovarianceFunctions>
#include <albatross/GP>
#include <gtest/gtest.h>

namespace albatross {

TEST(MixedPrecision, TemplatedCovarianceFunctions) {
  // Test that templated covariance functions work with float
  float distance_f = 1.0f;
  float length_scale_f = 10.0f;
  float sigma_f = 2.0f;

  // Call templated versions with float
  float result_se_f = squared_exponential_covariance(distance_f, length_scale_f, sigma_f);
  float result_exp_f = exponential_covariance(distance_f, length_scale_f, sigma_f);
  float result_m32_f = matern_32_covariance(distance_f, length_scale_f, sigma_f);
  float result_m52_f = matern_52_covariance(distance_f, length_scale_f, sigma_f);

  // Call with double for comparison
  double distance_d = 1.0;
  double length_scale_d = 10.0;
  double sigma_d = 2.0;

  double result_se_d = squared_exponential_covariance(distance_d, length_scale_d, sigma_d);
  double result_exp_d = exponential_covariance(distance_d, length_scale_d, sigma_d);
  double result_m32_d = matern_32_covariance(distance_d, length_scale_d, sigma_d);
  double result_m52_d = matern_52_covariance(distance_d, length_scale_d, sigma_d);

  // Results should be similar (within float precision)
  EXPECT_NEAR(result_se_f, result_se_d, 1e-6);
  EXPECT_NEAR(result_exp_f, result_exp_d, 1e-6);
  EXPECT_NEAR(result_m32_f, result_m32_d, 1e-6);
  EXPECT_NEAR(result_m52_f, result_m52_d, 1e-6);

  // Verify results are positive and reasonable
  EXPECT_GT(result_se_f, 0.0f);
  EXPECT_GT(result_exp_f, 0.0f);
  EXPECT_GT(result_m32_f, 0.0f);
  EXPECT_GT(result_m52_f, 0.0f);
}

TEST(MixedPrecision, PrecisionConversion) {
  // Test precision conversion utilities
  Eigen::VectorXd vec_d = Eigen::VectorXd::LinSpaced(10, 0.0, 1.0);

  // Convert to float
  Eigen::VectorXf vec_f = convert_precision<float>(vec_d);

  // Convert back to double
  Eigen::VectorXd vec_d2 = convert_precision<double>(vec_f);

  // Should be very close
  EXPECT_TRUE(vec_d.isApprox(vec_d2, 1e-6));

  // Test with identity (double to double)
  Eigen::VectorXd vec_d3 = convert_precision<double>(vec_d);
  EXPECT_TRUE(vec_d == vec_d3);
}

TEST(MixedPrecision, DistributionConversion) {
  // Create a MarginalDistribution with double
  Eigen::VectorXd mean_d = Eigen::VectorXd::LinSpaced(5, 1.0, 5.0);
  Eigen::VectorXd var_d = Eigen::VectorXd::Constant(5, 0.5);
  MarginalDistribution dist_d(mean_d, var_d);

  // Convert to float for computation
  auto dist_f = to_float(dist_d);

  // Verify float precision
  EXPECT_EQ(dist_f.mean.rows(), 5);
  EXPECT_EQ(dist_f.variance.rows(), 5);

  // Convert back to double
  MarginalDistribution dist_d2 = dist_f.to_double();

  // Should be very close
  EXPECT_TRUE(dist_d.mean.isApprox(dist_d2.mean, 1e-6));
  EXPECT_TRUE(dist_d.covariance.diagonal().isApprox(dist_d2.covariance.diagonal(), 1e-6));
}

TEST(MixedPrecision, MatrixMultiplyMixed) {
  // Test mixed-precision matrix multiplication
  Eigen::MatrixXd A = Eigen::MatrixXd::Random(50, 50);
  Eigen::MatrixXd B = Eigen::MatrixXd::Random(50, 50);

  // Standard double precision
  Eigen::MatrixXd C_double = A * B;

  // Mixed precision (float computation, double storage)
  Eigen::MatrixXd C_mixed = matrix_multiply_mixed(A, B);

  // Results should be very close (within float precision)
  EXPECT_TRUE(C_double.isApprox(C_mixed, 1e-5));

  // Verify dimensions
  EXPECT_EQ(C_mixed.rows(), 50);
  EXPECT_EQ(C_mixed.cols(), 50);
}

TEST(MixedPrecision, MatrixVectorMultiplyMixed) {
  // Test mixed-precision matrix-vector multiplication
  Eigen::MatrixXd A = Eigen::MatrixXd::Random(50, 50);
  Eigen::VectorXd v = Eigen::VectorXd::Random(50);

  // Standard double precision
  Eigen::VectorXd result_double = A * v;

  // Mixed precision
  Eigen::VectorXd result_mixed = matrix_vector_multiply_mixed(A, v);

  // Results should be very close
  EXPECT_TRUE(result_double.isApprox(result_mixed, 1e-5));

  // Verify dimension
  EXPECT_EQ(result_mixed.rows(), 50);
}

} // namespace albatross
