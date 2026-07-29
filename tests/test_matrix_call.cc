/*
 * Copyright (C) 2026 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include <albatross/GP>
#include <gtest/gtest.h>

#include <random>

namespace albatross {
namespace {

/*
 * A squared exponential clone WITHOUT a _call_matrix_impl which is used
 * as the (unchanged) per-pair reference implementation.
 */
class ReferenceSquaredExponential
    : public CovarianceFunction<ReferenceSquaredExponential> {
public:
  explicit ReferenceSquaredExponential(double length_scale = 1.,
                                       double sigma = 1.)
      : length_scale_(length_scale), sigma_(sigma) {}

  std::string name() const { return "reference_squared_exponential"; }

  template <
      typename X,
      typename std::enable_if<
          has_call_operator<EuclideanDistance, X &, X &>::value, int>::type = 0>
  double _call_impl(const X &x, const X &y) const {
    return squared_exponential_covariance(distance_metric_(x, y), length_scale_,
                                          sigma_);
  }

  EuclideanDistance distance_metric_;
  double length_scale_;
  double sigma_;
};

/*
 * A scalar-only test kernel used to exercise composition with a
 * matrix-capable kernel.
 */
class OnePlusDotProduct : public CovarianceFunction<OnePlusDotProduct> {
public:
  std::string name() const { return "one_plus_dot_product"; }

  double _call_impl(const Eigen::VectorXd &x, const Eigen::VectorXd &y) const {
    return 1. + x.dot(y);
  }

  double _call_impl(const double &x, const double &y) const {
    return 1. + x * y;
  }
};

std::vector<Eigen::VectorXd> random_vector_features(std::size_t n,
                                                    Eigen::Index dimension,
                                                    std::uint32_t seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(-3., 3.);
  std::vector<Eigen::VectorXd> features(n);
  for (auto &f : features) {
    f.resize(dimension);
    for (Eigen::Index i = 0; i < dimension; ++i) {
      f[i] = dist(gen);
    }
  }
  return features;
}

std::vector<double> random_double_features(std::size_t n, std::uint32_t seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(0., 10.);
  std::vector<double> features(n);
  for (auto &f : features) {
    f = dist(gen);
  }
  return features;
}

// The matrix path forms squared distances through a GEMM which can
// differ from the scalar path in the last ulps; comparisons therefore
// use a relative tolerance rather than exact equality.
void expect_relatively_equal(const Eigen::MatrixXd &actual,
                             const Eigen::MatrixXd &expected,
                             double relative_tolerance = 1.e-10) {
  ASSERT_EQ(actual.rows(), expected.rows());
  ASSERT_EQ(actual.cols(), expected.cols());
  for (Eigen::Index i = 0; i < actual.rows(); ++i) {
    for (Eigen::Index j = 0; j < actual.cols(); ++j) {
      const double scale = std::max(1., std::fabs(expected(i, j)));
      EXPECT_NEAR(actual(i, j), expected(i, j), relative_tolerance * scale)
          << "at (" << i << ", " << j << ")";
    }
  }
}

template <typename CovFunc, typename X, typename Y>
Eigen::MatrixXd per_pair_reference(const CovFunc &cov, const std::vector<X> &xs,
                                   const std::vector<Y> &ys) {
  Eigen::MatrixXd expected(cast::to_index(xs.size()),
                           cast::to_index(ys.size()));
  for (std::size_t i = 0; i < xs.size(); ++i) {
    for (std::size_t j = 0; j < ys.size(); ++j) {
      expected(cast::to_index(i), cast::to_index(j)) = cov(xs[i], ys[j]);
    }
  }
  return expected;
}

using SqExp = SquaredExponential<EuclideanDistance>;

TEST(test_matrix_call, test_trait_detection) {
  // The flagship implementation is enabled for Eigen vectors and doubles
  // with the Euclidean distance ...
  static_assert(
      has_valid_matrix_caller<SqExp, Eigen::VectorXd, Eigen::VectorXd>::value,
      "expected matrix caller for Eigen::VectorXd");
  static_assert(
      has_valid_matrix_caller<SqExp, Eigen::Vector3d, Eigen::Vector3d>::value,
      "expected matrix caller for fixed size Eigen vectors");
  static_assert(has_valid_matrix_caller<SqExp, double, double>::value,
                "expected matrix caller for double");
  // ... but not for other distance metrics, other feature types, or
  // kernels which never opted in.
  static_assert(
      !has_valid_matrix_caller<SquaredExponential<RadialDistance>,
                               Eigen::VectorXd, Eigen::VectorXd>::value,
      "matrix caller should require EuclideanDistance");
  static_assert(!has_valid_matrix_caller<SqExp, Measurement<double>,
                                         Measurement<double>>::value,
                "no matrix caller for wrapped types");
  static_assert(
      !has_valid_matrix_caller<ReferenceSquaredExponential, Eigen::VectorXd,
                               Eigen::VectorXd>::value,
      "reference kernel must not have a matrix caller");
  static_assert(
      !has_valid_matrix_caller<Exponential<EuclideanDistance>, Eigen::VectorXd,
                               Eigen::VectorXd>::value,
      "only the squared exponential opted in");

  // Compositions forward the matrix implementation when at least one
  // side has one and both sides stay consistent with the scalar path.
  using SumWithNoise =
      SumOfCovarianceFunctions<SqExp, IndependentNoise<double>>;
  static_assert(has_valid_matrix_caller<SumWithNoise, double, double>::value,
                "sum should forward the matrix caller");
  static_assert(!has_valid_matrix_caller<SumWithNoise, Measurement<double>,
                                         Measurement<double>>::value,
                "no matrix caller through Measurement<> for compositions");
  using ProductWithDot = ProductOfCovarianceFunctions<SqExp, OnePlusDotProduct>;
  static_assert(has_valid_matrix_caller<ProductWithDot, Eigen::VectorXd,
                                        Eigen::VectorXd>::value,
                "product should forward the matrix caller");
  using SumOfReferences =
      SumOfCovarianceFunctions<ReferenceSquaredExponential, OnePlusDotProduct>;
  static_assert(!has_valid_matrix_caller<SumOfReferences, Eigen::VectorXd,
                                         Eigen::VectorXd>::value,
                "compositions of per-pair kernels stay per-pair");
}

TEST(test_matrix_call, test_matrix_matches_per_pair_cross) {
  const SqExp cov(2., 1.5);
  const auto xs = random_vector_features(25, 3, 0);
  const auto ys = random_vector_features(17, 3, 1);

  expect_relatively_equal(cov(xs, ys), per_pair_reference(cov, xs, ys));
}

TEST(test_matrix_call, test_matrix_matches_per_pair_symmetric) {
  const SqExp cov(2., 1.5);
  auto xs = random_vector_features(25, 3, 2);
  // include an exact duplicate to exercise the diagonal / clamping
  xs.push_back(xs[3]);

  const Eigen::MatrixXd actual = cov(xs);
  expect_relatively_equal(actual, per_pair_reference(cov, xs, xs));

  // the vectorized path must stay exactly symmetric like the per-pair one
  for (Eigen::Index i = 0; i < actual.rows(); ++i) {
    for (Eigen::Index j = 0; j < actual.cols(); ++j) {
      EXPECT_EQ(actual(i, j), actual(j, i));
    }
  }
}

TEST(test_matrix_call, test_matrix_matches_per_pair_double) {
  const SqExp cov(2., 1.5);
  const auto xs = random_double_features(31, 3);
  const auto ys = random_double_features(19, 4);

  expect_relatively_equal(cov(xs, ys), per_pair_reference(cov, xs, ys));
  expect_relatively_equal(cov(xs), per_pair_reference(cov, xs, xs));
}

TEST(test_matrix_call, test_thread_pool_argument_is_accepted) {
  // When the matrix path is taken the pool is (deliberately) ignored;
  // passing one must give the same result.
  const SqExp cov(2., 1.5);
  const auto xs = random_vector_features(11, 3, 5);
  auto pool = std::make_shared<ThreadPool>(2);
  const Eigen::MatrixXd with_pool = cov(xs, pool.get());
  const Eigen::MatrixXd without_pool = cov(xs);
  EXPECT_EQ(with_pool, without_pool);
}

TEST(test_matrix_call, test_measurement_unwrapping) {
  const SqExp cov(2., 1.5);
  const auto xs = random_double_features(21, 6);
  const auto measurements = as_measurements(xs);

  // A plain (leaf) kernel with no Measurement<> specific behavior takes
  // the matrix path for measurements by stripping the wrapper.
  expect_relatively_equal(cov(measurements),
                          per_pair_reference(cov, measurements, measurements));
  EXPECT_EQ(cov(measurements), cov(xs));
}

TEST(test_matrix_call, test_sum_composition) {
  const SqExp sq_exp(2., 1.5);
  const IndependentNoise<double> noise(0.25);
  const auto sum = sq_exp + noise;

  auto xs = random_double_features(23, 7);
  // duplicate a feature so the noise term contributes off-diagonal
  xs.push_back(xs[5]);
  const auto ys = random_double_features(13, 8);

  expect_relatively_equal(sum(xs, ys), per_pair_reference(sum, xs, ys));
  expect_relatively_equal(sum(xs), per_pair_reference(sum, xs, xs));

  // Measurement<> features (the fit path) fall back to the unchanged
  // per-pair evaluation for compositions; results must be identical.
  const auto measurements = as_measurements(xs);
  const Eigen::MatrixXd actual = sum(measurements);
  const Eigen::MatrixXd expected =
      per_pair_reference(sum, measurements, measurements);
  EXPECT_EQ(actual, expected);
}

TEST(test_matrix_call, test_product_composition) {
  const SqExp sq_exp(2., 1.5);
  const OnePlusDotProduct dot;
  const auto product = sq_exp * dot;

  const auto xs = random_vector_features(21, 3, 9);
  const auto ys = random_vector_features(14, 3, 10);

  expect_relatively_equal(product(xs, ys), per_pair_reference(product, xs, ys));
  expect_relatively_equal(product(xs), per_pair_reference(product, xs, xs));
}

TEST(test_matrix_call, test_nested_composition) {
  // k1 * k2 + k3 with only k1 matrix-capable benefits term-wise.
  const SqExp sq_exp(2., 1.5);
  const OnePlusDotProduct dot;
  const IndependentNoise<double> noise(0.25);
  const auto composed = sq_exp * dot + noise;

  static_assert(
      has_valid_matrix_caller<decltype(composed), double, double>::value,
      "nested composition should forward the matrix caller");

  auto xs = random_double_features(23, 11);
  xs.push_back(xs[0]);
  const auto ys = random_double_features(17, 12);

  expect_relatively_equal(composed(xs, ys),
                          per_pair_reference(composed, xs, ys));
  expect_relatively_equal(composed(xs), per_pair_reference(composed, xs, xs));
}

TEST(test_matrix_call, test_zero_length_scale_guard) {
  const SqExp cov(0., 1.5);
  const auto xs = random_double_features(7, 13);
  const auto vs = random_vector_features(7, 3, 13);

  EXPECT_TRUE(cov(xs).isZero(0.));
  EXPECT_TRUE(cov(vs).isZero(0.));
}

TEST(test_matrix_call, test_empty_features) {
  const SqExp cov(2., 1.5);
  const std::vector<Eigen::VectorXd> empty;
  const auto xs = random_vector_features(5, 3, 14);

  EXPECT_EQ(cov(empty).rows(), 0);
  EXPECT_EQ(cov(empty, xs).rows(), 0);
  EXPECT_EQ(cov(empty, xs).cols(), 5);
  EXPECT_EQ(cov(xs, empty).rows(), 5);
  EXPECT_EQ(cov(xs, empty).cols(), 0);
}

template <typename CovFunc>
RegressionDataset<double> toy_dataset(const CovFunc &, std::uint32_t seed) {
  const auto features = random_double_features(60, seed);
  Eigen::VectorXd targets(cast::to_index(features.size()));
  for (std::size_t i = 0; i < features.size(); ++i) {
    targets[cast::to_index(i)] =
        std::sin(features[i]) + 0.1 * std::cos(10. * features[i]);
  }
  const Eigen::VectorXd variance =
      Eigen::VectorXd::Constant(targets.size(), 0.01);
  return RegressionDataset<double>(features,
                                   MarginalDistribution(targets, variance));
}

TEST(test_matrix_call, test_gp_end_to_end) {
  // The same GP with a matrix-capable kernel and with the per-pair
  // reference kernel must produce (numerically) identical predictions.
  const SqExp matrix_kernel(2., 1.5);
  const ReferenceSquaredExponential reference_kernel(2., 1.5);

  auto matrix_model = gp_from_covariance(matrix_kernel, "matrix_gp");
  auto reference_model = gp_from_covariance(reference_kernel, "reference_gp");

  const auto dataset = toy_dataset(matrix_kernel, 15);
  const auto test_features = random_double_features(29, 16);

  const auto matrix_fit = matrix_model.fit(dataset);
  const auto reference_fit = reference_model.fit(dataset);

  const auto matrix_pred = matrix_fit.predict(test_features);
  const auto reference_pred = reference_fit.predict(test_features);

  const Eigen::VectorXd mean_diff = matrix_pred.mean() - reference_pred.mean();
  EXPECT_LT(mean_diff.array().abs().maxCoeff(), 1.e-8);

  const auto matrix_marginal = matrix_pred.marginal();
  const auto reference_marginal = reference_pred.marginal();
  EXPECT_LT(
      (matrix_marginal.mean - reference_marginal.mean).array().abs().maxCoeff(),
      1.e-8);
  EXPECT_LT((matrix_marginal.covariance.diagonal() -
             reference_marginal.covariance.diagonal())
                .array()
                .abs()
                .maxCoeff(),
            1.e-8);

  const auto matrix_joint = matrix_pred.joint();
  const auto reference_joint = reference_pred.joint();
  EXPECT_LT((matrix_joint.mean - reference_joint.mean).array().abs().maxCoeff(),
            1.e-8);
  EXPECT_LT((matrix_joint.covariance - reference_joint.covariance)
                .array()
                .abs()
                .maxCoeff(),
            1.e-8);
}

TEST(test_matrix_call, test_per_pair_fallback_unchanged) {
  // A kernel without _call_matrix_impl must keep going through the
  // per-pair path and produce bit-for-bit identical results to a direct
  // scalar loop.
  const ReferenceSquaredExponential cov(2., 1.5);
  const auto xs = random_vector_features(15, 3, 17);
  const auto ys = random_vector_features(9, 3, 18);

  EXPECT_EQ(cov(xs, ys), per_pair_reference(cov, xs, ys));
  EXPECT_EQ(cov(xs), per_pair_reference(cov, xs, xs));
}

} // namespace
} // namespace albatross
