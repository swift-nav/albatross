/*
 * Copyright (C) 2018 Swift Navigation Inc.
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
#include <array>
#include <gtest/gtest.h>

#include "test_utils.h"

namespace albatross {

inline auto random_spherical_dataset(std::vector<Eigen::VectorXd> points,
                                     std::size_t seed = 7) {
  std::random_device rd{};
  std::mt19937 gen{rd()};
  gen.seed(static_cast<std::mt19937::result_type>(seed));
  std::normal_distribution<> d{0., 0.1};

  Eigen::VectorXd targets(cast::to_index(points.size()));

  auto spherical_function = [](Eigen::VectorXd &x) {
    return x[0] * x[1] + x[1] * x[2] + x[3];
  };

  for (std::size_t i = 0; i < points.size(); i++) {
    targets[cast::to_index(i)] = spherical_function(points[i]);
  }

  return RegressionDataset<Eigen::VectorXd>(points, targets);
}

template <typename T> class RadialCovarianceTester : public ::testing::Test {
public:
  T test_case;
};

using RadialTestCases =
    ::testing::Types<Exponential<EuclideanDistance>,
                     SquaredExponential<EuclideanDistance>,
                     Matern32<EuclideanDistance>, Matern52<EuclideanDistance>>;
TYPED_TEST_SUITE(RadialCovarianceTester, RadialTestCases);

TYPED_TEST(RadialCovarianceTester, test_edge_cases) {
  double sigma = NAN;
  for (const auto &pair : this->test_case.get_params()) {
    if (pair.first.find("sigma") != std::string::npos) {
      sigma = pair.second.value;
    }
  }
  // the same points should be fully correlated
  EXPECT_EQ(this->test_case(M_PI, M_PI), sigma * sigma);
  // extremely close points should be almost perfectly correlated
  EXPECT_NEAR(this->test_case(M_PI, M_PI + 1e-16), this->test_case(M_PI, M_PI),
              1e-8);
  // extremely distant points should be independent
  EXPECT_EQ(this->test_case(0., 1e32), 0.);
}

TYPED_TEST(RadialCovarianceTester, test_derive_length_scale) {

  auto set_sigma_length_scale = [this](double sigma, double length_scale) {
    for (const auto &pair : this->test_case.get_params()) {
      if (pair.first.find("length_scale") != std::string::npos) {
        albatross::set_param_value(pair.first, length_scale, &this->test_case);
      } else if (pair.first.find("sigma") != std::string::npos) {
        albatross::set_param_value(pair.first, sigma, &this->test_case);
      } else {
        assert(false && "unexpected radial parameter");
      }
    }
  };

  auto std_increase = [&](double dist, double sigma, double length_scale) {
    set_sigma_length_scale(sigma, length_scale);
    auto eval = [this](double x) { return this->test_case(0., x); };
    return process_noise_equivalent(eval, dist);
  };

  auto test_equivalence = [&](double dist, double sigma,
                              double desired_sd_increase) {
    double expected_sd_increase = desired_sd_increase;

    // We can only expect out solver to get the length scale within
    // the values achievable with the min/max length scale ratios
    const double min_length_scale = dist * MIN_LENGTH_SCALE_RATIO;
    const double max_increase = std_increase(dist, sigma, min_length_scale);
    if (max_increase <= expected_sd_increase) {
      expected_sd_increase = max_increase;
    }
    const double max_length_scale = dist * MAX_LENGTH_SCALE_RATIO;
    const double min_increase = std_increase(dist, sigma, max_length_scale);
    if (min_increase >= expected_sd_increase) {
      expected_sd_increase = min_increase;
    }

    const double length_scale =
        this->test_case.derive_length_scale(dist, sigma, expected_sd_increase);
    const double actual_sd_increase = std_increase(dist, sigma, length_scale);

    const double relative_sd_error =
        fabs(expected_sd_increase - actual_sd_increase) / sigma;
    const double absolute_sd_error =
        fabs(expected_sd_increase - actual_sd_increase);
    const bool success =
        (absolute_sd_error <= 1e-8 || relative_sd_error <= 1e-8);
    EXPECT_TRUE(success);
    if (!success) {
      std::cerr << "TEST WITH dist : " << dist << "  sigma: " << sigma
                << "  increase: " << desired_sd_increase
                << " expected: " << expected_sd_increase << std::endl;
      std::cerr << "FAIL: absolute_sd error: " << absolute_sd_error
                << " relative_sd_error: " << relative_sd_error << std::endl;
    }
    return relative_sd_error <= 1e-3;
  };

  // large sigmas are likely lead to numerical instabilities
  // in matrix inversions, so that seems like a reasonable limit;
  const std::vector<double> sigmas = {1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6};
  const std::vector<double> distances = {1e-8, 1e-4,  1e-2, 1.,  10.,
                                         100., 1000., 1e8,  1e12};
  const std::vector<double> proportion_increases = {
      0, 1e-8, 1e-4, 1e-2, 0.1, 0.5, 0.9, 0.99, 0.9999, 0.99999999, 1.};
  for (const auto &dist : distances) {
    for (const auto &inc : proportion_increases) {
      for (const auto &sigma : sigmas) {
        const double increase = sigma * inc;
        test_equivalence(dist, sigma, increase);
      }
    }
  }
}

TEST(test_radial, test_is_positive_definite) {
  const auto points = random_spherical_points(100);

  const Exponential<AngularDistance> term(2 * M_PI);

  const Eigen::MatrixXd cov = term(points);

  EXPECT_GE(cov.eigenvalues().real().array().minCoeff(), 0.);
}

class SquaredExponentialSSRTest {
public:
  std::vector<double> features() const { return linspace(0., 10., 101); }

  auto covariance_function() const {
    SquaredExponential<EuclideanDistance> cov(5., 1.);
    return cov;
  }

  double get_tolerance() const { return 1e-12; }
};

class ExponentialSSRTest {
public:
  std::vector<double> features() const { return linspace(0., 10., 11); }

  auto covariance_function() const {
    Exponential<EuclideanDistance> cov(5., 1.);
    return cov;
  }

  double get_tolerance() const { return 1e-2; }
};

class ExponentialAngularSSRTest {
public:
  std::vector<double> features() const { return linspace(0., M_2_PI, 11); }

  auto covariance_function() const {
    Exponential<EuclideanDistance> cov(M_PI_4, 1.);
    return cov;
  }

  double get_tolerance() const { return 1e-2; }
};

template <typename T>
class CovarianceStateSpaceTester : public ::testing::Test {
public:
  T test_case;
};

using StateSpaceTestCases =
    ::testing::Types<SquaredExponentialSSRTest, ExponentialSSRTest,
                     ExponentialAngularSSRTest>;
TYPED_TEST_SUITE(CovarianceStateSpaceTester, StateSpaceTestCases);

TYPED_TEST(CovarianceStateSpaceTester, test_state_space_representation) {

  const auto xs = this->test_case.features();

  const auto cov_func = this->test_case.covariance_function();

  expect_state_space_representation_quality(cov_func, xs,
                                            this->test_case.get_tolerance());
}

// These tests just compare our implementation to gpytorch.

constexpr std::size_t kMaternNumOraclePoints = 15;
constexpr double kMaternOracleLengthScale = 22.2;
constexpr double kMaternOracleSigma = 1.;
constexpr const std::array<double, kMaternNumOraclePoints> kOracleMaternX = {
    {-100, -10, -5, -2, -1, -0.01, -1e-05, 0, 1e-05, 0.01, 1, 2, 5, 10, 100}};

// See `python/gpytorch_covariance.py`
constexpr const std::array<std::array<double, kMaternNumOraclePoints>,
                           kMaternNumOraclePoints>
    kOracleMatern52Y{{{{1.0000000000000000e+00, 4.3310891754576569e-03,
                        2.8712281960142816e-03, 2.2391949268465348e-03,
                        2.0604610887790275e-03, 1.8972884311335531e-03,
                        1.8957080991240727e-03, 1.8957065178556913e-03,
                        1.8957049365885999e-03, 1.8941258953182204e-03,
                        1.7438658821262445e-03, 1.6039522834044202e-03,
                        1.2469681971661868e-03, 8.1743049019786381e-04,
                        2.7894734897077351e-07}},
                      {{4.3310891754576569e-03, 1.0000000000000000e+00,
                        9.5978981277189523e-01, 9.0339724550494072e-01,
                        8.8074344391004133e-01, 8.5685492180277034e-01,
                        8.5660731549496694e-01, 8.5660706757945804e-01,
                        8.5660681966382668e-01, 8.5635909074574545e-01,
                        8.3124352297714188e-01, 8.0489556223497594e-01,
                        7.2214387713800432e-01, 5.8253953200498254e-01,
                        8.1743049019786381e-04}},
                      {{2.8712281960142816e-03, 9.5978981277189523e-01,
                        1.0000000000000000e+00, 9.8507841480100966e-01,
                        9.7383504933905141e-01, 9.5994333468030346e-01,
                        9.5978996642145098e-01, 9.5978981277189523e-01,
                        9.5978965912208392e-01, 9.5963603539971132e-01,
                        9.4319121548875084e-01, 9.2430339531954464e-01,
                        8.5660706757945804e-01, 7.2214387713800432e-01,
                        1.2469681971661868e-03}},
                      {{2.2391949268465348e-03, 9.0339724550494072e-01,
                        9.8507841480100966e-01, 1.0000000000000000e+00,
                        9.9831318524994261e-01, 9.9336444225938858e-01,
                        9.9329823524791738e-01, 9.9329816881413147e-01,
                        9.9329810238002458e-01, 9.9323157441883914e-01,
                        9.8507841480100966e-01, 9.7383504933905141e-01,
                        9.2430339531954464e-01, 8.0489556223497594e-01,
                        1.6039522834044202e-03}},
                      {{2.0604610887790275e-03, 8.8074344391004133e-01,
                        9.7383504933905141e-01, 9.9831318524994261e-01,
                        1.0000000000000000e+00, 9.9834667565258917e-01,
                        9.9831321890692826e-01, 9.9831318524994261e-01,
                        9.9831315159262357e-01, 9.9827936137798967e-01,
                        9.9329816881413147e-01, 9.8507841480100966e-01,
                        9.4319121548875084e-01, 8.3124352297714188e-01,
                        1.7438658821262445e-03}},
                      {{1.8972884311335531e-03, 8.5685492180277034e-01,
                        9.5994333468030346e-01, 9.9336444225938858e-01,
                        9.9834667565258917e-01, 1.0000000000000000e+00,
                        9.9999983125004255e-01, 9.9999983091203604e-01,
                        9.9999983057369102e-01, 9.9999932364865785e-01,
                        9.9827936137798967e-01, 9.9323157441883914e-01,
                        9.5963603539971132e-01, 8.5635909074574545e-01,
                        1.8941258953182204e-03}},
                      {{1.8957080991240727e-03, 8.5660731549496694e-01,
                        9.5978996642145098e-01, 9.9329823524791738e-01,
                        9.9831321890692826e-01, 9.9999983125004255e-01,
                        1.0000000000000000e+00, 9.9999999999983102e-01,
                        9.9999999999932365e-01, 9.9999983057369102e-01,
                        9.9831315159262357e-01, 9.9329810238002447e-01,
                        9.5978965912208392e-01, 8.5660681966382668e-01,
                        1.8957049365885999e-03}},
                      {{1.8957065178556913e-03, 8.5660706757945804e-01,
                        9.5978981277189523e-01, 9.9329816881413147e-01,
                        9.9831318524994261e-01, 9.9999983091203604e-01,
                        9.9999999999983102e-01, 1.0000000000000000e+00,
                        9.9999999999983102e-01, 9.9999983091203604e-01,
                        9.9831318524994261e-01, 9.9329816881413147e-01,
                        9.5978981277189523e-01, 8.5660706757945804e-01,
                        1.8957065178556913e-03}},
                      {{1.8957049365885999e-03, 8.5660681966382668e-01,
                        9.5978965912208392e-01, 9.9329810238002458e-01,
                        9.9831315159262357e-01, 9.9999983057369102e-01,
                        9.9999999999932365e-01, 9.9999999999983102e-01,
                        1.0000000000000000e+00, 9.9999983125004255e-01,
                        9.9831321890692826e-01, 9.9329823524791727e-01,
                        9.5978996642145098e-01, 8.5660731549496694e-01,
                        1.8957080991240727e-03}},
                      {{1.8941258953182204e-03, 8.5635909074574545e-01,
                        9.5963603539971121e-01, 9.9323157441883914e-01,
                        9.9827936137798967e-01, 9.9999932364865785e-01,
                        9.9999983057369102e-01, 9.9999983091203604e-01,
                        9.9999983125004255e-01, 1.0000000000000000e+00,
                        9.9834667565258917e-01, 9.9336444225938858e-01,
                        9.5994333468030346e-01, 8.5685492180277034e-01,
                        1.8972884311335531e-03}},
                      {{1.7438658821262445e-03, 8.3124352297714188e-01,
                        9.4319121548875084e-01, 9.8507841480100966e-01,
                        9.9329816881413147e-01, 9.9827936137798967e-01,
                        9.9831315159262357e-01, 9.9831318524994261e-01,
                        9.9831321890692826e-01, 9.9834667565258917e-01,
                        1.0000000000000000e+00, 9.9831318524994250e-01,
                        9.7383504933905141e-01, 8.8074344391004133e-01,
                        2.0604610887790275e-03}},
                      {{1.6039522834044202e-03, 8.0489556223497594e-01,
                        9.2430339531954464e-01, 9.7383504933905141e-01,
                        9.8507841480100966e-01, 9.9323157441883914e-01,
                        9.9329810238002447e-01, 9.9329816881413147e-01,
                        9.9329823524791727e-01, 9.9336444225938858e-01,
                        9.9831318524994250e-01, 1.0000000000000000e+00,
                        9.8507841480100966e-01, 9.0339724550494072e-01,
                        2.2391949268465348e-03}},
                      {{1.2469681971661868e-03, 7.2214387713800432e-01,
                        8.5660706757945804e-01, 9.2430339531954464e-01,
                        9.4319121548875107e-01, 9.5963603539971132e-01,
                        9.5978965912208392e-01, 9.5978981277189523e-01,
                        9.5978996642145098e-01, 9.5994333468030346e-01,
                        9.7383504933905141e-01, 9.8507841480100966e-01,
                        1.0000000000000000e+00, 9.5978981277189523e-01,
                        2.8712281960142816e-03}},
                      {{8.1743049019786381e-04, 5.8253953200498254e-01,
                        7.2214387713800432e-01, 8.0489556223497616e-01,
                        8.3124352297714188e-01, 8.5635909074574545e-01,
                        8.5660681966382668e-01, 8.5660706757945804e-01,
                        8.5660731549496694e-01, 8.5685492180277034e-01,
                        8.8074344391004133e-01, 9.0339724550494072e-01,
                        9.5978981277189523e-01, 1.0000000000000000e+00,
                        4.3310891754576569e-03}},
                      {{2.7894734897077351e-07, 8.1743049019786381e-04,
                        1.2469681971661868e-03, 1.6039522834044202e-03,
                        1.7438658821262445e-03, 1.8941258953182204e-03,
                        1.8957049365885999e-03, 1.8957065178556913e-03,
                        1.8957080991240727e-03, 1.8972884311335531e-03,
                        2.0604610887790275e-03, 2.2391949268465348e-03,
                        2.8712281960142816e-03, 4.3310891754576569e-03,
                        1.0000000000000000e+00}}}};

TEST(test_radial, test_matern_52_oracle) {
  const Matern52<EuclideanDistance> cov(kMaternOracleLengthScale,
                                        kMaternOracleSigma);
  const auto pointwise = [&cov](std::size_t a, std::size_t b) {
    return cov(kOracleMaternX[a], kOracleMaternX[b]);
  };
  for (std::size_t i = 0; i < kMaternNumOraclePoints; ++i) {
    for (std::size_t j = 0; j < kMaternNumOraclePoints; ++j) {
      EXPECT_LT(fabs(pointwise(i, j) - kOracleMatern52Y[i][j]), 1e-15);
    }
  }
}

constexpr const std::array<std::array<double, kMaternNumOraclePoints>,
                           kMaternNumOraclePoints>
    kOracleMatern32Y{{{{1.0000000000000000e+00, 7.1570218859426105e-03,
                        5.0808419223605308e-03, 4.1324103770802364e-03,
                        3.8567465159746687e-03, 3.6016736010994355e-03,
                        3.5991861772829985e-03, 3.5991836882159939e-03,
                        3.5991811991506798e-03, 3.5966954684030486e-03,
                        3.3585576286345501e-03, 3.1337768724546203e-03,
                        2.5445937316078615e-03, 1.7957670897164237e-03,
                        2.7762373851026449e-06}},
                      {{7.1570218859426105e-03, 1.0000000000000000e+00,
                        9.4108224335132984e-01, 8.7007994243695330e-01,
                        8.4343591415494479e-01, 8.1616904235914178e-01,
                        8.1589036925202529e-01, 8.1589009026926218e-01,
                        8.1588981128643745e-01, 8.1561107685978818e-01,
                        7.8772792813480297e-01, 7.5919711542614232e-01,
                        6.7338565590950550e-01, 5.3781521661694287e-01,
                        1.7957670897164237e-03}},
                      {{5.0808419223605308e-03, 9.4108224335132973e-01,
                        1.0000000000000000e+00, 9.7652919861687926e-01,
                        9.6034121298915520e-01, 9.4128816463105169e-01,
                        9.4108244939823804e-01, 9.4108224335132984e-01,
                        9.4108203730417017e-01, 9.4087607073601343e-01,
                        9.1930446061348481e-01, 8.9549444935455369e-01,
                        8.1589009026926218e-01, 6.7338565590950550e-01,
                        2.5445937316078615e-03}},
                      {{4.1324103770802364e-03, 8.7007994243695330e-01,
                        9.7652919861687926e-01, 1.0000000000000000e+00,
                        9.9711018656985062e-01, 9.8912493747379715e-01,
                        9.8902110708352886e-01, 9.8902100292932393e-01,
                        9.8902089877467969e-01, 9.8891662887425402e-01,
                        9.7652919861687926e-01, 9.6034121298915520e-01,
                        8.9549444935455369e-01, 7.5919711542614232e-01,
                        3.1337768724546203e-03}},
                      {{3.8567465159746687e-03, 8.4343591415494479e-01,
                        9.6034121298915520e-01, 9.9711018656985062e-01,
                        1.0000000000000000e+00, 9.9716622987554637e-01,
                        9.9711024287258865e-01, 9.9711018656985062e-01,
                        9.9711013026659367e-01, 9.9705362416186716e-01,
                        9.8902100292932393e-01, 9.7652919861687926e-01,
                        9.1930446061348481e-01, 7.8772792813480286e-01,
                        3.3585576286345501e-03}},
                      {{3.6016736010994355e-03, 8.1616904235914178e-01,
                        9.4128816463105169e-01, 9.8912493747379715e-01,
                        9.9716622987554637e-01, 1.0000000000000000e+00,
                        9.9999969640778685e-01, 9.9999969579984893e-01,
                        9.9999969519130272e-01, 9.9999878383206997e-01,
                        9.9705362416186716e-01, 9.8891662887425402e-01,
                        9.4087607073601343e-01, 8.1561107685978818e-01,
                        3.5966954684030486e-03}},
                      {{3.5991861772829985e-03, 8.1589036925202529e-01,
                        9.4108244939823804e-01, 9.8902110708352886e-01,
                        9.9711024287258865e-01, 9.9999969640778685e-01,
                        1.0000000000000000e+00, 9.9999999999969547e-01,
                        9.9999999999878264e-01, 9.9999969519130272e-01,
                        9.9711013026659367e-01, 9.8902089877467969e-01,
                        9.4108203730417039e-01, 8.1588981128643745e-01,
                        3.5991811991506798e-03}},
                      {{3.5991836882159939e-03, 8.1589009026926218e-01,
                        9.4108224335132984e-01, 9.8902100292932393e-01,
                        9.9711018656985062e-01, 9.9999969579984893e-01,
                        9.9999999999969547e-01, 1.0000000000000000e+00,
                        9.9999999999969547e-01, 9.9999969579984893e-01,
                        9.9711018656985062e-01, 9.8902100292932393e-01,
                        9.4108224335132984e-01, 8.1589009026926218e-01,
                        3.5991836882159939e-03}},
                      {{3.5991811991506798e-03, 8.1588981128643745e-01,
                        9.4108203730417017e-01, 9.8902089877467969e-01,
                        9.9711013026659367e-01, 9.9999969519130272e-01,
                        9.9999999999878264e-01, 9.9999999999969547e-01,
                        1.0000000000000000e+00, 9.9999969640778685e-01,
                        9.9711024287258865e-01, 9.8902110708352886e-01,
                        9.4108244939823804e-01, 8.1589036925202529e-01,
                        3.5991861772829985e-03}},
                      {{3.5966954684030486e-03, 8.1561107685978818e-01,
                        9.4087607073601331e-01, 9.8891662887425402e-01,
                        9.9705362416186716e-01, 9.9999878383206997e-01,
                        9.9999969519130272e-01, 9.9999969579984893e-01,
                        9.9999969640778685e-01, 1.0000000000000000e+00,
                        9.9716622987554637e-01, 9.8912493747379715e-01,
                        9.4128816463105169e-01, 8.1616904235914178e-01,
                        3.6016736010994355e-03}},
                      {{3.3585576286345501e-03, 7.8772792813480286e-01,
                        9.1930446061348481e-01, 9.7652919861687926e-01,
                        9.8902100292932393e-01, 9.9705362416186716e-01,
                        9.9711013026659367e-01, 9.9711018656985062e-01,
                        9.9711024287258865e-01, 9.9716622987554637e-01,
                        1.0000000000000000e+00, 9.9711018656985062e-01,
                        9.6034121298915520e-01, 8.4343591415494479e-01,
                        3.8567465159746687e-03}},
                      {{3.1337768724546203e-03, 7.5919711542614232e-01,
                        8.9549444935455369e-01, 9.6034121298915520e-01,
                        9.7652919861687926e-01, 9.8891662887425402e-01,
                        9.8902089877467969e-01, 9.8902100292932393e-01,
                        9.8902110708352886e-01, 9.8912493747379715e-01,
                        9.9711018656985062e-01, 1.0000000000000000e+00,
                        9.7652919861687926e-01, 8.7007994243695330e-01,
                        4.1324103770802364e-03}},
                      {{2.5445937316078615e-03, 6.7338565590950550e-01,
                        8.1589009026926218e-01, 8.9549444935455369e-01,
                        9.1930446061348492e-01, 9.4087607073601343e-01,
                        9.4108203730417039e-01, 9.4108224335132984e-01,
                        9.4108244939823804e-01, 9.4128816463105169e-01,
                        9.6034121298915520e-01, 9.7652919861687926e-01,
                        1.0000000000000000e+00, 9.4108224335132973e-01,
                        5.0808419223605308e-03}},
                      {{1.7957670897164237e-03, 5.3781521661694287e-01,
                        6.7338565590950550e-01, 7.5919711542614232e-01,
                        7.8772792813480297e-01, 8.1561107685978818e-01,
                        8.1588981128643745e-01, 8.1589009026926218e-01,
                        8.1589036925202529e-01, 8.1616904235914178e-01,
                        8.4343591415494479e-01, 8.7007994243695330e-01,
                        9.4108224335132984e-01, 1.0000000000000000e+00,
                        7.1570218859426105e-03}},
                      {{2.7762373851026449e-06, 1.7957670897164237e-03,
                        2.5445937316078615e-03, 3.1337768724546203e-03,
                        3.3585576286345501e-03, 3.5966954684030486e-03,
                        3.5991811991506798e-03, 3.5991836882159939e-03,
                        3.5991861772829985e-03, 3.6016736010994355e-03,
                        3.8567465159746687e-03, 4.1324103770802364e-03,
                        5.0808419223605308e-03, 7.1570218859426105e-03,
                        1.0000000000000000e+00}}}};

TEST(test_radial, test_matern_32_oracle) {
  const Matern32<EuclideanDistance> cov(kMaternOracleLengthScale,
                                        kMaternOracleSigma);
  const auto pointwise = [&cov](std::size_t a, std::size_t b) {
    return cov(kOracleMaternX[a], kOracleMaternX[b]);
  };
  for (std::size_t i = 0; i < kMaternNumOraclePoints; ++i) {
    for (std::size_t j = 0; j < kMaternNumOraclePoints; ++j) {
      EXPECT_LT(fabs(pointwise(i, j) - kOracleMatern32Y[i][j]), 1e-15);
    }
  }
}

} // namespace albatross
