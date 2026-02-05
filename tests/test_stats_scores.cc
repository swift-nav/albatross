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

#include <albatross/Common>
#include <albatross/Distribution>
#include <albatross/Evaluation>
#include <albatross/Stats>
#include <albatross/utils/RandomUtils>

#include <albatross/src/utils/eigen_utils.hpp>
#include <gtest/gtest.h>
#include <limits>

namespace albatross {

static constexpr const Eigen::Index cDistributionMaxSize{30};

template <typename RandomNumberGenerator>
JointDistribution random_distribution(Eigen::Index dimension,
                                      RandomNumberGenerator &gen) {
  const auto covariance = random_covariance_matrix(dimension, gen);
  Eigen::VectorXd mean(dimension);
  gaussian_fill(mean, gen);

  return {mean, covariance};
}

// Death tests for energy_score
TEST(test_stats, test_energy_score_death_num_samples) {
  std::default_random_engine gen(2222);
  const auto dist = random_distribution(5, gen);
  Eigen::VectorXd truth = Eigen::VectorXd::Random(5);

  EXPECT_DEATH(score::energy_score(dist, truth, {}, 22ULL, 1),
               "Cannot form an MC approximation with 1 or fewer samples");
  EXPECT_DEATH(score::energy_score(dist, truth, {}, 22ULL, 0),
               "Cannot form an MC approximation with 1 or fewer samples");
}

TEST(test_stats, test_energy_score_death_size_mismatch) {
  std::default_random_engine gen(2222);
  const auto dist = random_distribution(5, gen);
  Eigen::VectorXd truth = Eigen::VectorXd::Random(3);

  EXPECT_DEATH(score::energy_score(dist, truth),
               "Predictive distribution and truth have different sizes!");
}

// Death tests for variogram_score
TEST(test_stats, test_variogram_score_death_size_mismatch) {
  std::default_random_engine gen(2222);
  const auto dist = random_distribution(5, gen);
  Eigen::VectorXd truth = Eigen::VectorXd::Random(3);

  EXPECT_DEATH(score::variogram_score(dist, truth, {}),
               "Predictive distribution and truth have different sizes!");
}

TEST(test_stats, test_variogram_score_death_weight_mismatch) {
  std::default_random_engine gen(2222);
  const auto dist = random_distribution(5, gen);
  Eigen::VectorXd truth = Eigen::VectorXd::Random(5);
  Eigen::MatrixXd weights = Eigen::MatrixXd::Random(3, 3);

  EXPECT_DEATH(
      score::variogram_score(dist, truth, &weights),
      "Variogram score weights must be a square matrix matched to.*the size of "
      "the problem!");
}

TEST(test_stats, test_variogram_score_death_weight_non_square) {
  std::default_random_engine gen(2222);
  const auto dist = random_distribution(5, gen);
  Eigen::VectorXd truth = Eigen::VectorXd::Random(5);
  Eigen::MatrixXd weights = Eigen::MatrixXd::Random(5, 3);

  EXPECT_DEATH(
      score::variogram_score(dist, truth, &weights),
      "Variogram score weights must be a square matrix matched to.*the size of "
      "the problem!");
}

static constexpr const std::size_t cNumRandomIterations{200};

TEST(test_stats, test_expected_abs_normal_zero_mean_1) {
  // For zero mean, E[|X|] = sigma * sqrt(2/pi) for any p=1
  std::default_random_engine gen(5555);
  std::uniform_real_distribution<double> sigma_dist(0.1, 10.0);

  for (std::size_t iter = 0; iter < cNumRandomIterations; ++iter) {
    const double sigma = sigma_dist(gen);
    const double expected = sigma * std::sqrt(2.0 / M_PI);

    const double p1_specialized =
        score::detail::expected_abs_normal_1(0.0, sigma);

    EXPECT_NEAR(p1_specialized, expected, 1e-12) << "sigma=" << sigma;
  }
}

TEST(test_stats, test_expected_abs_normal_zero_mean_2) {
  std::default_random_engine gen(5555);
  std::uniform_real_distribution<double> sigma_dist(0.1, 10.0);

  for (std::size_t iter = 0; iter < cNumRandomIterations; ++iter) {
    const double sigma = sigma_dist(gen);
    const double expected =
        sigma * sigma * 2 * std::tgamma(3. / 2.) / std::sqrt(M_PI);

    const double p2_specialized =
        score::detail::expected_abs_normal_2(0.0, sigma);

    EXPECT_NEAR(p2_specialized, expected, 1e-12) << "sigma=" << sigma;
  }
}

static constexpr const Eigen::Index cMCSamples{500};

// Test energy score translation invariance
TEST(test_stats, test_energy_score_translation_invariance) {
  std::default_random_engine gen(6666);

  for (std::size_t iter = 0; iter < cNumRandomIterations; ++iter) {
    const Eigen::Index dimension = std::uniform_int_distribution<Eigen::Index>(
        2, cDistributionMaxSize)(gen);
    const auto prediction = random_distribution(dimension, gen);
    Eigen::VectorXd truth(dimension);
    gaussian_fill(truth, gen);

    // Compute original energy score
    const double es_original =
        score::energy_score(prediction, truth, {}, 222ULL, 1000);

    // Apply random translation
    Eigen::VectorXd offset(dimension);
    gaussian_fill(offset, gen);

    JointDistribution translated_prediction(prediction);
    translated_prediction.mean += offset;
    Eigen::VectorXd translated_truth = truth + offset;

    // Compute translated energy score
    const double es_translated = score::energy_score(
        translated_prediction, translated_truth, {}, 22ULL, 1000);

    // We have a dynamic threshold that accounts for MC estimation
    // variance.  Increase `cMCSamples` and this threshold will get
    // tighter.
    const double se_bound =
        std::sqrt(2.0 *
                  (prediction.covariance.trace() +
                   (prediction.mean - truth).squaredNorm()) /
                  static_cast<double>(cMCSamples));
    EXPECT_NEAR(es_original, es_translated, 2. * se_bound)
        << "Translation invariance violated at iteration " << iter
        << ", dimension " << dimension;
  }
}

// Test energy score rotation invariance
TEST(test_stats, test_energy_score_rotation_invariance) {
  std::default_random_engine gen(7777);

  for (std::size_t iter = 0; iter < cNumRandomIterations; ++iter) {
    const Eigen::Index dimension = std::uniform_int_distribution<Eigen::Index>(
        2, cDistributionMaxSize)(gen);
    const auto prediction = random_distribution(dimension, gen);
    Eigen::VectorXd truth(dimension);
    gaussian_fill(truth, gen);

    // Compute original energy score
    const double es_original =
        score::energy_score(prediction, truth, {}, 222ULL, cMCSamples);

    // Generate random rotation matrix via QR decomposition
    Eigen::MatrixXd random_matrix(dimension, dimension);
    gaussian_fill(random_matrix, gen);
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(random_matrix);
    Eigen::MatrixXd Q = qr.householderQ();

    // Apply rotation
    JointDistribution rotated_prediction(prediction);
    rotated_prediction.mean = Q * prediction.mean;
    rotated_prediction.covariance = Q * prediction.covariance * Q.transpose();
    Eigen::VectorXd rotated_truth = Q * truth;

    // Compute rotated energy score
    const double es_rotated = score::energy_score(
        rotated_prediction, rotated_truth, {}, 22ULL, cMCSamples);

    // We have a dynamic threshold that accounts for MC estimation
    // variance.  Increase `cMCSamples` and this threshold will get
    // tighter.
    const double se_bound =
        std::sqrt(2.0 *
                  (prediction.covariance.trace() +
                   (prediction.mean - truth).squaredNorm()) /
                  static_cast<double>(cMCSamples));
    EXPECT_NEAR(es_original, es_rotated, 2. * se_bound)
        << "Rotation invariance violated at iteration " << iter
        << ", dimension " << dimension;
  }
}

// Test energy score matches CRPS for 1-dimensional distributions
TEST(test_stats, test_energy_score_matches_crps_1d) {
  std::default_random_engine gen(9999);
  std::uniform_real_distribution<double> mu_dist(-10.0, 10.0);
  std::uniform_real_distribution<double> sigma_dist(0.1, 5.0);

  for (std::size_t iter = 0; iter < cNumRandomIterations; ++iter) {
    // Create 1-dimensional distribution
    const double mu = mu_dist(gen);
    const double sigma = sigma_dist(gen);
    const double truth_val = mu_dist(gen);

    JointDistribution prediction;
    prediction.mean = Eigen::VectorXd(1);
    prediction.mean(0) = mu;
    prediction.covariance = Eigen::MatrixXd(1, 1);
    prediction.covariance(0, 0) = sigma * sigma;

    Eigen::VectorXd truth(1);
    truth(0) = truth_val;

    // Compute energy score
    const double es =
        score::energy_score(prediction, truth, {}, 444ULL, cMCSamples);

    // Compute CRPS
    const double crps = score::crps_normal(mu, sigma, truth_val);

    // Energy score should match CRPS for 1D case
    // Allow tolerance based on MC sampling variance
    const double se_bound =
        std::sqrt(2.0 * sigma * sigma / static_cast<double>(cMCSamples));

    EXPECT_NEAR(es, crps, 2.0 * se_bound)
        << "ES/CRPS mismatch at iteration " << iter << ", mu=" << mu
        << ", sigma=" << sigma << ", truth=" << truth_val;
  }
}

// Test energy score approaches deterministic case as variance -> 0
TEST(test_stats, test_energy_score_approaches_deterministic) {
  std::default_random_engine gen(8888);

  for (std::size_t iter = 0; iter < cNumRandomIterations; ++iter) {
    const Eigen::Index dimension = std::uniform_int_distribution<Eigen::Index>(
        2, cDistributionMaxSize)(gen);

    // Create prediction with random mean
    JointDistribution prediction;
    prediction.mean = Eigen::VectorXd(dimension);
    gaussian_fill(prediction.mean, gen);

    // Create random truth vector
    Eigen::VectorXd truth(dimension);
    gaussian_fill(truth, gen);

    // Expected deterministic score (2-norm of difference)
    const double expected_score = (prediction.mean - truth).norm();

    // Test with progressively smaller variances
    std::vector<double> variances = {1e-4, 1e-6, 1e-8, 1e-10};

    for (const double var : variances) {
      // Create diagonal covariance with small variance
      prediction.covariance =
          var * Eigen::MatrixXd::Identity(dimension, dimension);

      const double es =
          score::energy_score(prediction, truth, {}, 333ULL, cMCSamples);

      // As variance -> 0, energy score should approach the deterministic
      // distance
      //
      // I manually fixed this to verify that it's doing something
      // relative to what the method computes.
      const double tolerance = std::sqrt(var) * dimension + 1e-10;

      EXPECT_NEAR(es, expected_score, tolerance)
          << "Deterministic limit failed at iteration " << iter
          << ", dimension " << dimension << ", variance " << var;
    }
  }
}

// Test energy score scaling property
TEST(test_stats, test_energy_score_scaling) {
  std::default_random_engine gen(9000);
  std::uniform_real_distribution<double> scale_dist(0.1, 10.0);

  for (std::size_t iter = 0; iter < cNumRandomIterations; ++iter) {
    const Eigen::Index dimension = std::uniform_int_distribution<Eigen::Index>(
        2, cDistributionMaxSize)(gen);
    const auto prediction = random_distribution(dimension, gen);
    Eigen::VectorXd truth(dimension);
    gaussian_fill(truth, gen);

    // Compute original energy score
    const double es_original =
        score::energy_score(prediction, truth, {}, 111ULL, cMCSamples);

    // Generate random positive scale factor
    const double scale = scale_dist(gen);

    // Scale both prediction and truth
    JointDistribution scaled_prediction = prediction * scale;
    Eigen::VectorXd scaled_truth = truth * scale;

    // Compute scaled energy score
    const double es_scaled = score::energy_score(
        scaled_prediction, scaled_truth, {}, 111ULL, cMCSamples);

    // Energy score should scale linearly with the scale factor
    // Allow tolerance based on MC sampling variance
    const double se_bound =
        std::sqrt(2.0 *
                  (prediction.covariance.trace() +
                   (prediction.mean - truth).squaredNorm()) /
                  static_cast<double>(cMCSamples));

    EXPECT_NEAR(es_scaled, scale * es_original, 2.0 * scale * se_bound)
        << "Energy score scaling property violated at iteration " << iter
        << ", dimension " << dimension << ", scale " << scale;
  }
}

// Parameterized tests for variogram score properties
class VariogramScorePropertiesTest
    : public ::testing::TestWithParam<score::VariogramScoreOrder> {
protected:
  double get_p_value() const {
    return GetParam() == score::VariogramScoreOrder::cMadogram ? 1.0 : 2.0;
  }

  std::string get_order_name() const {
    return GetParam() == score::VariogramScoreOrder::cMadogram ? "madogram"
                                                               : "variogram";
  }
};

// Test variogram score mean offset invariance
TEST_P(VariogramScorePropertiesTest, MeanOffsetInvariance) {
  const auto order = GetParam();
  std::default_random_engine gen(10000);
  std::uniform_real_distribution<double> offset_dist(-10.0, 10.0);

  for (std::size_t iter = 0; iter < cNumRandomIterations; ++iter) {
    const Eigen::Index dimension = std::uniform_int_distribution<Eigen::Index>(
        2, cDistributionMaxSize)(gen);
    const auto prediction = random_distribution(dimension, gen);
    Eigen::VectorXd truth(dimension);
    gaussian_fill(truth, gen);

    // Compute original variogram score
    const double vs_original =
        score::variogram_score(prediction, truth, nullptr, order);

    // Generate random constant offset
    const double offset_value = offset_dist(gen);
    const Eigen::VectorXd offset =
        Eigen::VectorXd::Constant(dimension, offset_value);

    // Test 1: Offset prediction mean only
    JointDistribution offset_prediction(prediction);
    offset_prediction.mean += offset;
    const double vs_offset_prediction =
        score::variogram_score(offset_prediction, truth, nullptr, order);

    EXPECT_NEAR(vs_original, vs_offset_prediction, 1e-10)
        << "Variogram score (" << get_order_name()
        << ") changed when offsetting prediction mean at iteration " << iter
        << ", dimension " << dimension << ", offset " << offset_value;

    // Test 2: Offset truth only
    Eigen::VectorXd offset_truth = truth + offset;
    const double vs_offset_truth =
        score::variogram_score(prediction, offset_truth, nullptr, order);

    EXPECT_NEAR(vs_original, vs_offset_truth, 1e-10)
        << "Variogram score (" << get_order_name()
        << ") changed when offsetting truth at iteration " << iter
        << ", dimension " << dimension << ", offset " << offset_value;
  }
}

// Test variogram score scaling property
TEST_P(VariogramScorePropertiesTest, Scaling) {
  const auto order = GetParam();
  const double p = get_p_value();
  std::default_random_engine gen(11000);
  std::uniform_real_distribution<double> scale_dist(0.1, 10.0);

  for (std::size_t iter = 0; iter < cNumRandomIterations; ++iter) {
    const Eigen::Index dimension = std::uniform_int_distribution<Eigen::Index>(
        2, cDistributionMaxSize)(gen);
    const auto prediction = random_distribution(dimension, gen);
    Eigen::VectorXd truth(dimension);
    gaussian_fill(truth, gen);

    // Compute original variogram score
    const double vs_original =
        score::variogram_score(prediction, truth, nullptr, order);

    // Generate random positive scale factor
    const double scale = scale_dist(gen);

    // Scale both prediction and truth
    JointDistribution scaled_prediction = prediction * scale;
    Eigen::VectorXd scaled_truth = truth * scale;

    // Compute scaled variogram score
    const double vs_scaled =
        score::variogram_score(scaled_prediction, scaled_truth, nullptr, order);

    // Variogram score should scale by c^(2*p)
    const double expected_scale = std::pow(scale, 2.0 * p);

    // Use slightly relaxed tolerance for p=2 due to higher power
    const double tolerance = (p == 2.0) ? 1e-8 : 1e-10;

    EXPECT_NEAR(vs_scaled, expected_scale * vs_original, tolerance)
        << "Variogram score (" << get_order_name()
        << ") scaling property violated at iteration " << iter << ", dimension "
        << dimension << ", scale " << scale;
  }
}

// The random perturbation tests are even more expensive than normal
// random tests, so we reduce sizes here.
static constexpr const std::size_t cNumPerturbedSamples{40};
static constexpr const std::size_t cNumPerturbationRandomIterations{40};
static constexpr const std::size_t cPerturbationDistributionMaxSize{12};

// Test variogram score is proper: mean perturbation
TEST_P(VariogramScorePropertiesTest, ProperScoringMeanPerturbation) {
  const auto order = GetParam();
  std::default_random_engine gen(13000);

  for (std::size_t iter = 0; iter < cNumPerturbationRandomIterations; ++iter) {
    const Eigen::Index dimension = std::uniform_int_distribution<Eigen::Index>(
        2, cPerturbationDistributionMaxSize)(gen);

    // Generate true distribution
    const auto true_dist = random_distribution(dimension, gen);

    // Create perturbed distribution by shifting mean
    JointDistribution perturbed_dist(true_dist);
    Eigen::VectorXd mean_perturbation(dimension);
    gaussian_fill(mean_perturbation, gen);
    perturbed_dist.mean += mean_perturbation;

    // Draw many samples from true distribution and average scores
    double avg_score_true = 0.0;
    double avg_score_perturbed = 0.0;

    for (std::size_t sample = 0; sample < cNumPerturbedSamples; ++sample) {
      Eigen::VectorXd y = random_multivariate_normal(true_dist.covariance, gen);
      y += true_dist.mean;

      avg_score_true += score::variogram_score(true_dist, y, nullptr, order);
      avg_score_perturbed +=
          score::variogram_score(perturbed_dist, y, nullptr, order);
    }

    avg_score_true /= static_cast<double>(cNumPerturbedSamples);
    avg_score_perturbed /= static_cast<double>(cNumPerturbedSamples);

    // Proper scoring rule: true distribution should have lower expected score
    EXPECT_LT(avg_score_true, avg_score_perturbed)
        << "Variogram score (" << get_order_name()
        << ") proper scoring property (mean) violated at iteration " << iter
        << ", dimension " << dimension << ", avg_true=" << avg_score_true
        << ", avg_perturbed=" << avg_score_perturbed;
  }
}

INSTANTIATE_TEST_SUITE_P(
    VariogramOrders, VariogramScorePropertiesTest,
    ::testing::Values(score::VariogramScoreOrder::cMadogram,
                      score::VariogramScoreOrder::cVariogram),
    [](const ::testing::TestParamInfo<score::VariogramScoreOrder> &info) {
      return info.param == score::VariogramScoreOrder::cMadogram ? "Madogram"
                                                                 : "Variogram";
    });

// Test energy score is proper: mean perturbation
TEST(test_stats, test_energy_score_proper_scoring_mean_perturbation) {
  std::default_random_engine gen(12000);
  for (std::size_t iter = 0; iter < cNumPerturbationRandomIterations; ++iter) {
    const Eigen::Index dimension = std::uniform_int_distribution<Eigen::Index>(
        2, cPerturbationDistributionMaxSize)(gen);

    // Generate true distribution
    const auto true_dist = random_distribution(dimension, gen);

    // Create perturbed distribution by shifting mean
    JointDistribution perturbed_dist(true_dist);
    Eigen::VectorXd mean_perturbation(dimension);
    gaussian_fill(mean_perturbation, gen);
    perturbed_dist.mean += mean_perturbation;

    // Draw many samples from true distribution and average scores
    double avg_score_true = 0.0;
    double avg_score_perturbed = 0.0;

    for (std::size_t sample = 0; sample < cNumPerturbedSamples; ++sample) {
      Eigen::VectorXd y = random_multivariate_normal(true_dist.covariance, gen);
      y += true_dist.mean;

      avg_score_true +=
          score::energy_score(true_dist, y, {}, 555ULL, cMCSamples);
      avg_score_perturbed +=
          score::energy_score(perturbed_dist, y, {}, 555ULL, cMCSamples);
    }

    avg_score_true /= static_cast<double>(cNumPerturbedSamples);
    avg_score_perturbed /= static_cast<double>(cNumPerturbedSamples);

    // Proper scoring rule: true distribution should have lower expected score
    EXPECT_LT(avg_score_true, avg_score_perturbed)
        << "Energy score proper scoring property (mean) violated at iteration "
        << iter << ", dimension " << dimension
        << ", avg_true=" << avg_score_true
        << ", avg_perturbed=" << avg_score_perturbed;
  }
}

// Test energy score is proper: covariance perturbation
TEST(test_stats, test_energy_score_proper_scoring_covariance_perturbation) {
  std::default_random_engine gen(12001);
  for (std::size_t iter = 0; iter < cNumPerturbationRandomIterations; ++iter) {
    const Eigen::Index dimension = std::uniform_int_distribution<Eigen::Index>(
        2, cPerturbationDistributionMaxSize)(gen);

    // Generate true distribution
    const auto true_dist = random_distribution(dimension, gen);

    // Create perturbed distribution by adding PD noise to covariance
    JointDistribution perturbed_dist(true_dist);
    Eigen::MatrixXd random_matrix(dimension, dimension);
    gaussian_fill(random_matrix, gen);
    const double scale = 0.5; // Scale factor for perturbation
    perturbed_dist.covariance +=
        scale * random_matrix * random_matrix.transpose();

    // Draw many samples from true distribution and average scores
    double avg_score_true = 0.0;
    double avg_score_perturbed = 0.0;

    for (std::size_t sample = 0; sample < cNumPerturbedSamples; ++sample) {
      Eigen::VectorXd y = random_multivariate_normal(true_dist.covariance, gen);
      y += true_dist.mean;

      avg_score_true +=
          score::energy_score(true_dist, y, {}, 666ULL, cMCSamples);
      avg_score_perturbed +=
          score::energy_score(perturbed_dist, y, {}, 666ULL, cMCSamples);
    }

    avg_score_true /= static_cast<double>(cNumPerturbedSamples);
    avg_score_perturbed /= static_cast<double>(cNumPerturbedSamples);

    // Proper scoring rule: true distribution should have lower expected score
    EXPECT_LT(avg_score_true, avg_score_perturbed)
        << "Energy score proper scoring property (covariance) violated at "
           "iteration "
        << iter << ", dimension " << dimension
        << ", avg_true=" << avg_score_true
        << ", avg_perturbed=" << avg_score_perturbed;
  }
}

// Test energy score is proper: mean and covariance perturbation
TEST(test_stats,
     test_energy_score_proper_scoring_mean_and_covariance_perturbation) {
  std::default_random_engine gen(12002);
  for (std::size_t iter = 0; iter < cNumPerturbationRandomIterations; ++iter) {
    const Eigen::Index dimension = std::uniform_int_distribution<Eigen::Index>(
        2, cPerturbationDistributionMaxSize)(gen);

    // Generate true distribution
    const auto true_dist = random_distribution(dimension, gen);

    // Create perturbed distribution by perturbing both mean and covariance
    JointDistribution perturbed_dist(true_dist);

    // Perturb mean
    Eigen::VectorXd mean_perturbation(dimension);
    gaussian_fill(mean_perturbation, gen);
    perturbed_dist.mean += mean_perturbation;

    // Perturb covariance
    Eigen::MatrixXd random_matrix(dimension, dimension);
    gaussian_fill(random_matrix, gen);
    const double scale = 0.5;
    perturbed_dist.covariance +=
        scale * random_matrix * random_matrix.transpose();

    // Draw many samples from true distribution and average scores
    double avg_score_true = 0.0;
    double avg_score_perturbed = 0.0;

    for (std::size_t sample = 0; sample < cNumPerturbedSamples; ++sample) {
      Eigen::VectorXd y = random_multivariate_normal(true_dist.covariance, gen);
      y += true_dist.mean;

      avg_score_true +=
          score::energy_score(true_dist, y, {}, 777ULL, cMCSamples);
      avg_score_perturbed +=
          score::energy_score(perturbed_dist, y, {}, 777ULL, cMCSamples);
    }

    avg_score_true /= static_cast<double>(cNumPerturbedSamples);
    avg_score_perturbed /= static_cast<double>(cNumPerturbedSamples);

    // Proper scoring rule: true distribution should have lower expected score
    EXPECT_LT(avg_score_true, avg_score_perturbed)
        << "Energy score proper scoring property (mean+cov) violated at "
           "iteration "
        << iter << ", dimension " << dimension
        << ", avg_true=" << avg_score_true
        << ", avg_perturbed=" << avg_score_perturbed;
  }
}

// Test CRPS non-finite inputs
TEST(test_stats, test_crps_normal_non_finite_inputs) {
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();

  // Non-finite mu
  EXPECT_TRUE(std::isnan(score::crps_normal(inf, 1.0, 0.0)));
  EXPECT_TRUE(std::isnan(score::crps_normal(-inf, 1.0, 0.0)));
  EXPECT_TRUE(std::isnan(score::crps_normal(nan, 1.0, 0.0)));

  // Non-finite sigma
  EXPECT_TRUE(std::isnan(score::crps_normal(0.0, inf, 0.0)));
  EXPECT_TRUE(std::isnan(score::crps_normal(0.0, nan, 0.0)));

  // Non-finite y (truth)
  EXPECT_TRUE(std::isnan(score::crps_normal(0.0, 1.0, inf)));
  EXPECT_TRUE(std::isnan(score::crps_normal(0.0, 1.0, -inf)));
  EXPECT_TRUE(std::isnan(score::crps_normal(0.0, 1.0, nan)));
}

TEST(test_stats, test_crps_normal_degenerate_sigma) {
  // sigma = 0 should return absolute error
  EXPECT_DOUBLE_EQ(score::crps_normal(5.0, 0.0, 3.0), 2.0);
  EXPECT_DOUBLE_EQ(score::crps_normal(5.0, 0.0, 5.0), 0.0);
  EXPECT_DOUBLE_EQ(score::crps_normal(5.0, 0.0, 8.0), 3.0);

  // Negative sigma should also return absolute error
  EXPECT_DOUBLE_EQ(score::crps_normal(5.0, -1.0, 3.0), 2.0);
}

// Test energy score with weights
TEST(test_stats, test_energy_score_with_weights) {
  std::default_random_engine gen(14000);

  for (std::size_t iter = 0; iter < cNumRandomIterations; ++iter) {
    const Eigen::Index dimension = std::uniform_int_distribution<Eigen::Index>(
        2, cDistributionMaxSize)(gen);
    const auto prediction = random_distribution(dimension, gen);
    Eigen::VectorXd truth(dimension);
    gaussian_fill(truth, gen);

    // Uniform weights should match no weights
    Eigen::VectorXd uniform_weights = Eigen::VectorXd::Ones(dimension);
    const double es_no_weights =
        score::energy_score(prediction, truth, nullptr, 123ULL, cMCSamples);
    const double es_uniform_weights = score::energy_score(
        prediction, truth, &uniform_weights, 123ULL, cMCSamples);

    EXPECT_NEAR(es_no_weights, es_uniform_weights, 1e-10)
        << "Uniform weights should match no weights at iteration " << iter;

    // Zero weight on a dimension should ignore errors in that dimension
    Eigen::VectorXd zero_first_weight = Eigen::VectorXd::Ones(dimension);
    zero_first_weight(0) = 0.0;

    // Create prediction with large error only in first dimension
    JointDistribution offset_prediction(prediction);
    offset_prediction.mean(0) += 1000.0;

    const double es_with_large_error = score::energy_score(
        offset_prediction, truth, nullptr, 456ULL, cMCSamples);
    const double es_zero_weight_large_error = score::energy_score(
        offset_prediction, truth, &zero_first_weight, 456ULL, cMCSamples);

    // Score with zero weight should be much smaller
    EXPECT_LT(es_zero_weight_large_error, es_with_large_error * 0.5)
        << "Zero weight should reduce impact of error in that dimension";
  }
}

// Test variogram score with weights
TEST(test_stats, test_variogram_score_with_weights) {
  std::default_random_engine gen(15000);

  for (std::size_t iter = 0; iter < cNumRandomIterations; ++iter) {
    const Eigen::Index dimension = std::uniform_int_distribution<Eigen::Index>(
        3, cDistributionMaxSize)(gen);
    const auto prediction = random_distribution(dimension, gen);
    Eigen::VectorXd truth(dimension);
    gaussian_fill(truth, gen);

    // Uniform weights should match no weights
    Eigen::MatrixXd uniform_weights =
        Eigen::MatrixXd::Ones(dimension, dimension);
    const double vs_no_weights =
        score::variogram_score(prediction, truth, nullptr);
    const double vs_uniform_weights =
        score::variogram_score(prediction, truth, &uniform_weights);

    EXPECT_NEAR(vs_no_weights, vs_uniform_weights, 1e-10)
        << "Uniform weights should match no weights at iteration " << iter;

    // Zero weights should give zero score
    Eigen::MatrixXd zero_weights = Eigen::MatrixXd::Zero(dimension, dimension);
    const double vs_zero_weights =
        score::variogram_score(prediction, truth, &zero_weights);

    EXPECT_DOUBLE_EQ(vs_zero_weights, 0.0)
        << "Zero weights should give zero variogram score";
  }
}

// Death tests for draw_mvn
TEST(test_stats, test_draw_mvn_death_non_positive_definite) {
  std::default_random_engine gen(2222);
  Eigen::MatrixXd non_pd(3, 3);
  non_pd << 1, 0, 0, 0, -1, 0, 0, 0, 1;
  Eigen::LDLT<Eigen::MatrixXd> bad_decomp(non_pd);
  Eigen::VectorXd mean = Eigen::VectorXd::Random(3);

  EXPECT_DEATH(score::detail::draw_mvn(bad_decomp, mean, 10, gen),
               "Please pass a positive definite covariance!");
}

} // namespace albatross