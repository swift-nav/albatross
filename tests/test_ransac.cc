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
#include <gtest/gtest.h>

#include "test_models.h"

#include <albatross/Evaluation>
#include <albatross/Ransac>

namespace albatross {

TEST(test_outlier, test_ransac_direct) {
  const MakeGaussianProcess test_case;
  auto dataset = test_case.get_dataset();
  auto model = test_case.get_model();

  std::vector<std::size_t> bad_inds = {3, 5};
  for (const auto &i : bad_inds) {
    dataset.targets.mean[static_cast<Eigen::Index>(i)] = pow(-1, i) * 400.;
  }

  const auto indexer = dataset.group_by(LeaveOneOutGrouper()).indexers();
  NegativeLogLikelihood<JointDistribution> nll;
  LeaveOneOutLikelihood<> loo_nll;

  auto ransac_functions =
      get_generic_ransac_functions(model, dataset, indexer, nll, loo_nll);

  double inlier_threshold = 1.;
  std::size_t sample_size = 3;
  std::size_t min_consensus_size = 3;
  std::size_t max_iterations = 20;

  const auto result = ransac(ransac_functions, indexer, inlier_threshold,
                             sample_size, min_consensus_size, max_iterations);

  const auto consensus = result.best.consensus();
  EXPECT_EQ(consensus.size(), dataset.features.size() - bad_inds.size());
  EXPECT_TRUE(ransac_success(result.return_code));
  EXPECT_FALSE(std::isnan(result.best.consensus_metric_value));

  for (const auto &i : bad_inds) {
    // Make sure we threw out the correct features.
    EXPECT_EQ(std::find(consensus.begin(), consensus.end(), i),
              consensus.end());
    EXPECT_TRUE(map_contains(result.best.outliers, i));
  }
}

TEST(test_outlier, test_ransac_model) {
  const MakeGaussianProcess test_case;
  auto dataset = test_case.get_dataset();
  auto model = test_case.get_model();

  std::vector<std::size_t> bad_inds = {3, 5};
  for (const auto &i : bad_inds) {
    dataset.targets.mean[static_cast<Eigen::Index>(i)] = pow(-1, i) * 400.;
  }

  DefaultRansacStrategy ransac_strategy;

  double inlier_threshold = 1.;
  std::size_t sample_size = 3;
  std::size_t min_consensus_size = 3;
  std::size_t max_iterations = 20;
  const auto ransac_model =
      model.ransac(ransac_strategy, inlier_threshold, sample_size,
                   min_consensus_size, max_iterations);
  const auto fit_model = ransac_model.fit(dataset);

  const auto pred = fit_model.predict(dataset.features);
  expect_predict_variants_consistent(pred);

  const auto ransac_functions = ransac_strategy(model, dataset);

  const auto indexer = ransac_strategy.get_indexer(dataset);
  const auto result = ransac(ransac_functions, indexer, inlier_threshold,
                             sample_size, min_consensus_size, max_iterations);
  const auto consensus_keys = result.best.consensus();
  const auto consensus_inds = indices_from_groups(indexer, consensus_keys);
  const auto consensus_dataset = subset(dataset, consensus_inds);

  const auto direct_pred =
      model.fit(consensus_dataset).predict(dataset.features);
  expect_predict_variants_consistent(direct_pred);

  EXPECT_EQ(pred.mean(), direct_pred.mean());

  NegativeLogLikelihood<JointDistribution> nll;
  const auto cv_nll =
      ransac_model.cross_validate().scores(nll, dataset, indexer);

  // The cross validated outliers should look very unlikely
  EXPECT_GE(subset(cv_nll, bad_inds).minCoeff(), 1e4);

  // But the valid data point should have reasonable likelihood
  EXPECT_LE(
      subset(cv_nll, indices_complement(bad_inds, dataset.size())).maxCoeff(),
      1.);
}

// Group values by interval, but return keys that once sorted won't be
// in order
std::string group_by_modulo(const double &x) {
  const int x_int = static_cast<int>(x);
  return std::to_string(x_int % 4);
}

TEST(test_outlier, test_ransac_groups) {
  const MakeGaussianProcess test_case;
  auto dataset = test_case.get_dataset();
  auto model = test_case.get_model();

  dataset.targets.mean[5] = -300.;

  NegativeLogLikelihood<JointDistribution> nll;
  LeaveOneOutLikelihood<> consensus_metric;

  const auto ransac_strategy =
      get_generic_ransac_strategy(nll, consensus_metric, group_by_modulo);
  const auto indexer = ransac_strategy.get_indexer(dataset);
  const auto ransac_functions = ransac_strategy(model, dataset);

  const auto result = ransac(ransac_functions, indexer, 0., 1, 1, 20);
  EXPECT_TRUE(ransac_success(result.return_code));
  EXPECT_LE(result.best.consensus().size(), indexer.size());
}

inline bool never_accept_candidates(const std::vector<std::string> &) {
  return false;
}

RansacConfig get_reasonable_ransac_config() {
  RansacConfig config;
  config.inlier_threshold = 1.;
  config.max_failed_candidates = 0;
  config.max_iterations = 20;
  config.min_consensus_size = 2;
  config.random_sample_size = 1;
  return config;
}

TEST(test_outlier, test_ransac_edge_cases) {
  const MakeGaussianProcess test_case;
  auto dataset = test_case.get_dataset();
  auto model = test_case.get_model();

  NegativeLogLikelihood<JointDistribution> nll;
  LeaveOneOutLikelihood<> consensus_metric;

  const auto ransac_strategy =
      get_generic_ransac_strategy(nll, consensus_metric, group_by_modulo);
  const auto indexer = ransac_strategy.get_indexer(dataset);
  auto ransac_functions = ransac_strategy(model, dataset);

  auto bad_inlier_config = get_reasonable_ransac_config();
  bad_inlier_config.inlier_threshold = -HUGE_VAL;

  auto result = ransac(ransac_functions, indexer, bad_inlier_config);
  EXPECT_EQ(result.return_code, RANSAC_RETURN_CODE_NO_CONSENSUS);

  auto bad_consensus_size_config = get_reasonable_ransac_config();
  bad_consensus_size_config.min_consensus_size = indexer.size();
  result = ransac(ransac_functions, indexer, bad_consensus_size_config);
  EXPECT_EQ(result.return_code, RANSAC_RETURN_CODE_INVALID_ARGUMENTS);

  auto bad_random_sample_size_config = get_reasonable_ransac_config();
  bad_random_sample_size_config.random_sample_size = indexer.size();
  result = ransac(ransac_functions, indexer, bad_random_sample_size_config);
  EXPECT_EQ(result.return_code, RANSAC_RETURN_CODE_INVALID_ARGUMENTS);

  auto bad_max_iterations_config = get_reasonable_ransac_config();
  bad_max_iterations_config.max_iterations = 0;
  result = ransac(ransac_functions, indexer, bad_max_iterations_config);
  EXPECT_EQ(result.return_code, RANSAC_RETURN_CODE_INVALID_ARGUMENTS);

  auto bad_is_valid_candidate = get_reasonable_ransac_config();
  ransac_functions.is_valid_candidate = never_accept_candidates;
  bad_is_valid_candidate.max_failed_candidates = 3;
  result = ransac(ransac_functions, indexer, bad_is_valid_candidate);
  EXPECT_EQ(result.return_code,
            RANSAC_RETURN_CODE_EXCEEDED_MAX_FAILED_CANDIDATES);
}

} // namespace albatross
