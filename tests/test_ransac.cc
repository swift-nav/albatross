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

  NegativeLogLikelihood<JointDistribution> nll;
  LeaveOneOutLikelihood<> loo_nll;

  const auto indexer = build_indexer(LeaveOneOut(), dataset.features);

  auto ransac_functions =
      get_generic_ransac_functions(model, dataset, indexer, nll, loo_nll);

  double inlier_threshold = 1.;
  std::size_t sample_size = 3;
  std::size_t min_consensus_size = 3;
  std::size_t max_iterations = 20;

  const auto inliers = ransac(ransac_functions, indexer, inlier_threshold,
                              sample_size, min_consensus_size, max_iterations);

  EXPECT_EQ(inliers.size(), dataset.features.size() - bad_inds.size());

  for (const auto &i : bad_inds) {
    // Make sure we threw out the correct features.
    EXPECT_EQ(std::find(inliers.begin(), inliers.end(), i),
              inliers.end());
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
  const auto inlier_names =
      ransac(ransac_functions, indexer, inlier_threshold, sample_size,
             min_consensus_size, max_iterations);
  const auto inlier_inds = indices_from_groups(indexer, inlier_names);
  const auto inlier_dataset = subset(dataset, inlier_inds);

  const auto direct_pred = model.fit(inlier_dataset).predict(dataset.features);
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


struct GroupByModulo {

  std::string operator()(const double &x) const{
    const int x_int = static_cast<int>(x);
    return std::to_string(x_int % 4);
  }

};

TEST(test_outlier, test_ransac_groups) {
  const MakeGaussianProcess test_case;
  auto dataset = test_case.get_dataset();
  auto model = test_case.get_model();

  dataset.targets.mean[5] = -300.;

  NegativeLogLikelihood<JointDistribution> nll;
  GroupByModulo group_by_modulo;
  LeaveOneOutLikelihood<> consensus_metric;

  const auto ransac_strategy =
      get_generic_ransac_strategy(nll, consensus_metric, group_by_modulo);
  const auto indexer = ransac_strategy.get_indexer(dataset);
  const auto ransac_functions = ransac_strategy(model, dataset);

  const auto inlier_groups = ransac(ransac_functions, indexer, 0., 1, 1, 20);

  EXPECT_LE(inlier_groups.size(), indexer.size());
}

} // namespace albatross
