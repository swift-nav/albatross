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
#include "models/ransac_gp.h"
#include "outlier.h"
#include "test_utils.h"
#include <gtest/gtest.h>

namespace albatross {

TEST(test_outlier, test_ransac) {
  auto dataset = make_toy_linear_data();
  const auto model_ptr = toy_gaussian_process();

  EvaluationMetric<JointDistribution> nll =
      albatross::evaluation_metrics::negative_log_likelihood;

  dataset.targets.mean[3] = 400.;
  dataset.targets.mean[5] = -300.;

  const auto fold_indexer = leave_one_out_indexer(dataset);
  const auto modified =
      ransac(dataset, fold_indexer, model_ptr.get(), nll, 1., 3, 3, 20);

  EXPECT_EQ(modified.features.size(), dataset.features.size() - 2);

  // Make sure we threw out the correct features.
  EXPECT_EQ(std::find(modified.features.begin(), modified.features.end(),
                      dataset.features[3]),
            modified.features.end());
  EXPECT_EQ(std::find(modified.features.begin(), modified.features.end(),
                      dataset.features[5]),
            modified.features.end());
}

static inline std::unique_ptr<RegressionModel<double>>
toy_ransac_gaussian_process() {
  return ransac_gp_pointer_from_covariance<double>(toy_covariance_function());
}

TEST(test_outlier, test_ransac_gp) {
  auto dataset = make_toy_linear_data();

  const auto fold_indexer = leave_one_out_indexer(dataset);

  const auto model_ptr = toy_ransac_gaussian_process();

  EvaluationMetric<JointDistribution> nll =
      albatross::evaluation_metrics::negative_log_likelihood;

  dataset.targets.mean[3] = 400.;
  dataset.targets.mean[5] = -300.;

  const auto scores =
      cross_validated_scores(nll, dataset, fold_indexer, model_ptr.get());

  // Here we make sure the leave one out likelihoods for inliers are all
  // reasonable, and for the known outliers we assert the likelihood is
  // really really really small.
  for (Eigen::Index i = 0; i < scores.size(); i++) {
    if (i == 3 || i == 5) {
      EXPECT_GE(scores[i], 1.e5);
    } else {
      EXPECT_LE(scores[i], 0.);
    }
  }
}

} // namespace albatross
