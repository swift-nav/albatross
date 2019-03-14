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

// Group values by interval, but return keys that once sorted won't be
// in order
std::string group_by_modulo(const double &x) {
  const int x_int = static_cast<int>(x);
  return std::to_string(x_int % 4);
}

TEST(test_outlier, test_ransac_groups) {
  auto dataset = make_toy_linear_data();
  const auto model_ptr = toy_gaussian_process();

  EvaluationMetric<JointDistribution> nll =
      albatross::evaluation_metrics::negative_log_likelihood;

  dataset.targets.mean[5] = -300.;

  const auto fold_indexer =
      leave_one_group_out_indexer<double>(dataset, group_by_modulo);
  const auto modified =
      ransac(dataset, fold_indexer, model_ptr.get(), nll, 0., 1, 1, 20);

  EXPECT_LE(modified.features.size(), dataset.features.size());
}

TEST(test_outlier, test_ransac_gp) {
  auto dataset = make_toy_linear_data();

  const auto fold_indexer = leave_one_out_indexer(dataset);

  const auto model_ptr = toy_gaussian_process();

  double inlier_threshold = 1.;
  std::size_t min_inliers = 2;
  std::size_t min_features = 3;
  std::size_t max_iterations = 20;

  auto ransac_model = model_ptr->ransac_model(inlier_threshold, min_inliers,
                                              min_features, max_iterations);

  EvaluationMetric<JointDistribution> nll =
      albatross::evaluation_metrics::negative_log_likelihood;

  dataset.targets.mean[3] = 400.;
  dataset.targets.mean[5] = -300.;

  ransac_model->fit(dataset);

  const auto scores =
      cross_validated_scores(nll, dataset, fold_indexer, ransac_model.get());

  // Here we use the original model_ptr and make sure it also was fit after
  // we called `model_ptr->ransac_model.fit()`
  const auto in_sample_preds =
      model_ptr->template predict<Eigen::VectorXd>(dataset.features);

  // Here we make sure the leave one out likelihoods for inliers are all
  // reasonable, and for the known outliers we assert the likelihood is
  // really really really small.
  for (Eigen::Index i = 0; i < scores.size(); i++) {
    double in_sample_error = fabs(in_sample_preds[i] - dataset.targets.mean[i]);
    if (i == 3 || i == 5) {
      EXPECT_GE(scores[i], 1.e5);
      EXPECT_GE(in_sample_error, 100.);
    } else {
      EXPECT_LE(scores[i], 0.);
      EXPECT_LE(in_sample_error, 0.1);
    }
  }
}

} // namespace albatross
