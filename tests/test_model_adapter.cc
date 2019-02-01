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

#include "covariance_functions/covariance_functions.h"
#include "evaluate.h"
#include "models/gp.h"
#include "test_utils.h"
#include <cereal/archives/json.hpp>
#include <gtest/gtest.h>

namespace albatross {

using SqrExp = SquaredExponential<EuclideanDistance>;

using TestBaseModel = GaussianProcessRegression<Eigen::VectorXd, SqrExp>;

using TestAdaptedModelBase = AdaptedRegressionModel<double, TestBaseModel>;

class TestAdaptedModel : public TestAdaptedModelBase {
public:
  TestAdaptedModel() { this->params_["center"] = 0.; };

  std::string get_name() const override { return "test_adapted"; };

  Eigen::VectorXd convert_feature(const double &x) const override {
    Eigen::VectorXd converted(2);
    converted << 1., (x - this->get_param_value("center"));
    return converted;
  }

  /*
   * save/load methods are inherited from the SerializableRegressionModel,
   * but by defining them here and explicitly showing the inheritance
   * through the use of `base_class` we can make use of cereal's
   * polymorphic serialization.
   */
  template <class Archive> void save(Archive &archive) const {
    archive(cereal::make_nvp("test_adapted",
                             cereal::base_class<TestAdaptedModelBase>(this)));
  }

  template <class Archive> void load(Archive &archive) {
    archive(cereal::make_nvp("test_adapted",
                             cereal::base_class<TestAdaptedModelBase>(this)));
  }
};

void test_get_set(RegressionModel<double> &model, const std::string &key) {
  // Make sure a key exists, then modify it and make sure it
  // takes on the new value.
  const auto orig = model.get_param_value(key);
  model.set_param(key, orig + 1.);
  EXPECT_EQ(model.get_params().at(key), orig + 1.);
}

TEST(test_model_adapter, test_get_set_params) {
  // An adapted model should contain both higher level parameters,
  // and the sub model parameters.
  auto model = TestAdaptedModel();
  auto sqr_exp_params = SqrExp().get_params();
  auto params = model.get_params();
  // Make sure all the sub model params are in the adapted params
  for (const auto &pair : sqr_exp_params) {
    test_get_set(model, pair.first);
  }
  // And the higher level parameter.
  test_get_set(model, "center");
};

TEST(test_model_adapter, test_fit) {
  auto adpated_dataset = make_adapted_toy_linear_data();
  auto adapted_model = adapted_toy_gaussian_process();
  adapted_model->fit(adpated_dataset);
  const auto adapted_pred = adapted_model->predict(adpated_dataset.features);

  auto dataset = make_toy_linear_data();
  auto model = toy_gaussian_process();
  model->fit(dataset);
  const auto pred = model->predict(dataset.features);

  EXPECT_EQ(adapted_pred, pred);
}

TEST(test_model_adapter, test_ransac_fit) {
  auto dataset = make_adapted_toy_linear_data();
  auto adapted_model = adapted_toy_gaussian_process();
  adapted_model->fit(dataset);
  const auto adapted_pred = adapted_model->predict(dataset.features);

  const auto fold_indexer = leave_one_out_indexer(dataset);

  double inlier_threshold = 1.;
  std::size_t min_inliers = 2;
  std::size_t min_features = 3;
  std::size_t max_iterations = 20;

  auto ransac_model = adapted_model->ransac_model(inlier_threshold, min_inliers,
                                                  min_features, max_iterations);

  EvaluationMetric<JointDistribution> nll =
      evaluation_metrics::negative_log_likelihood;

  dataset.targets.mean[3] = 400.;
  dataset.targets.mean[5] = -300.;

  ransac_model->fit(dataset);

  const auto scores =
      cross_validated_scores(nll, dataset, fold_indexer, ransac_model.get());

  // Here we use the original model_ptr and make sure it also was fit after
  // we called `model_ptr->ransac_model.fit()`
  const auto in_sample_preds =
      adapted_model->template predict<Eigen::VectorXd>(dataset.features);

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
}
