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

#include "test_utils.h"
#include "GP"


namespace albatross {

 template <typename CovFunc>
 class TestAdaptedModel : public GaussianProcessBase<double, CovFunc, TestAdaptedModel<CovFunc>> {
 public:
   using Base = GaussianProcessBase<double, CovFunc, TestAdaptedModel>;
//   friend Base;

   TestAdaptedModel() {
     this->params_["center"] = {1., std::make_shared<UniformPrior>(-10., 10.)};
   }

   std::vector<double> convert(const std::vector<AdaptedFeature> &features) const {
     std::vector<double> converted;
     for (const auto &f : features) {
       converted.push_back(f.value - this->get_param_value("center"));
     }
     return converted;
   }

   auto fit_impl_(const std::vector<AdaptedFeature> &features,
                  const MarginalDistribution &targets) const {
     return Base::fit_impl_(convert(features), targets);
   }

   JointDistribution predict_(const std::vector<AdaptedFeature> &features,
                        PredictTypeIdentity<JointDistribution> &&) const {
     return Base::predict_(convert(features), PredictTypeIdentity<JointDistribution>());
   }

 };

template <typename ModelType>
void test_get_set(ModelType &model, const std::string &key) {
  // Make sure a key exists, then modify it and make sure it
  // takes on the new value.
  const auto orig = model.get_param_value(key);
  model.set_param(key, orig + 1.);
  EXPECT_EQ(model.get_params().at(key), orig + 1.);
}

TEST(test_model_adapter, test_get_set_params) {
  // An adapted model should contain both higher level parameters,
  // and the sub model parameters.
  using SqrExp = SquaredExponential<EuclideanDistance>;
  TestAdaptedModel<SqrExp> model;
  auto sqr_exp_params = SqrExp().get_params();
  auto params = model.get_params();
  // Make sure all the sub model params are in the adapted params
  for (const auto &pair : sqr_exp_params) {
    std::cout << "key: " << pair.first << std::endl;
    test_get_set(model, pair.first);
  }
  std::cout << "key: center" << std::endl;
  // And the higher level parameter.
  test_get_set(model, "center");
  std::cout << "complete" << std::endl;
};

TEST(test_model_adapter, test_fit) {
  auto adpated_dataset = make_adapted_toy_linear_data();

  using CovFunc = decltype(toy_covariance_function());
  TestAdaptedModel<CovFunc> adapted_model;

  adapted_model.fit(adpated_dataset);
  const auto adapted_pred = adapted_model.predict(adpated_dataset.features).joint();

  GaussianProcessRegression<double, CovFunc> model;

  auto dataset = make_toy_linear_data();
  model.fit(dataset);
  const auto pred = model.predict(dataset.features).joint();

  EXPECT_EQ(adapted_pred, pred);
}

//TEST(test_model_adapter, test_ransac_fit) {
//  auto dataset = make_adapted_toy_linear_data();
//  TestAdaptedModel<decltype(toy_covariance_function())> adapted_model;
//  adapted_model.fit(dataset);
//  const auto adapted_pred = adapted_model.predict(dataset.features);
//
//  const auto fold_indexer = leave_one_out_indexer(dataset);
//
//  double inlier_threshold = 1.;
//  std::size_t min_inliers = 2;
//  std::size_t min_features = 3;
//  std::size_t max_iterations = 20;
//
//  auto ransac_model = adapted_model->ransac_model(inlier_threshold, min_inliers,
//                                                  min_features, max_iterations);
//
//  EvaluationMetric<JointDistribution> nll =
//      evaluation_metrics::negative_log_likelihood;
//
//  dataset.targets.mean[3] = 400.;
//  dataset.targets.mean[5] = -300.;
//
//  ransac_model->fit(dataset);
//
//  const auto scores =
//      cross_validated_scores(nll, dataset, fold_indexer, ransac_model.get());
//
//  // Here we use the original model_ptr and make sure it also was fit after
//  // we called `model_ptr->ransac_model.fit()`
//  const auto in_sample_preds =
//      adapted_model->template predict<Eigen::VectorXd>(dataset.features);
//
//  // Here we make sure the leave one out likelihoods for inliers are all
//  // reasonable, and for the known outliers we assert the likelihood is
//  // really really really small.
//  for (Eigen::Index i = 0; i < scores.size(); i++) {
//    double in_sample_error = fabs(in_sample_preds[i] - dataset.targets.mean[i]);
//    if (i == 3 || i == 5) {
//      EXPECT_GE(scores[i], 1.e5);
//      EXPECT_GE(in_sample_error, 100.);
//    } else {
//      EXPECT_LE(scores[i], 0.);
//      EXPECT_LE(in_sample_error, 0.1);
//    }
//  }
//}
}
