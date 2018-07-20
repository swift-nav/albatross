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

using SqrExp = CovarianceFunction<SquaredExponential<EuclideanDistance>>;

using TestBaseModel = GaussianProcessRegression<Eigen::VectorXd, SqrExp>;

using TestAdaptedModelBase = AdaptedRegressionModel<double, TestBaseModel>;

class TestAdaptedModel : public TestAdaptedModelBase {
public:
  TestAdaptedModel() { this->params_["center"] = 0.; };

  std::string get_name() const override { return "test_adapted"; };

  const Eigen::VectorXd convert_feature(const double &x) const override {
    Eigen::VectorXd converted(2);
    converted << 1., (x - this->get_param_value("center"));
    return converted;
  }

  /*
   * save/load methods are inherited from the SerializableRegressionModel,
   * but by defining them here and explicitly showing the inheritence
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
} // namespace albatross

void test_get_set(albatross::RegressionModel<double> &model,
                  const std::string &key) {
  // Make sure a key exists, then modify it and make sure it
  // takes on the new value.
  const auto orig = model.get_param_value(key);
  model.set_param(key, orig + 1.);
  EXPECT_EQ(model.get_params().at(key), orig + 1.);
}

TEST(test_model_adapter, test_get_set_params) {
  // An adapted model should contain both higher level parameters,
  // and the sub model parameters.
  auto model = albatross::TestAdaptedModel();
  auto sqr_exp_params = albatross::SqrExp().get_params();
  auto params = model.get_params();
  // Make sure all the sub model params are in the adapted params
  for (const auto &pair : sqr_exp_params) {
    test_get_set(model, pair.first);
  }
  // And the higher level parameter.
  test_get_set(model, "center");
};
