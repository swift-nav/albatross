/*
 * Copyright (C) 2019 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "Evaluation"

#include <gtest/gtest.h>

namespace albatross {

struct X {};
struct Y {};

/*
 * Predict Traits
 */
class HasMeanPredictImpl {
public:
  template <typename FeatureType>
  std::vector<Eigen::VectorXd>
  cross_validated_predictions(const RegressionDataset<FeatureType> &,
                              const FoldIndexer &,
                              PredictTypeIdentity<Eigen::VectorXd>) const {
    return {Eigen::VectorXd(0)};
  }
};

class HasMarginalPredictImpl {
public:
  template <typename FeatureType>
  std::vector<MarginalDistribution>
  cross_validated_predictions(const RegressionDataset<FeatureType> &,
                              const FoldIndexer &,
                              PredictTypeIdentity<MarginalDistribution>) const {

    const auto mean = Eigen::VectorXd::Zero(0);
    return {MarginalDistribution(mean)};
  }
};

class HasJointPredictImpl {
public:
  template <typename FeatureType>
  std::vector<JointDistribution>
  cross_validated_predictions(const RegressionDataset<FeatureType> &,
                              const FoldIndexer &,
                              PredictTypeIdentity<JointDistribution>) const {

    const auto mean = Eigen::VectorXd::Zero(0);
    return {JointDistribution(mean)};
  }
};

class HasAllPredictImpls {
public:
  std::vector<Eigen::VectorXd>
  cross_validated_predictions(const RegressionDataset<X> &, const FoldIndexer &,
                              PredictTypeIdentity<Eigen::VectorXd>) const {
    return {Eigen::VectorXd(0)};
  }

  template <typename FeatureType>
  std::vector<MarginalDistribution>
  cross_validated_predictions(const RegressionDataset<FeatureType> &,
                              const FoldIndexer &,
                              PredictTypeIdentity<MarginalDistribution>) const {

    const auto mean = Eigen::VectorXd::Zero(0);
    return {MarginalDistribution(mean)};
  }

  std::vector<JointDistribution>
  cross_validated_predictions(const RegressionDataset<X> &, const FoldIndexer &,
                              PredictTypeIdentity<JointDistribution>) const {

    const auto mean = Eigen::VectorXd::Zero(0);
    return {JointDistribution(mean)};
  }
};

TEST(test_traits_core, test_has_cross_validated_predictions) {
  EXPECT_TRUE(bool(has_valid_cv_mean<HasMeanPredictImpl, X>::value));
  EXPECT_FALSE(bool(has_valid_cv_marginal<HasMeanPredictImpl, X>::value));
  EXPECT_FALSE(bool(has_valid_cv_joint<HasMeanPredictImpl, X>::value));

  EXPECT_FALSE(bool(has_valid_cv_mean<HasMarginalPredictImpl, X>::value));
  EXPECT_TRUE(bool(has_valid_cv_marginal<HasMarginalPredictImpl, X>::value));
  EXPECT_FALSE(bool(has_valid_cv_joint<HasMarginalPredictImpl, X>::value));

  EXPECT_FALSE(bool(has_valid_cv_mean<HasJointPredictImpl, X>::value));
  EXPECT_FALSE(bool(has_valid_cv_marginal<HasJointPredictImpl, X>::value));
  EXPECT_TRUE(bool(has_valid_cv_joint<HasJointPredictImpl, X>::value));

  EXPECT_TRUE(bool(has_valid_cv_mean<HasAllPredictImpls, X>::value));
  EXPECT_FALSE(bool(has_valid_cv_mean<HasAllPredictImpls, Y>::value));
  EXPECT_TRUE(bool(has_valid_cv_marginal<HasAllPredictImpls, X>::value));
  EXPECT_TRUE(bool(has_valid_cv_marginal<HasAllPredictImpls, Y>::value));
  EXPECT_TRUE(bool(has_valid_cv_joint<HasAllPredictImpls, X>::value));
  EXPECT_FALSE(bool(has_valid_cv_joint<HasAllPredictImpls, Y>::value));
}

} // namespace albatross
