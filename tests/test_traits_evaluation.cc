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

#include <albatross/Evaluation>
#include <gtest/gtest.h>

namespace albatross {

struct X {};
struct Y {};

/*
 * Predict Traits
 */
class HasMeanPredictImpl {
public:
  template <typename FeatureType, typename GroupKey>
  std::map<GroupKey, Eigen::VectorXd>
  cross_validated_predictions(const RegressionDataset<FeatureType> &,
                              const GroupIndexer<GroupKey> &,
                              PredictTypeIdentity<Eigen::VectorXd>) const {
    return std::map<GroupKey, Eigen::VectorXd>();
  }
};

class HasMarginalPredictImpl {
public:
  template <typename FeatureType, typename GroupKey>
  std::map<GroupKey, MarginalDistribution>
  cross_validated_predictions(const RegressionDataset<FeatureType> &,
                              const GroupIndexer<GroupKey> &,
                              PredictTypeIdentity<MarginalDistribution>) const {
    return std::map<GroupKey, MarginalDistribution>();
  }
};

class HasJointPredictImpl {
public:
  template <typename FeatureType, typename GroupKey>
  std::map<GroupKey, JointDistribution>
  cross_validated_predictions(const RegressionDataset<FeatureType> &,
                              const GroupIndexer<GroupKey> &,
                              PredictTypeIdentity<JointDistribution>) const {
    return std::map<GroupKey, JointDistribution>();
  }
};

class HasAllPredictImpls {
public:
  template <typename GroupKey>
  std::map<GroupKey, Eigen::VectorXd>
  cross_validated_predictions(const RegressionDataset<X> &,
                              const GroupIndexer<GroupKey> &,
                              PredictTypeIdentity<Eigen::VectorXd>) const {
    return std::map<GroupKey, Eigen::VectorXd>();
  }

  template <typename FeatureType, typename GroupKey>
  std::map<GroupKey, MarginalDistribution>
  cross_validated_predictions(const RegressionDataset<FeatureType> &,
                              const GroupIndexer<GroupKey> &,
                              PredictTypeIdentity<MarginalDistribution>) const {
    return std::map<GroupKey, MarginalDistribution>();
  }

  template <typename GroupKey>
  std::map<GroupKey, JointDistribution>
  cross_validated_predictions(const RegressionDataset<X> &,
                              const GroupIndexer<GroupKey> &,
                              PredictTypeIdentity<JointDistribution>) const {
    return std::map<GroupKey, JointDistribution>();
  }
};

struct TestKey {};

TEST(test_traits_core, test_has_cross_validated_predictions) {
  EXPECT_TRUE(bool(has_valid_cv_mean<HasMeanPredictImpl, X, TestKey>::value));
  EXPECT_FALSE(
      bool(has_valid_cv_marginal<HasMeanPredictImpl, X, TestKey>::value));
  EXPECT_FALSE(bool(has_valid_cv_joint<HasMeanPredictImpl, X, TestKey>::value));

  EXPECT_FALSE(
      bool(has_valid_cv_mean<HasMarginalPredictImpl, X, TestKey>::value));
  EXPECT_TRUE(
      bool(has_valid_cv_marginal<HasMarginalPredictImpl, X, TestKey>::value));
  EXPECT_FALSE(
      bool(has_valid_cv_joint<HasMarginalPredictImpl, X, TestKey>::value));

  EXPECT_FALSE(bool(has_valid_cv_mean<HasJointPredictImpl, X, TestKey>::value));
  EXPECT_FALSE(
      bool(has_valid_cv_marginal<HasJointPredictImpl, X, TestKey>::value));
  EXPECT_TRUE(bool(has_valid_cv_joint<HasJointPredictImpl, X, TestKey>::value));

  EXPECT_TRUE(bool(has_valid_cv_mean<HasAllPredictImpls, X, TestKey>::value));
  EXPECT_FALSE(bool(has_valid_cv_mean<HasAllPredictImpls, Y, TestKey>::value));
  EXPECT_TRUE(
      bool(has_valid_cv_marginal<HasAllPredictImpls, X, TestKey>::value));
  EXPECT_TRUE(
      bool(has_valid_cv_marginal<HasAllPredictImpls, Y, TestKey>::value));
  EXPECT_TRUE(bool(has_valid_cv_joint<HasAllPredictImpls, X, TestKey>::value));
  EXPECT_FALSE(bool(has_valid_cv_joint<HasAllPredictImpls, Y, TestKey>::value));
}

} // namespace albatross
