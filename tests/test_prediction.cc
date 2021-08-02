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

#include <albatross/Core>
#include <gtest/gtest.h>

namespace albatross {

struct X {};

class MeanOnlyModel : public ModelBase<MeanOnlyModel> {
public:
  Fit<MeanOnlyModel> _fit_impl(const std::vector<X> &,
                               const MarginalDistribution &) const {
    return {};
  }

  Eigen::VectorXd _predict_impl(const std::vector<X> &features,
                                const Fit<MeanOnlyModel> &,
                                PredictTypeIdentity<Eigen::VectorXd>) const {
    return Eigen::VectorXd::Zero(static_cast<Eigen::Index>(features.size()));
  }
};

TEST(test_prediction, test_mean_only) {
  MeanOnlyModel m;
  std::vector<X> xs = {{}, {}};
  const auto zeros =
      Eigen::VectorXd::Zero(static_cast<Eigen::Index>(xs.size()));
  MarginalDistribution targets(zeros);

  const auto fit_model = m.fit(xs, targets);
  const auto prediction = fit_model.predict(xs);
  auto mean = prediction.mean();
  EXPECT_TRUE(bool(std::is_same<Eigen::VectorXd, decltype(mean)>::value));

  std::vector<X> empty = {};
  EXPECT_EQ(fit_model.predict(empty).mean().size(), 0);
}

class MarginalOnlyModel : public ModelBase<MarginalOnlyModel> {
public:
  Fit<MarginalOnlyModel> _fit_impl(const std::vector<X> &,
                                   const MarginalDistribution &) const {
    return {};
  }

  MarginalDistribution
  _predict_impl(const std::vector<X> &features, const Fit<MarginalOnlyModel> &,
                PredictTypeIdentity<MarginalDistribution>) const {
    auto mean =
        Eigen::VectorXd::Zero(static_cast<Eigen::Index>(features.size()));
    return MarginalDistribution(mean);
  }
};

TEST(test_prediction, test_marginal_only) {
  MarginalOnlyModel m;
  std::vector<X> xs = {{}, {}};
  const auto zeros =
      Eigen::VectorXd::Zero(static_cast<Eigen::Index>(xs.size()));
  MarginalDistribution targets(zeros);

  const auto fit_model = m.fit(xs, targets);
  const auto prediction = fit_model.predict(xs);
  auto mean = prediction.mean();
  EXPECT_TRUE(bool(std::is_same<Eigen::VectorXd, decltype(mean)>::value));

  auto marginal = prediction.marginal();
  EXPECT_TRUE(
      bool(std::is_same<MarginalDistribution, decltype(marginal)>::value));

  std::vector<X> empty = {};
  EXPECT_EQ(fit_model.predict(empty).marginal().size(), 0);
}

class JointOnlyModel : public ModelBase<JointOnlyModel> {
public:
  Fit<JointOnlyModel> _fit_impl(const std::vector<X> &,
                                const MarginalDistribution &) const {
    return {};
  }

  JointDistribution
  _predict_impl(const std::vector<X> &features, const Fit<JointOnlyModel> &,
                PredictTypeIdentity<JointDistribution>) const {
    const Eigen::Index n = static_cast<Eigen::Index>(features.size());
    const auto mean = Eigen::VectorXd::Zero(n);
    const auto covariance = Eigen::MatrixXd::Zero(n, n);
    return JointDistribution(mean, covariance);
  }
};

TEST(test_prediction, test_joint_only) {
  JointOnlyModel m;
  std::vector<X> xs = {{}, {}};
  const auto zeros =
      Eigen::VectorXd::Zero(static_cast<Eigen::Index>(xs.size()));
  MarginalDistribution targets(zeros);

  const auto fit_model = m.fit(xs, targets);
  const auto prediction = fit_model.predict(xs);

  auto mean = prediction.mean();
  EXPECT_TRUE(bool(std::is_same<Eigen::VectorXd, decltype(mean)>::value));

  auto marginal = prediction.marginal();
  EXPECT_TRUE(
      bool(std::is_same<MarginalDistribution, decltype(marginal)>::value));

  auto joint = prediction.joint();
  EXPECT_TRUE(bool(std::is_same<JointDistribution, decltype(joint)>::value));

  std::vector<X> empty = {};
  EXPECT_EQ(fit_model.predict(empty).joint().size(), 0);
}

} // namespace albatross
