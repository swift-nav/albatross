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
#include <cmath>
#include <Eigen/Dense>
#include <iostream>
#include <random>
#include "evaluate.h"

namespace albatross {

class LinearRegression : public RegressionModel<double> {
 public:
  LinearRegression() : params_(){};
  std::string get_name() const { return "linear_regression"; };

 private:
  // builds the map from int to value
  void fit_(const std::vector<double> &features,
            const Eigen::VectorXd &targets) {
    (void)targets;
    s32 n = static_cast<s32>(features.size());

    Eigen::VectorXd x(n);
    for (s32 i = 0; i < n; i++) {
      x[i] = features[static_cast<std::size_t>(i)];
    }

    Eigen::MatrixXd A(n, 2);
    A << Eigen::VectorXd::Ones(n), x;

    // use the normal equations to solve for the params
    Eigen::VectorXd rhs = A.transpose() * targets;
    params_ = (A.transpose() * A).ldlt().solve(rhs);
  }

  // looks up the prediction in the map
  PredictionDistribution predict_(const std::vector<double> &features) const {
    s32 n = static_cast<s32>(features.size());
    Eigen::VectorXd predictions(n);

    Eigen::VectorXd x(n);
    for (s32 i = 0; i < n; i++) {
      x[i] = features[static_cast<std::size_t>(i)];
    }

    Eigen::MatrixXd A(n, 2);
    A << Eigen::VectorXd::Ones(n), x;

    return PredictionDistribution(A * params_);
  }

  Eigen::VectorXd params_;
};

class LinearModelTest : public ::testing::Test {
 public:
  LinearModelTest() : model_ptr_(), dataset_({}, {}) {
    double a = 5.;
    double b = 1.;
    double sigma = 0.1;

    std::random_device rd{};
    std::mt19937 gen{rd()};
    gen.seed(3);
    std::normal_distribution<> d{0., sigma};

    s32 n = 10;
    std::vector<double> features(static_cast<std::size_t>(n));
    Eigen::VectorXd targets(n);

    std::map<s32, s32> hist{};
    for (s32 i = 0; i < n; i++) {
      features[static_cast<std::size_t>(i)] = static_cast<double>(i);
      targets[i] = a + b * features[static_cast<std::size_t>(i)] + d(gen);
    }

    model_ptr_ = std::make_unique<LinearRegression>();
    dataset_ = RegressionDataset<double>(features, targets);
  };

  std::unique_ptr<RegressionModel<double>> model_ptr_;
  RegressionDataset<double> dataset_;
};

TEST_F(LinearModelTest, test_leave_one_out) {
  PredictionDistribution preds = model_ptr_->fit_and_predict(
      dataset_.features, dataset_.targets, dataset_.features);
  double in_sample_rmse = root_mean_square_error(preds, dataset_.targets);

  const auto folds = leave_one_out(dataset_);
  Eigen::VectorXd rmses =
      cross_validated_scores(folds, root_mean_square_error, model_ptr_.get());
  double out_of_sample_rmse = rmses.mean();

  // Make sure the RMSE computed doing leave one out cross validation is larger
  // than the in sample version.  This should always be true as the in sample
  // version has already seen the values we're trying to predict.
  EXPECT_LT(in_sample_rmse, out_of_sample_rmse);
}

// Group values by interval, but return keys that once sorted won't be
// in order
std::string group_by_interval(double x) {
  if (x <= 3) {
    return "2";
  } else if (x <= 6) {
    return "3";
  } else {
    return "1";
  }
}

bool is_monotonic_increasing(Eigen::VectorXd &x) {
  for (s32 i = 0; i < static_cast<s32>(x.size()) - 1; i++) {
    if (x[i + 1] - x[i] <= 0.) {
      return false;
    }
  }
  return true;
}

TEST_F(LinearModelTest, test_cross_validated_predict) {
  const auto folds = leave_one_group_out<double>(dataset_, group_by_interval);

  PredictionDistribution preds =
      cross_validated_predict(folds, model_ptr_.get());

  // Make sure the group cross validation resulted in folds that
  // are out of order
  EXPECT_TRUE(folds[0].name == "1");
  // And that cross_validate_predict put them back in order.
  EXPECT_TRUE(is_monotonic_increasing(preds.mean));
}

TEST_F(LinearModelTest, test_leave_one_group_out) {
  const auto folds = leave_one_group_out<double>(dataset_, group_by_interval);
  Eigen::VectorXd rmses =
      cross_validated_scores(folds, root_mean_square_error, model_ptr_.get());

  // Make sure we get a single RMSE for each of the three groups.
  EXPECT_EQ(rmses.size(), 3);
}
}
