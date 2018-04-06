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

#ifndef ALBATROSS_MODELS_LINEAR_REGRESSION_H
#define ALBATROSS_MODELS_LINEAR_REGRESSION_H

/*
 * Here we define a LinearRegression model which is less because
 * it'll be super useful on its own (albatross is probably overkill
 * if all you need is linear regression) but serves as an example of
 * extending the `RegressionModel` for non Gaussian process models.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <Eigen/Dense>
#include <iostream>
#include <random>

namespace albatross {

struct LinearRegressionCoefs {
  Eigen::VectorXd coefs;
};

class LinearRegression : public RegressionModel<Eigen::VectorXd,
                                                LinearRegressionCoefs> {
 public:
  LinearRegression() {};
  std::string get_name() const { return "linear_regression"; };

 private:

  LinearRegressionCoefs fit_(const std::vector<Eigen::VectorXd> &features,
            const Eigen::VectorXd &targets) const override {
    int m = static_cast<int>(features.size());
    int n = static_cast<int>(features[0].size());

    Eigen::MatrixXd A(m, n);
    for (int i = 0; i < m; i++) {
      A.row(i) = features[static_cast<std::size_t>(i)];
    }

    Eigen::VectorXd rhs = A.transpose() * targets;
    LinearRegressionCoefs model_fit;
    model_fit.coefs = (A.transpose() * A).ldlt().solve(rhs);
    return model_fit;
  }

  PredictionDistribution predict_(const std::vector<Eigen::VectorXd> &features) const {
    int n = static_cast<s32>(features.size());
    Eigen::VectorXd predictions(n);

    for (s32 i = 0; i < n; i++) {
      predictions(i) = features[static_cast<std::size_t>(i)].dot(model_fit_->coefs);
    }

    return PredictionDistribution(predictions);
  }

  Eigen::VectorXd coefs_;
};

}

#endif
