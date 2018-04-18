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

#ifndef ALBATROSS_MODELS_LEAST_SQUARES_H
#define ALBATROSS_MODELS_LEAST_SQUARES_H

#include <gtest/gtest.h>
#include <cmath>
#include <Eigen/Dense>
#include <iostream>
#include <random>
#include "core/serialize.h"
#include "core/model_adapter.h"

namespace albatross {

struct LeastSquaresFit {
  Eigen::VectorXd coefs;
};

/*
 * This model supports a family of RegressionModels which consist of
 * first creating a design matrix, A, then solving least squares.  Ie,
 *
 *   min_x |y - Ax|_2^2
 *
 * The FeatureType in this case is a single row from the design matrix.
 */
class LeastSquaresRegression : public SerializableRegressionModel<Eigen::VectorXd,
                                                                  LeastSquaresFit> {
 public:
  LeastSquaresRegression() {};
  std::string get_name() const override { return "least_squares"; };

 protected:

  /*
   * This lets you customize the least squares approach if need be,
   * default uses the QR decomposition.
   */
  virtual Eigen::VectorXd least_squares_solver(const Eigen::MatrixXd &A,
                                               const Eigen::VectorXd &b) const {
    return A.colPivHouseholderQr().solve(b);
  }

  LeastSquaresFit serializable_fit_(const std::vector<Eigen::VectorXd> &features,
                                     const Eigen::VectorXd &targets) const override {
    // Build the design matrix
    int m = static_cast<int>(features.size());
    int n = static_cast<int>(features[0].size());
    Eigen::MatrixXd A(m, n);
    for (int i = 0; i < m; i++) {
      A.row(i) = features[static_cast<std::size_t>(i)];
    }
    // Solve for the coefficients using the QR decomposition.
    LeastSquaresFit model_fit = {least_squares_solver(A, targets)};
    return model_fit;
  }

  PredictionDistribution predict_(const std::vector<Eigen::VectorXd> &features) const {
    int n = static_cast<s32>(features.size());
    Eigen::VectorXd predictions(n);
    for (s32 i = 0; i < n; i++) {
      predictions(i) = features[static_cast<std::size_t>(i)].dot(this->model_fit_.coefs);
    }

    return PredictionDistribution(predictions);
  }
};


/*
 * Creates a least squares problem by building a design matrix that looks like:
 *
 *   A_i = [1 x]
 *
 * Setup like this the resulting fit will represent an offset and slope.
 */
class LinearRegression : public AdaptedRegressionModel<double, LeastSquaresRegression> {

  std::string get_name() const override { return "linear_regression"; };

  Eigen::VectorXd convert_feature(const double& x) const {
    Eigen::VectorXd converted(2);
    converted << 1., x;
    return converted;
  }

};

}

#endif
