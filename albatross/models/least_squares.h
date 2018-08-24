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

#include "core/model_adapter.h"
#include "core/serialize.h"
#include <Eigen/Dense>
#include <cmath>
#include <gtest/gtest.h>
#include <iostream>
#include <random>

namespace albatross {

struct LeastSquaresFit {
  Eigen::VectorXd coefs;

  bool operator==(const LeastSquaresFit &other) const {
    return coefs == other.coefs;
  }

  template <typename Archive> void serialize(Archive &archive) {
    archive(coefs);
  }
};

/*
 * This model supports a family of RegressionModels which consist of
 * first creating a design matrix, A, then solving least squares.  Ie,
 *
 *   min_x |y - Ax|_2^2
 *
 * The FeatureType in this case is a single row from the design matrix.
 */
class LeastSquaresRegression
    : public SerializableRegressionModel<Eigen::VectorXd, LeastSquaresFit> {
public:
  LeastSquaresRegression(){};
  std::string get_name() const override { return "least_squares"; };

  LeastSquaresFit
  serializable_fit_(const std::vector<Eigen::VectorXd> &features,
                    const MarginalDistribution &targets) const override {
    // The way this is currently implemented we assume all targets have the same
    // variance (or zero variance).
    assert(!targets.has_covariance());
    // Build the design matrix
    int m = static_cast<int>(features.size());
    int n = static_cast<int>(features[0].size());
    Eigen::MatrixXd A(m, n);
    for (int i = 0; i < m; i++) {
      A.row(i) = features[static_cast<std::size_t>(i)];
    }
    // Solve for the coefficients using the QR decomposition.
    LeastSquaresFit model_fit = {least_squares_solver(A, targets.mean)};
    return model_fit;
  }

protected:
  Eigen::VectorXd
  predict_mean_(const std::vector<Eigen::VectorXd> &features) const override {
    std::size_t n = features.size();
    Eigen::VectorXd mean(n);
    for (std::size_t i = 0; i < n; i++) {
      mean(static_cast<Eigen::Index>(i)) =
          features[i].dot(this->model_fit_.coefs);
    }
    return mean;
  }

  JointDistribution
  predict_(const std::vector<Eigen::VectorXd> &features) const override {
    return JointDistribution(predict_mean_(features));
  }

  /*
   * This lets you customize the least squares approach if need be,
   * default uses the QR decomposition.
   */
  virtual Eigen::VectorXd least_squares_solver(const Eigen::MatrixXd &A,
                                               const Eigen::VectorXd &b) const {
    return A.colPivHouseholderQr().solve(b);
  }
};

/*
 * Creates a least squares problem by building a design matrix where the
 * i^th row looks like:
 *
 *   A_i = [1 x]
 *
 * Setup like this the resulting least squares solve will represent
 * an offset and slope.
 */
using LinearRegressionBase =
    AdaptedRegressionModel<double, LeastSquaresRegression>;

class LinearRegression : public LinearRegressionBase {

public:
  LinearRegression(){};
  std::string get_name() const override { return "linear_regression"; };

  const Eigen::VectorXd convert_feature(const double &x) const override {
    Eigen::VectorXd converted(2);
    converted << 1., x;
    return converted;
  }

  /*
   * save/load methods are inherited from the SerializableRegressionModel,
   * but by defining them here and explicitly showing the inheritence
   * through the use of `base_class` we can make use of cereal's
   * polymorphic serialization.
   */
  template <class Archive> void save(Archive &archive) const {
    archive(cereal::make_nvp("linear_regression",
                             cereal::base_class<LinearRegressionBase>(this)));
  }

  template <class Archive> void load(Archive &archive) {
    archive(cereal::make_nvp("linear_regression",
                             cereal::base_class<LinearRegressionBase>(this)));
  }
};
} // namespace albatross

CEREAL_REGISTER_TYPE(albatross::LinearRegression);

#endif
