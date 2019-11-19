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

namespace albatross {

template <typename ImplType> class LeastSquares;

template <typename ImplType> struct Fit<LeastSquares<ImplType>> {
  Eigen::VectorXd coefs;

  bool operator==(const Fit &other) const { return (coefs == other.coefs); }
};

inline bool all_same_value(const Eigen::VectorXd &x) {
  if (x.size() == 0) {
    return true;
  } else {
    const double first = x[0];
    for (Eigen::Index i = 0; i < x.size(); ++i) {
      if (x[i] != first) {
        return false;
      }
    }
    return true;
  }
}

/*
 * This model supports a family of models which consist of
 * first creating a design matrix, A, then solving least squares.  Ie,
 *
 *   min_x |y - Ax|_2^2
 *
 * The FeatureType in this case is a single row from the design matrix.
 */
template <typename ImplType> class LeastSquares : public ModelBase<ImplType> {
public:
  using FitType = Fit<LeastSquares<ImplType>>;

  std::string get_name() const { return "least_squares"; }

  FitType _fit_impl(const std::vector<Eigen::VectorXd> &features,
                    const MarginalDistribution &targets) const {
    // The way this is currently implemented we assume all targets have the same
    // variance (or zero variance).
    assert(all_same_value(targets.covariance.diagonal()));
    // Build the design matrix
    int m = static_cast<int>(features.size());
    int n = static_cast<int>(features[0].size());
    Eigen::MatrixXd A(m, n);
    for (int i = 0; i < m; i++) {
      A.row(i) = features[static_cast<std::size_t>(i)];
    }

    FitType model_fit = {least_squares_solver(A, targets.mean)};
    return model_fit;
  }

  Eigen::VectorXd _predict_impl(const std::vector<Eigen::VectorXd> &features,
                                const FitType &least_squares_fit,
                                PredictTypeIdentity<Eigen::VectorXd>) const {
    std::size_t n = features.size();
    Eigen::VectorXd mean(n);
    for (std::size_t i = 0; i < n; i++) {
      mean(static_cast<Eigen::Index>(i)) =
          features[i].dot(least_squares_fit.coefs);
    }
    return Eigen::VectorXd(mean);
  }

  /*
   * This lets you customize the least squares approach if need be,
   * default uses the QR decomposition.
   */
  Eigen::VectorXd least_squares_solver(const Eigen::MatrixXd &A,
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
class LinearRegression : public LeastSquares<LinearRegression> {

public:
  using Base = LeastSquares<LinearRegression>;

  std::string get_name() const { return "linear_regression"; }

  Eigen::VectorXd convert_feature(const double &f) const {
    Eigen::VectorXd converted(2);
    converted << 1., f;
    return converted;
  }

  std::vector<Eigen::VectorXd>
  convert_features(const std::vector<double> &features) const {
    std::vector<Eigen::VectorXd> output;
    for (const auto &f : features) {
      output.emplace_back(convert_feature(f));
    }
    return output;
  }

  Base::FitType _fit_impl(const std::vector<double> &features,
                          const MarginalDistribution &targets) const {
    return Base::_fit_impl(convert_features(features), targets);
  }

  Eigen::VectorXd _predict_impl(const std::vector<double> &features,
                                const Base::FitType &least_squares_fit,
                                PredictTypeIdentity<Eigen::VectorXd>) const {
    return Base::_predict_impl(convert_features(features), least_squares_fit,
                               PredictTypeIdentity<Eigen::VectorXd>());
  }
};
} // namespace albatross

#endif
