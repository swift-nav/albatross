/*
 * Copyright (C) 2020 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef INCLUDE_ALBATROSS_SRC_MODELS_INTERPOLATE_HPP_
#define INCLUDE_ALBATROSS_SRC_MODELS_INTERPOLATE_HPP_

namespace albatross {

auto interpolation_cov_func() {
  SquaredExponential<EuclideanDistance> sqr_exp;
  IndependentNoise<double> noise;
  return sqr_exp + measurement_only(noise);
}

using InterpolationFunction = decltype(interpolation_cov_func());


/*
 * Generic Gaussian Process Implementation.
 */
class GaussianProcessInterpolator
    : public GaussianProcessBase<InterpolationFunction, ZeroMean,
                                 GaussianProcessInterpolator> {
public:
  using Base =
      GaussianProcessBase<InterpolationFunction, ZeroMean,
                          GaussianProcessInterpolator>;

};

using GPInterpolatorFitType = typename fit_type<GaussianProcessInterpolator, double>::type;

template <>
class Prediction<GaussianProcessInterpolator, double, GPInterpolatorFitType> {

public:
  Prediction(const GaussianProcessInterpolator &model,
             const GPInterpolatorFitType &fit,
             const std::vector<double> &features)
      : model_(model), fit_(fit), features_(features) {}

  Prediction(GaussianProcessInterpolator &&model, GPInterpolatorFitType &&fit,
             const std::vector<double> &features)
      : model_(std::move(model)), fit_(std::move(fit)), features_(features) {}

  // Mean
  Eigen::VectorXd mean() const {
    return MeanPredictor()._mean(model_, fit_, features_);
  }

  Eigen::VectorXd derivative() const {

    std::vector<Derivative<double>> derivative_features;
    for (const auto &f : features_) {
      derivative_features.emplace_back(f);
    }

    return MeanPredictor()._mean(model_, fit_, derivative_features);
  }

  Eigen::VectorXd second_derivative() const {

    std::vector<SecondDerivative<double>> derivative_features;
    for (const auto &f : features_) {
      derivative_features.emplace_back(f);
    }
    return MeanPredictor()._mean(model_, fit_, derivative_features);
  }

  const GaussianProcessInterpolator model_;
  const GPInterpolatorFitType fit_;
  const std::vector<double> features_;
};

}
#endif /* INCLUDE_ALBATROSS_SRC_MODELS_INTERPOLATE_HPP_ */
