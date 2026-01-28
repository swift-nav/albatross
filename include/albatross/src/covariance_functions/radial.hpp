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

#ifndef ALBATROSS_COVARIANCE_FUNCTIONS_RADIAL_H
#define ALBATROSS_COVARIANCE_FUNCTIONS_RADIAL_H

constexpr double default_length_scale = 100000.;
constexpr double default_radial_sigma = 10.;

namespace albatross {

constexpr std::size_t MAX_NEWTON_ITERATIONS = 50;
constexpr double MAX_LENGTH_SCALE_RATIO = 1e7;
constexpr double MIN_LENGTH_SCALE_RATIO = 1e-7;

// Template version for mixed precision (parameters stay as double for precision)
template <typename Scalar = double>
inline Scalar squared_exponential_covariance(Scalar distance,
                                             Scalar length_scale,
                                             Scalar sigma = Scalar(1.)) {
  if (length_scale <= Scalar(0.)) {
    return Scalar(0.);
  }
  ALBATROSS_ASSERT(distance >= Scalar(0.));
  return sigma * sigma * std::exp(-std::pow(distance / length_scale, 2));
}

// Backward compatibility: non-template version
inline double squared_exponential_covariance(double distance,
                                             double length_scale,
                                             double sigma) {
  return squared_exponential_covariance<double>(distance, length_scale, sigma);
}

template <typename RadialFunction>
inline double process_noise_equivalent(RadialFunction func, double distance) {
  // to get the increase in standard deviation over a given distance we can
  // ask for the predictive variance for one point given another point
  // separated by a distance `d`.
  //
  //   [f_0,      ~ N(|0 , |k(0), k(d)| )
  //    f_d]          |0   |k(d), k(0)|
  //
  //   VAR[f_d|f_0] = k(0) - k(d) k(d) / k(0)
  //   STD[f_d|f_0] = sqrt(k(0) - k(d)^2/ k(0))
  const double k0 = func(0);
  const double kd = func(distance);
  return sqrt(k0 - kd * kd / k0);
}

namespace detail {

inline bool valid_args_for_derive_length_scale(double reference_distance,
                                               double prior_sigma,
                                               double std_dev_increase) {
  ALBATROSS_ASSERT(reference_distance > 0.);
  return (std_dev_increase > 0. && prior_sigma > 0. &&
          std_dev_increase < prior_sigma);
}

inline double fallback_length_scale_for_invalid_args(double reference_distance,
                                                     double prior_sigma,
                                                     double std_dev_increase) {
  if (std_dev_increase <= 0.) {
    // an increase of 0. means an extremely large length scale and would
    // lead to a divide by zero below, so we early return.
    return MAX_LENGTH_SCALE_RATIO * reference_distance;
  }
  if (prior_sigma <= 0.) {
    // with values of zero it doesn't matter what the length scale is.
    return MAX_LENGTH_SCALE_RATIO * reference_distance;
  }
  const double ratio = std_dev_increase / prior_sigma;
  assert(ratio > 0.);
  if (ratio >= 1.) {
    // there's no way for the std deviation to exceed the prior std dev if
    // this is specified just assume a tiny length scale;
    return MIN_LENGTH_SCALE_RATIO * reference_distance;
  }
  if (ratio <= 0.) {
    // in order to keep the standard deviation from increasing we need
    // the longest length scale possible.
    return MAX_LENGTH_SCALE_RATIO * reference_distance;
  }
  // all edge cases should have been handled
  assert(false);
  return NAN;
}
} // namespace detail

inline double derive_squared_exponential_length_scale(double reference_distance,
                                                      double prior_sigma,
                                                      double std_dev_increase) {
  if (!detail::valid_args_for_derive_length_scale(
          reference_distance, prior_sigma, std_dev_increase)) {
    return detail::fallback_length_scale_for_invalid_args(
        reference_distance, prior_sigma, std_dev_increase);
  }
  // See process_noise_equivalent for an intro.
  //
  // for the squared exponential function this means an increase in
  // standard deviation, d_sd, can be written:
  //
  //   d_sd = sqrt(sigma^2 - sigma^2 exp[-(d / length_scale)^2]^2)
  //        = sigma * sqrt(1 - exp[-(d / length_scale)^2]^2)
  //        = sigma * sqrt(1 - exp[-2 * (d / length_scale)^2])
  //
  // solving for length_scale gives us
  //
  //   d_sd / sigma = sqrt(1 - exp[-2 * (d / length_scale)^2])
  //
  //   (d_sd / sigma)^2 = 1 - exp[-2 * (d / length_scale)^2]
  //
  //   exp[-2 * (d / length_scale)^2] = 1 - (d_sd / sigma)^2
  //
  //   -2 (d / length_scale)^2 = log[1 - (d_sd / sigma)^2]
  //
  //   (d / length_scale) = sqrt(-1/2 log[1 - (d_sd / sigma)^2])
  //
  //   length_scale = d / sqrt(-1/2 log[1 - (d_sd / sigma)^2])
  const double ratio = std_dev_increase / prior_sigma;
  assert(ratio > 0.);
  assert(ratio < 1.);
  return sqrt(2.0) * reference_distance / sqrt(-log(1. - ratio * ratio));
}

/*
 * SquaredExponential distance
 *    covariance(d) = sigma^2 exp(-(d/length_scale)^2)
 */
template <class DistanceMetricType>
class SquaredExponential
    : public CovarianceFunction<SquaredExponential<DistanceMetricType>> {
public:
  // The SquaredExponential radial function is not positive definite
  // when the distance is an angular (or great circle) distance.
  // See:
  // Gneiting, Strictly and non-strictly positive definite functions on spheres
  static_assert(
      !std::is_base_of<AngularDistance, DistanceMetricType>::value,
      "SquaredExponential covariance with AngularDistance is not PSD.");

  ALBATROSS_DECLARE_PARAMS(squared_exponential_length_scale,
                           sigma_squared_exponential)

  SquaredExponential(double length_scale_ = default_length_scale,
                     double sigma_squared_exponential_ = default_radial_sigma)
      : distance_metric_() {
    squared_exponential_length_scale = {length_scale_, PositivePrior()};
    sigma_squared_exponential = {sigma_squared_exponential_,
                                 NonNegativePrior()};
  }

  std::string name() const {
    return "squared_exponential[" + this->distance_metric_.get_name() + "]";
  }

  std::vector<double> _ssr_impl(const std::vector<double> &xs) const {
    double min = *std::min_element(xs.begin(), xs.end());
    double max = *std::max_element(xs.begin(), xs.end());

    double range = max - min;
    // using 1/10th of the length scale should result in grids with
    // one percent decorrelation between them. exp(- 0.1**2)
    double n = ceil(10 * range / squared_exponential_length_scale.value);
    n = std::max(n, 3.);
    return linspace(min, max, safe_cast_to_size_t(n));
  }

  double derive_length_scale(double reference_distance, double sigma,
                             double std_dev_increase) const {
    return derive_squared_exponential_length_scale(reference_distance, sigma,
                                                   std_dev_increase);
  }

  // This operator is only defined when the distance metric is also defined.
  template <typename X,
            typename std::enable_if<
                has_call_operator<DistanceMetricType, X &, X &>::value,
                int>::type = 0>
  double _call_impl(const X &x, const X &y) const {
    double distance = this->distance_metric_(x, y);
    return squared_exponential_covariance(
        distance, squared_exponential_length_scale.value,
        sigma_squared_exponential.value);
  }

  DistanceMetricType distance_metric_;
};

// Template version for mixed precision
template <typename Scalar = double>
inline Scalar exponential_covariance(Scalar distance, Scalar length_scale,
                                     Scalar sigma = Scalar(1.)) {
  if (length_scale <= Scalar(0.)) {
    return Scalar(0.);
  }
  ALBATROSS_ASSERT(distance >= Scalar(0.));
  return sigma * sigma * std::exp(-std::fabs(distance / length_scale));
}

// Backward compatibility
inline double exponential_covariance(double distance, double length_scale,
                                     double sigma) {
  return exponential_covariance<double>(distance, length_scale, sigma);
}

inline double derive_exponential_length_scale(double reference_distance,
                                              double prior_sigma,
                                              double std_dev_increase) {
  if (!detail::valid_args_for_derive_length_scale(
          reference_distance, prior_sigma, std_dev_increase)) {
    return detail::fallback_length_scale_for_invalid_args(
        reference_distance, prior_sigma, std_dev_increase);
  }
  // See process_noise_equivalent for an intro
  //
  // For the exponential function the equations vary slightly,
  //
  //   d_sd = sqrt(sigma^2 - sigma^2 exp[-|distance / length_scale|]^2)
  //        = sigma * sqrt(1 - exp[-|distance / length_scale|]^2)
  //        = sigma * sqrt(1 - exp[-2 * |distance / length_scale|])
  //
  // solving for length_scale gives us
  //
  //   d_sd / sigma = sqrt(1 - exp[-2 * |distance / length_scale|])
  //
  //   (d_sd / sigma)^2 = 1 - exp[-2 * |distance / length_scale|]
  //
  //   exp[-2 * |distance / length_scale|] = 1 - (d_sd / sigma)^2
  //
  //   -2 |distance / length_scale| = log[1 - (d_sd / sigma)^2]
  //
  //   |distance / length_scale| = -1/2 log[1 - (d_sd / sigma)^2]
  //
  //   length_scale = -2 * distance / log[1 - (d_sd / sigma)^2]
  const double ratio = std_dev_increase / prior_sigma;
  assert(ratio > 0.);
  assert(ratio < 1.);
  return -2.0 * reference_distance / log(1. - ratio * ratio);
}

/*
 * Exponential distance
 *    covariance(d) = sigma^2 exp(-|d|/length_scale)
 */
template <class DistanceMetricType>
class Exponential : public CovarianceFunction<Exponential<DistanceMetricType>> {
public:
  ALBATROSS_DECLARE_PARAMS(exponential_length_scale, sigma_exponential)

  Exponential(double length_scale_ = default_length_scale,
              double sigma_exponential_ = default_radial_sigma)
      : distance_metric_() {
    exponential_length_scale = {length_scale_, PositivePrior()};
    sigma_exponential = {sigma_exponential_, NonNegativePrior()};
  }

  std::string name() const {
    return "exponential[" + this->distance_metric_.get_name() + "]";
  }

  ~Exponential() {}

  std::vector<double> _ssr_impl(const std::vector<double> &xs) const {
    double min = *std::min_element(xs.begin(), xs.end());
    double max = *std::max_element(xs.begin(), xs.end());

    double range = max - min;
    // using 1/20th of the length scale should result in grids with
    // five percent decorrelation between them. exp(-0.05)
    double n = ceil(20 * range / exponential_length_scale.value);
    n = std::max(n, 3.);
    return linspace(min, max, safe_cast_to_size_t(n));
  }

  double derive_length_scale(double reference_distance, double sigma,
                             double std_dev_increase) const {
    return derive_exponential_length_scale(reference_distance, sigma,
                                           std_dev_increase);
  }

  // This operator is only defined when the distance metric is also defined.
  template <typename X,
            typename std::enable_if<
                has_call_operator<DistanceMetricType, X &, X &>::value,
                int>::type = 0>
  double _call_impl(const X &x, const X &y) const {
    double distance = this->distance_metric_(x, y);
    return exponential_covariance(distance, exponential_length_scale.value,
                                  sigma_exponential.value);
  }

  DistanceMetricType distance_metric_;
};

// Template version for mixed precision
template <typename Scalar = double>
inline Scalar matern_32_covariance(Scalar distance, Scalar length_scale,
                                   Scalar sigma = Scalar(1.)) {
  if (length_scale <= Scalar(0.)) {
    return Scalar(0.);
  }
  assert(distance >= Scalar(0.));
  const Scalar sqrt_3_d = std::sqrt(Scalar(3.)) * distance / length_scale;
  return sigma * sigma * (Scalar(1.) + sqrt_3_d) * std::exp(-sqrt_3_d);
}

// Backward compatibility
inline double matern_32_covariance(double distance, double length_scale,
                                   double sigma) {
  return matern_32_covariance<double>(distance, length_scale, sigma);
}

namespace detail {

template <typename Func, typename Grad>
inline double newton_solve(double guess, double target, Func func, Grad grad,
                           double lower_bound, double upper_bound,
                           double tolerance = 1e-12) {
  // NB: While technically this could be a generalized solver, it hasn't been
  // tested beyond the use case of solving for length scales, reuse with
  // caution.
  for (std::size_t i = 0; i < MAX_NEWTON_ITERATIONS; ++i) {
    const double f_i = func(guess);
    const double error = target - f_i;
    if (!std::isfinite(error)) {
      break;
    }
    const double delta = error / grad(guess);

    if (fabs(error) < tolerance) {
      break;
    }

    if (guess - delta <= lower_bound) {
      guess = 0.5 * (guess + lower_bound);
    } else if (guess - delta >= upper_bound) {
      guess = 0.5 * (guess + upper_bound);
    } else {
      guess -= delta;
    }
    guess = std::min(upper_bound, std::max(lower_bound, guess));
  }
  return guess;
}

template <typename Func, typename Grad>
inline double derive_length_scale(double reference_distance, double prior_sigma,
                                  double std_dev_increase, Func func,
                                  Grad grad) {
  if (!detail::valid_args_for_derive_length_scale(
          reference_distance, prior_sigma, std_dev_increase)) {
    return detail::fallback_length_scale_for_invalid_args(
        reference_distance, prior_sigma, std_dev_increase);
  }
  // func and grad should accept a single argument "ratio" which is the
  // length scale expressed as multiples of the reference distance.
  //
  //    ratio = length_scale / reference_distance
  //
  // The goal behind the reformulation is to be able to keep the solver
  // stable. For example, a length scale of 1e-16 meters might
  // be perfectly reasonable on an atomic scale, while 1e32 meters might
  // be reasonable on an astronmical scale, so working directly with
  // length scale could require exploring a very large range. Instead
  // using the ratio allows a user to pick a reasonable reference
  // distance and keep the domain searched by this back solver smaller.
  static_assert(is_invocable_with_result<Func, double, double>::value,
                "func should take a single double and return the covariance");
  static_assert(is_invocable_with_result<Grad, double, double>::value,
                "grad should take a single double and return the covariance");

  // We run the newton solver using the log increase in standard
  // deviation to improve numerical stability.
  auto log_f_eval = [&func, &prior_sigma](double ratio) {
    // with ratio = ell / reference_distance
    // here we assume func(ratio) == cov(reference_distance, ell)
    const double cov = func(ratio);
    if (cov * cov >= 1) {
      return log(1e-16);
    }
    const double log_f = log(prior_sigma) + 0.5 * log(1 - cov * cov);
    return log_f;
  };

  auto log_g_eval = [&func, &grad](double ratio) {
    const double cov = func(ratio);
    const double denom = (1 - cov * cov);
    assert(denom > 0);
    return grad(ratio) * cov / denom;
  };

  const double log_target = log(std_dev_increase);
  const double max_increase = log_f_eval(MIN_LENGTH_SCALE_RATIO);
  if (max_increase <= log_target) {
    return MIN_LENGTH_SCALE_RATIO * reference_distance;
  }
  const double min_increase = log_f_eval(MAX_LENGTH_SCALE_RATIO);
  if (min_increase >= log_target) {
    return MAX_LENGTH_SCALE_RATIO * reference_distance;
  }

  // linearly interpolate between log of scales as a coarse guess
  const double alpha =
      (max_increase - log_target) / (max_increase - min_increase);
  const double guess =
      exp(log(MIN_LENGTH_SCALE_RATIO) +
          alpha * (log(MAX_LENGTH_SCALE_RATIO) - log(MIN_LENGTH_SCALE_RATIO)));

  const double solution =
      newton_solve(guess, log_target, log_f_eval, log_g_eval,
                   MIN_LENGTH_SCALE_RATIO, MAX_LENGTH_SCALE_RATIO);
  return solution * reference_distance;
}

} // namespace detail

inline double derive_matern_32_length_scale(double reference_distance,
                                            double prior_sigma,
                                            double std_dev_increase) {
  // Note that an alternative method would be to write the solution
  // in terms of the LambertW function and then use an implementation
  // like this one: https://github.com/DarkoVeberic/LambertW
  auto func = [](double ratio) { return matern_32_covariance(1., ratio, 1.); };

  auto grad = [](double ratio) {
    return sqrt(3) * (1 + sqrt(3) / ratio) * exp(-sqrt(3) / ratio) /
               pow(ratio, 2) -
           sqrt(3) * exp(-sqrt(3) / ratio) / pow(ratio, 2);
  };

  return detail::derive_length_scale(reference_distance, prior_sigma,
                                     std_dev_increase, func, grad);
}

template <class DistanceMetricType>
class Matern32 : public CovarianceFunction<Matern32<DistanceMetricType>> {
public:
  // The Matern nu = 3/2 radial function is not positive definite
  // when the distance is an angular (or great circle) distance.
  static_assert(!std::is_base_of<AngularDistance, DistanceMetricType>::value,
                "Matern32 covariance with AngularDistance is not PSD.");

  ALBATROSS_DECLARE_PARAMS(matern_32_length_scale, sigma_matern_32)

  Matern32(double length_scale_ = default_length_scale,
           double sigma_matern_32_ = default_radial_sigma)
      : distance_metric_() {
    matern_32_length_scale = {length_scale_, PositivePrior()};
    sigma_matern_32 = {sigma_matern_32_, NonNegativePrior()};
  }

  std::string name() const {
    return "matern_32[" + this->distance_metric_.get_name() + "]";
  }

  double derive_length_scale(double reference_distance, double sigma,
                             double std_dev_increase) const {
    return derive_matern_32_length_scale(reference_distance, sigma,
                                         std_dev_increase);
  }

  template <typename X,
            typename std::enable_if<
                has_call_operator<DistanceMetricType, X &, X &>::value,
                int>::type = 0>
  double _call_impl(const X &x, const X &y) const {
    double distance = this->distance_metric_(x, y);
    return matern_32_covariance(distance, matern_32_length_scale.value,
                                sigma_matern_32.value);
  }

  DistanceMetricType distance_metric_;
};

// Template version for mixed precision
template <typename Scalar = double>
inline Scalar matern_52_covariance(Scalar distance, Scalar length_scale,
                                   Scalar sigma = Scalar(1.)) {
  if (length_scale <= Scalar(0.)) {
    return Scalar(0.);
  }
  assert(distance >= Scalar(0.));
  const Scalar sqrt_5_d = std::sqrt(Scalar(5.)) * distance / length_scale;
  return sigma * sigma * (Scalar(1.) + sqrt_5_d + sqrt_5_d * sqrt_5_d / Scalar(3.)) *
         std::exp(-sqrt_5_d);
}

// Backward compatibility
inline double matern_52_covariance(double distance, double length_scale,
                                   double sigma) {
  return matern_52_covariance<double>(distance, length_scale, sigma);
}

inline double derive_matern_52_length_scale(double reference_distance,
                                            double prior_sigma,
                                            double std_dev_increase) {
  // Note that an alternative method would be to write the solution
  // in terms of the LambertW function and then use an implementation
  // like this one: https://github.com/DarkoVeberic/LambertW
  auto func = [](double ratio) { return matern_52_covariance(1., ratio, 1.); };

  auto grad = [](double ratio) {
    return (-sqrt(5) / pow(ratio, 2) - 10. / 3. / pow(ratio, 3)) *
               exp(-sqrt(5) / ratio) +
           sqrt(5) * (1 + sqrt(5) / ratio + 10. / 6. / pow(ratio, 2)) *
               exp(-sqrt(5) / ratio) / pow(ratio, 2);
  };

  return detail::derive_length_scale(reference_distance, prior_sigma,
                                     std_dev_increase, func, grad);
}

template <class DistanceMetricType>
class Matern52 : public CovarianceFunction<Matern52<DistanceMetricType>> {
public:
  // The Matern nu = 5/2 radial function is not positive definite
  // when the distance is an angular (or great circle) distance.
  static_assert(!std::is_base_of<AngularDistance, DistanceMetricType>::value,
                "Matern52 covariance with AngularDistance is not PSD.");

  ALBATROSS_DECLARE_PARAMS(matern_52_length_scale, sigma_matern_52)

  Matern52(double length_scale_ = default_length_scale,
           double sigma_matern_52_ = default_radial_sigma)
      : distance_metric_() {
    matern_52_length_scale = {length_scale_, PositivePrior()};
    sigma_matern_52 = {sigma_matern_52_, NonNegativePrior()};
  }

  std::string name() const {
    return "matern_52[" + this->distance_metric_.get_name() + "]";
  }

  double derive_length_scale(double reference_distance, double sigma,
                             double std_dev_increase) const {
    return derive_matern_52_length_scale(reference_distance, sigma,
                                         std_dev_increase);
  }

  template <typename X,
            typename std::enable_if<
                has_call_operator<DistanceMetricType, X &, X &>::value,
                int>::type = 0>
  double _call_impl(const X &x, const X &y) const {
    double distance = this->distance_metric_(x, y);
    return matern_52_covariance(distance, matern_52_length_scale.value,
                                sigma_matern_52.value);
  }

  DistanceMetricType distance_metric_;
};

} // namespace albatross
#endif
