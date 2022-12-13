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

#ifndef ALBATROSS_COVARIANCE_FUNCTIONS_LINEAR_COMBINATION_HPP_
#define ALBATROSS_COVARIANCE_FUNCTIONS_LINEAR_COMBINATION_HPP_

namespace albatross {

template <typename X> struct LinearCombination {

  static_assert(
      !is_measurement<X>::value,
      "Putting a Measurement type inside a LinearCombination will lead to "
      "unexpected behavior due to the ordering of the DefaultCaller");

  LinearCombination(){};

  LinearCombination(const std::vector<X> &values_)
      : values(values_),
        coefficients(Eigen::VectorXd::Ones(cast::to_index(values_.size()))){};

  LinearCombination(const std::vector<X> &values_,
                    const Eigen::VectorXd &coefficients_)
      : values(values_), coefficients(coefficients_) {
    ALBATROSS_ASSERT(values_.size() == cast::to_size(coefficients_.size()));
  };

  bool operator==(const LinearCombination &other) const {
    return values == other.values && coefficients == other.coefficients;
  }

  std::vector<X> values;
  Eigen::VectorXd coefficients;
};

template <typename X> inline auto sum(const X &x, const X &y) {
  Eigen::Vector2d coefs;
  coefs << 1., 1.;
  return LinearCombination<X>({x, y}, coefs);
}

template <typename X, typename Y> inline auto sum(const X &x, const Y &y) {
  variant<X, Y> vx = x;
  variant<X, Y> vy = y;
  return sum(vx, vy);
}

template <typename X> inline auto mean(const std::vector<X> &xs) {
  const auto n = cast::to_index(xs.size());
  Eigen::VectorXd coefs(n);
  coefs.fill(1. / cast::to_double(n));
  return LinearCombination<X>(xs, coefs);
}

template <typename X> inline auto difference(const X &x, const X &y) {
  Eigen::Vector2d coefs;
  coefs << 1., -1.;
  return LinearCombination<X>({x, y}, coefs);
}

template <typename X, typename Y>
inline auto difference(const X &x, const Y &y) {
  variant<X, Y> vx = x;
  variant<X, Y> vy = y;
  return difference(vx, vy);
}

} // namespace albatross

#endif /* ALBATROSS_COVARIANCE_FUNCTIONS_LINEAR_COMBINATION_HPP_ */
