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

namespace linear_combination {

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

// to_linear_combination(x) lets you construct a LinearCombination
// without explicitly knowing the type of x. If x is already a
// LinearCombination it gets returned without modification.

template <typename X> inline auto to_linear_combination(const X &x) {
  return LinearCombination<X>({x}, Eigen::VectorXd::Ones(1));
}

template <typename X>
inline auto to_linear_combination(const LinearCombination<X> &x) {
  return x;
}
} // namespace linear_combination

template <typename Derived, typename X>
inline auto operator*(const Eigen::SparseMatrixBase<Derived> &matrix,
                      const std::vector<X> &features) {
  std::vector<LinearCombination<X>> output(cast::to_size(matrix.rows()));
  std::vector<std::vector<double>> output_coefs(cast::to_size(matrix.rows()));

  for (Eigen::Index k = 0; k < matrix.outerSize(); ++k) {
    for (typename Derived::InnerIterator it(matrix.derived(), k); it; ++it) {
      const auto row = cast::to_size(it.row());
      const auto col = cast::to_size(it.col());
      output[row].values.push_back(features[col]);
      output_coefs[row].push_back(it.value());
    }
  }

  for (std::size_t i = 0; i < output_coefs.size(); ++i) {
    if (output_coefs[i].size() > 0) {
      output[i].coefficients = Eigen::Map<Eigen::VectorXd>(
          &output_coefs[i][0], cast::to_index(output_coefs[i].size()));
    }
  }

  return output;
}

template <typename Derived, typename X>
inline auto operator*(const Eigen::SparseMatrixBase<Derived> &matrix,
                      const std::vector<LinearCombination<X>> &features) {
  std::vector<LinearCombination<X>> output(cast::to_size(matrix.rows()));

  std::vector<std::vector<double>> output_coefs(cast::to_size(matrix.rows()));

  for (Eigen::Index k = 0; k < matrix.outerSize(); ++k) {
    for (typename Derived::InnerIterator it(matrix.derived(), k); it; ++it) {
      const auto row = cast::to_size(it.row());
      const auto col = cast::to_size(it.col());
      const auto &combo = features[col];
      for (Eigen::Index i = 0; i < combo.coefficients.size(); ++i) {
        const auto si = cast::to_size(i);
        output[row].values.push_back(combo.values[si]);
        output_coefs[row].push_back(it.value() * combo.coefficients[i]);
      }
    }
  }

  for (std::size_t i = 0; i < output_coefs.size(); ++i) {
    if (output_coefs[i].size() > 0) {
      output[i].coefficients = Eigen::Map<Eigen::VectorXd>(
          &output_coefs[i][0],
          static_cast<Eigen::Index>(output_coefs[i].size()));
    }
  }

  return output;
}

template <typename Derived, typename X>
inline auto operator*(const Eigen::MatrixBase<Derived> &matrix,
                      const std::vector<X> &features) {
  Eigen::SparseMatrix<typename Derived::Scalar> sparse = matrix.sparseView();
  return sparse * features;
}

} // namespace albatross

#endif /* ALBATROSS_COVARIANCE_FUNCTIONS_LINEAR_COMBINATION_HPP_ */
