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

#ifndef ALBATROSS_UTILS_LINALG_UTILS_HPP_
#define ALBATROSS_UTILS_LINALG_UTILS_HPP_

namespace albatross {

namespace details {

template <typename FeatureType, typename Comparator>
inline void _print_eigen_directions(const Eigen::MatrixXd &matrix,
                                    const std::vector<FeatureType> &features,
                                    std::size_t count, Comparator comp,
                                    std::ostream *stream) {
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> ev(matrix);
  const Eigen::VectorXd values = ev.eigenvalues().real();
  const Eigen::MatrixXd vectors = ev.eigenvectors();

  using ValueAndVector = std::tuple<double, Eigen::VectorXd>;
  std::vector<ValueAndVector> values_and_vectors;

  for (Eigen::Index i = 0; i < values.size(); ++i) {
    values_and_vectors.emplace_back(values[i], vectors.col(i));
  }

  std::sort(values_and_vectors.begin(), values_and_vectors.end(),
            [&](const ValueAndVector &a, const ValueAndVector &b) -> bool {
              return comp(std::get<0>(a), std::get<0>(b));
            });

  (*stream) << "MIN: " << values.minCoeff() << std::endl;
  (*stream) << "MAX: " << values.maxCoeff() << std::endl;

  for (std::size_t i = 0; i < count; ++i) {
    const double value = std::get<0>(values_and_vectors[i]);
    const auto vector = std::get<1>(values_and_vectors[i]);
    (*stream) << "eigen value: " << value << std::endl;
    for (Eigen::Index j = 0; j < vector.size(); ++j) {
      double coef = vector[j];
      if (fabs(coef) > 1e-3) {
        (*stream) << "    " << std::setw(12) << coef << "   " << features[j]
                  << std::endl;
      }
    }
  }
}

} // namespace details

template <typename FeatureType>
inline void print_small_eigen_directions(
    const Eigen::MatrixXd &matrix, const std::vector<FeatureType> &features,
    std::size_t count, std::ostream *stream = &std::cout) {
  const auto ascending = [&](const double &a, const double &b) -> bool {
    return a < b;
  };

  details::_print_eigen_directions(matrix, features, count, ascending, stream);
}

template <typename FeatureType>
inline void print_large_eigen_directions(
    const Eigen::MatrixXd &matrix, const std::vector<FeatureType> &features,
    std::size_t count, std::ostream *stream = &std::cout) {
  const auto decending = [&](const double &a, const double &b) -> bool {
    return a > b;
  };

  details::_print_eigen_directions(matrix, features, count, decending, stream);
}

} // namespace albatross

#endif /* ALBATROSS_UTILS_LINALG_UTILS_HPP_ */
