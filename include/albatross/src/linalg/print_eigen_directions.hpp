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

#ifndef ALBATROSS_SRC_LINALG_PRINT_EIGEN_DIRECTIONS_HPP
#define ALBATROSS_SRC_LINALG_PRINT_EIGEN_DIRECTIONS_HPP

namespace albatross {

namespace details {

constexpr double DEFAULT_EIGEN_VALUE_PRINT_THRESHOLD = 1e-3;

template <typename FeatureType, typename Comparator>
inline void _print_eigen_directions(const Eigen::MatrixXd &matrix,
                                    const std::vector<FeatureType> &features,
                                    std::size_t count, Comparator comp,
                                    double print_if_above,
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

  std::ios_base::fmtflags f(stream->flags());

  (*stream) << std::scientific;

  (*stream) << "MIN: " << values.minCoeff() << std::endl;
  (*stream) << "MAX: " << values.maxCoeff() << std::endl;

  stream->flags(f);

  for (std::size_t i = 0; i < count; ++i) {
    const double value = std::get<0>(values_and_vectors[i]);
    const auto vector = std::get<1>(values_and_vectors[i]);

    (*stream) << std::scientific;
    (*stream) << "eigen value: " << value << std::endl;
    stream->flags(f);

    // Sort the indices from largest to smallest coef
    std::vector<Eigen::Index> sorted_idx(cast::to_size(vector.size()));
    std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
    std::sort(sorted_idx.begin(), sorted_idx.end(),
              [&vector](Eigen::Index ii, Eigen::Index jj) {
                return fabs(vector[ii]) > fabs(vector[jj]);
              });

    for (std::size_t j = 0; j < sorted_idx.size(); ++j) {
      double coef = vector[sorted_idx[j]];
      if (fabs(coef) > print_if_above) {
        (*stream) << "    " << std::setw(12) << coef << "   "
                  << features[cast::to_size(sorted_idx[j])] << std::endl;
      }
    }
  }
}

} // namespace details

template <typename FeatureType>
inline void print_small_eigen_directions(
    const Eigen::MatrixXd &matrix, const std::vector<FeatureType> &features,
    std::size_t count,
    double print_if_above = details::DEFAULT_EIGEN_VALUE_PRINT_THRESHOLD,
    std::ostream *stream = &std::cout) {
  const auto ascending = [&](const double &a, const double &b) -> bool {
    return a < b;
  };

  details::_print_eigen_directions(matrix, features, count, ascending,
                                   print_if_above, stream);
}

template <typename FeatureType>
inline void print_large_eigen_directions(
    const Eigen::MatrixXd &matrix, const std::vector<FeatureType> &features,
    std::size_t count,
    double print_if_above = details::DEFAULT_EIGEN_VALUE_PRINT_THRESHOLD,
    std::ostream *stream = &std::cout) {
  const auto decending = [&](const double &a, const double &b) -> bool {
    return a > b;
  };

  details::_print_eigen_directions(matrix, features, count, decending,
                                   print_if_above, stream);
}

} // namespace albatross

#endif // ALBATROSS_SRC_LINALG_PRINT_EIGEN_DIRECTIONS_HPP
