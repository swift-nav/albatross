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

inline Eigen::MatrixXd
get_R(const Eigen::ColPivHouseholderQR<Eigen::MatrixXd> &qr) {
  // Unfortunately the matrixR() method in Eigen's QR decomposition isn't
  // actually the R matrix, it's tall skinny matrix whose lower trapezoid
  // contains internal data, only the upper triangular portion is useful
  return qr.matrixR()
      .topRows(qr.matrixR().cols())
      .template triangularView<Eigen::Upper>();
}

/*
 * Computes R^-T P^T rhs given R and P from a ColPivHouseholderQR decomposition.
 */
template <typename MatrixType>
inline Eigen::MatrixXd sqrt_solve(const Eigen::MatrixXd &R,
                                  const Eigen::VectorXi &permutation_indices,
                                  const MatrixType &rhs) {

  Eigen::MatrixXd sqrt(rhs.rows(), rhs.cols());
  for (Eigen::Index i = 0; i < permutation_indices.size(); ++i) {
    sqrt.row(i) = rhs.row(permutation_indices.coeff(i));
  }
  sqrt = R.template triangularView<Eigen::Upper>().transpose().solve(sqrt);
  return sqrt;
}

template <typename MatrixType>
inline Eigen::MatrixXd
sqrt_solve(const Eigen::ColPivHouseholderQR<Eigen::MatrixXd> &qr,
           const MatrixType &rhs) {
  const Eigen::MatrixXd R = get_R(qr);
  return sqrt_solve(R, qr.colsPermutation().indices(), rhs);
}

namespace details {

constexpr double DEFAULT_EIGEN_VALUE_PRINT_THRESHOLD = 1e-3;

/*
 * When building models it can be helpful to understand what the
 * model is good (or bad) at predicting. This provides one way of
 * understanding a model's predictions by looking at the predicted
 * covariance. So, for example, if you have a model and make a
 * prediction,
 *
 *   pred = model.predict(features).joint();
 *
 * you could then call:
 *
 *   print_large_eigen_directions(pred.covariance, features, 1);
 *
 * this will compute the eigen decomposition of the covariance,
 *
 *   V D V^T = pred.covariance
 *
 * Here D is diagonal and holds the eigen values and V is orthonormal
 * holding the eigen vectors. In this function we print out the
 * largest (or smallest depending on `comp`) eigen value along with
 * any of the features which have non-zero values in the corresponding
 * eigen vector, something like:
 *
 *   eigen value: D[0]
 *       V[0, 0]  features[0]
 *       V[1, 0]  features[1]
 *       ...
 *
 * Note that any V[i, j] which is smaller in magnitude than `print_if_above`
 * will be ignored in the interest of brevity.
 */
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

#endif /* ALBATROSS_UTILS_LINALG_UTILS_HPP_ */
