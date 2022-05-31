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

#include <Eigen/Sparse>
#include <Eigen/SPQRSupport>

namespace albatross {

// https://gitlab.com/libeigen/eigen/-/issues/1706
//
// Unfortunately we inherit the default definition of `Eigen::Index`
// from any code that calls this.
using SparseMatrix = Eigen::SparseMatrix<double>;
static_assert(sizeof(SparseMatrix::StorageIndex) <= sizeof(SparseMatrix::Index),
              "The type `SparseMatrix::StorageIndex` has a bigger range than `SparseMatrix::Index`; this will cause `bad_alloc()` in sparse matrix storage operations!");

using SparseQR = Eigen::SPQR<SparseMatrix>;
static_assert(sizeof(SparseQR::StorageIndex) <= sizeof(SparseMatrix::Index),
              "The type `SparseQR::StorageIndex` has a bigger range than `SparseMatrix::Index`; this will cause `bad_alloc()` in sparse matrix storage operations!");


using SparsePermutationMatrix =
  Eigen::Matrix<long int, Eigen::Dynamic, Eigen::Dynamic>;
using DensePermutationMatrix =
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;

inline Eigen::MatrixXd
get_R(const Eigen::ColPivHouseholderQR<Eigen::MatrixXd> &qr) {
  // Unfortunately the matrixR() method in Eigen's QR decomposition isn't
  // actually the R matrix, it's tall skinny matrix whose lower trapezoid
  // contains internal data, only the upper triangular portion is useful
  return qr.matrixR()
      .topRows(qr.matrixR().cols())
      .template triangularView<Eigen::Upper>();
}

inline DensePermutationMatrix get_column_permutation_indices(
    const SparseQR &qr) {
  return SparsePermutationMatrix(qr.colsPermutation().indices()).cast<int>();
}

inline Eigen::MatrixXd get_R(const SparseQR &qr){
  // TODO(MP): is this necessary?
  //
  // https://eigen.tuxfamily.org/dox/classEigen_1_1SparseQR.html#a7764a1f00c0d83f6423f81a4a40d5f8c
  // const Eigen::SparseMatrix<double, Eigen::RowMajor> Rrow = qr.matrixR();
  // const Eigen::SparseMatrix<double> R = Rrow;

  auto R = qr.matrixR();
  return Eigen::MatrixXd(R.topRows(R.cols()))
      .template triangularView<Eigen::Upper>();
}

// This is the formula given in the SPQR user guide [1] for
// calculation of the default pivot threshold.  The `scale_factor`
// defaults to the SPQR value but provides a convenient knob for
// adjusting the pivot policy in a scale-invariant and
// condition-invariant way.
//
// [1] https://raw.githubusercontent.com/DrTimothyAldenDavis/SuiteSparse/master/SPQR/Doc/spqr_user_guide.pdf
inline double calc_pivot_threshold(const SparseMatrix &m,
                                   double scale_factor = 20.0) {
  double max2Norm = 0.0;
  for (int j = 0; j < m.cols(); j++) {
    max2Norm = std::max(max2Norm, m.col(j).norm());
  }
  if (max2Norm < Eigen::NumTraits<double>::epsilon()) {
    max2Norm = 1;
  }
  return scale_factor * static_cast<double>(m.rows() + m.cols()) * max2Norm *
         Eigen::NumTraits<double>::epsilon();
};

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
    std::vector<std::size_t> sorted_idx(vector.size());
    std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
    std::sort(sorted_idx.begin(), sorted_idx.end(),
              [&vector](std::size_t ii, std::size_t jj) {
                return fabs(vector[ii]) > fabs(vector[jj]);
              });

    for (std::size_t j = 0; j < sorted_idx.size(); ++j) {
      double coef = vector[sorted_idx[j]];
      if (fabs(coef) > print_if_above) {
        (*stream) << "    " << std::setw(12) << coef << "   "
                  << features[sorted_idx[j]] << std::endl;
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
