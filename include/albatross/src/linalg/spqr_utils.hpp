/*
 * Copyright (C) 2022 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef ALBATROSS_SRC_LINALG_SPQR_UTILS_HPP
#define ALBATROSS_SRC_LINALG_SPQR_UTILS_HPP

namespace albatross {

using SparseMatrix = Eigen::SparseMatrix<double>;

using SPQR = Eigen::SPQR<SparseMatrix>;

using SparsePermutationMatrix =
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic,
                             SPQR::StorageIndex>;

inline Eigen::MatrixXd get_R(const SPQR &qr) {
  return qr.matrixR()
      .topLeftCorner(qr.cols(), qr.cols())
      .template triangularView<Eigen::Upper>();
}

template <typename MatrixType>
inline Eigen::MatrixXd sqrt_solve(const SPQR &qr, const MatrixType &rhs) {
  return sqrt_solve(get_R(qr), qr.colsPermutation().indices(), rhs);
}

// Matrices with any dimension smaller than this will use a special
// "small problem" pivot threshold.
constexpr Eigen::Index kMinSparsePivotSize = 300;

// This coefficient replaces the hardcoded factor of 20 in the default
// SPQR pivot threshold policy.
constexpr double kSPQRPivotCoefficient = 0.1;

// Set the pivot threshold of the solver based on the problem to be
// solved.
//
// The default SPQR pivot threshold calculation is oriented towards
// problems considerably larger than your typical Albatross use case.
// Depending on your problem, it may be advantageous to use this
// function to choose a smaller threshold than the default and trade a
// minimal speed reduction for accuracy improvements.
template <typename MatrixType>
inline void SPQR_set_pivot_threshold(SPQR *spqr, const MatrixType &m,
                                     Eigen::Index min_sparse_pivot_size,
                                     double spqr_pivot_coefficient) {
  if (m.rows() < min_sparse_pivot_size || m.cols() < min_sparse_pivot_size) {
    // For small matrices, just use a really tight threshold for pivot
    // detection to reduce the inaccuracy of the sparse method.
    spqr->setPivotThreshold(1e-15);
  }
  // For larger matrices, use the default SuiteSparseQR policy to
  // choose the pivot threshold, but accept a user-controlled
  // coefficient.
  double max2Norm = 0.0;
  for (Eigen::Index j = 0; j < m.cols(); j++) {
    max2Norm = Eigen::numext::maxi(max2Norm, m.col(j).norm());
  }
  if (max2Norm == 0.) {
    max2Norm = 1.;
  }
  spqr->setPivotThreshold(spqr_pivot_coefficient *
                          cast::to_double(m.rows() + m.cols()) * max2Norm *
                          Eigen::NumTraits<double>::epsilon());
}

template <typename MatrixType>
inline void SPQR_set_pivot_threshold(SPQR *spqr, const MatrixType &m,
                                     Eigen::Index min_sparse_pivot_size) {
  SPQR_set_pivot_threshold(spqr, m, min_sparse_pivot_size,
                           kSPQRPivotCoefficient);
}

template <typename MatrixType>
inline void SPQR_set_pivot_threshold(
    SPQR *spqr, const MatrixType &m,
    double spqr_pivot_coefficient = kSPQRPivotCoefficient) {
  SPQR_set_pivot_threshold(spqr, m, kMinSparsePivotSize,
                           spqr_pivot_coefficient);
}

inline void SPQR_set_threads(SPQR *spqr, std::size_t num_threads) {
  spqr->cholmodCommon()->SPQR_nthreads = static_cast<int>(num_threads);
}

inline void SPQR_set_threads(SPQR *spqr, const ThreadPool *pool) {
  if (pool != nullptr) {
    SPQR_set_threads(spqr, pool->thread_count());
  }
}

inline std::unique_ptr<SPQR> SPQR_create(const ThreadPool *pool = nullptr) {
  auto spqr = std::make_unique<SPQR>();
  spqr->setSPQROrdering(SPQR_ORDERING_COLAMD);
  SPQR_set_threads(spqr.get(), pool);
  return spqr;
}

inline std::unique_ptr<SPQR> SPQR_create(const SparseMatrix &A,
                                         const ThreadPool *pool = nullptr) {
  auto spqr = SPQR_create(pool);
  SPQR_set_pivot_threshold(spqr.get(), A);
  spqr->compute(A);
  return spqr;
}

} // namespace albatross

#endif // ALBATROSS_SRC_LINALG_SPQR_UTILS_HPP
