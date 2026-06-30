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

#ifndef ALBATROSS_EIGEN_SERIALIZABLE_LDLT_H
#define ALBATROSS_EIGEN_SERIALIZABLE_LDLT_H

namespace Eigen {

// See LDLT.h in Eigen for a detailed description of the decomposition
class SerializableLDLT : public LDLT<MatrixXd, Lower> {
public:
  using RealScalar = double;
  using Scalar = double;
  using MatrixType = MatrixXd;

  SerializableLDLT() : LDLT<MatrixXd, Lower>() {}

  SerializableLDLT(const MatrixXd &x) : LDLT<MatrixXd, Lower>(x.ldlt()) {}

  SerializableLDLT(const LDLT<MatrixXd, Lower> &ldlt)
      // Can we get around copying here?
      : LDLT<MatrixXd, Lower>(ldlt) {}

  SerializableLDLT(const LDLT<MatrixXd, Lower> &&ldlt)
      : LDLT<MatrixXd, Lower>(std::move(ldlt)) {}

  bool is_positive_definite() const { return this->vectorD().minCoeff() > 0.; }

  LDLT<MatrixXd, Lower>::TranspositionType &mutable_transpositions() {
    return this->m_transpositions;
  }

  LDLT<MatrixXd, Lower>::MatrixType &mutable_matrix() { return this->m_matrix; }

  const LDLT<MatrixXd, Lower>::MatrixType &matrix() const {
    return this->m_matrix;
  }

  bool &mutable_is_initialized() { return this->m_isInitialized; }

  bool is_initialized() const { return this->m_isInitialized; }

  double l1_norm() const {
    ALBATROSS_ASSERT(is_initialized() && "Must initialize first!");
    return this->m_l1_norm;
  }

  /*
   * Computes the inverse of the square root of the diagonal, D^{-1/2}
   */
  Eigen::DiagonalMatrix<double, Eigen::Dynamic> diagonal_sqrt_inverse() const {
    Eigen::VectorXd thresholded_diag_sqrt_inverse(this->vectorD());
    for (Eigen::Index i = 0; i < thresholded_diag_sqrt_inverse.size(); ++i) {
      if (thresholded_diag_sqrt_inverse[i] > 0.) {
        thresholded_diag_sqrt_inverse[i] =
            1. / std::sqrt(thresholded_diag_sqrt_inverse[i]);
      } else {
        thresholded_diag_sqrt_inverse[i] = 0.;
      }
    }
    return thresholded_diag_sqrt_inverse.asDiagonal();
  }

  /*
   * Computes the square root of the diagonal, D^{1/2}
   */
  Eigen::DiagonalMatrix<double, Eigen::Dynamic> diagonal_sqrt() const {
    Eigen::VectorXd thresholded_diag = this->vectorD();
    for (Eigen::Index i = 0; i < thresholded_diag.size(); ++i) {
      if (thresholded_diag[i] > 0.) {
        thresholded_diag[i] = std::sqrt(thresholded_diag[i]);
      } else {
        thresholded_diag[i] = 0.;
      }
    }
    return thresholded_diag.asDiagonal();
  }

  /*
   * Computes the product of the square root of A with rhs,
   *   P^T L D^{1/2} rhs
   */
  template <class Rhs>
  Eigen::MatrixXd sqrt_product(const MatrixBase<Rhs> &rhs) const {
    return this->transpositionsP().transpose() *
           (this->matrixL() * (diagonal_sqrt() * rhs));
  }

  /*
   * Computes the product of the square root of A with rhs,
   *   D^{-1/2} L^-1 P rhs
   */
  template <class Rhs> Eigen::MatrixXd sqrt_solve(const Rhs &rhs) const {
    return diagonal_sqrt_inverse() *
           this->matrixL().solve(this->transpositionsP() *
                                 Eigen::MatrixXd(rhs));
  }

  Eigen::MatrixXd sqrt_transpose() const {
    return this->diagonal_sqrt() * (this->transpositionsP().transpose() *
                                    this->matrixL().toDenseMatrix())
                                       .transpose();
  }

  /*
   * Computes the product of the square root of A with rhs,
   *   P^T L^-T D^{-1/2} rhs
   */
  template <class _Scalar, int _Rows, int _Cols>
  Eigen::Matrix<_Scalar, _Rows, _Cols>
  sqrt_transpose_solve(const Eigen::Matrix<_Scalar, _Rows, _Cols> &rhs) const {
    return this->transpositionsP().transpose() *
           (this->matrixL().transpose().solve(diagonal_sqrt_inverse() * rhs));
  }

  double log_determinant() const {
    // The log determinant can be found by starting with the full decomposition
    //   log(|A|) = log(|P| |L| |D| |L| |P^T|)
    // then realizing that P and L are both unit matrices so we get:
    //   log(|A|) = log(|D|)
    //            = sum(log(diag(D)))
    return this->vectorD().array().log().sum();
  }

  std::vector<Eigen::MatrixXd>
  inverse_blocks(const std::vector<std::vector<std::size_t>> &blocks,
                 ThreadPool *pool = nullptr) const {
    /*
     * The LDLT decomposition is stored such that,
     *
     *     A = P^T LDL^T P
     *
     * We first need to compute the inverse of the cholesky, R^-1 such that
     *
     *     A^{-1} = R^-T R^-1
     *     R^{-1} = D^1/2 L^-1 P
     *
     * we can then pull out sub blocks of A^{-1} by dot products of
     * corresponding
     * columns of R^{-1}.
     */
    Eigen::Index n = this->matrixL().rows();

    // Compute the pre-permutation factor D^{-1/2} L^{-1}: solve L X = I, then
    // row-scale by D^{-1/2}. Skipping the column permutation P that would
    // finish R^{-1} avoids the O(n^2) eager row-permutation of the identity;
    // we recover correctness below by translating each block's column indices
    // through pi.
    Eigen::MatrixXd pre_perm = Eigen::MatrixXd::Identity(n, n);
    this->matrixL().solveInPlace(pre_perm);
    pre_perm = diagonal_sqrt_inverse() * pre_perm;

    ALBATROSS_ASSERT(!pre_perm.hasNaN());

    // pi[k] = source column in pre_perm that lives at output column k of the
    // fully-permuted inverse cholesky factor R^{-1} = D^{-1/2} L^{-1} P.
    const Eigen::VectorXi pi =
        (this->transpositionsP().transpose() *
         Eigen::VectorXi::LinSpaced(n, 0, static_cast<int>(n) - 1))
            .eval();

    return albatross::apply(
        blocks,
        [&](const auto &block_indices) -> Eigen::MatrixXd {
          std::vector<std::size_t> permuted_indices(block_indices.size());
          for (std::size_t i = 0; i < block_indices.size(); ++i) {
            permuted_indices[i] = static_cast<std::size_t>(
                pi(static_cast<Eigen::Index>(block_indices[i])));
          }
          Eigen::MatrixXd sub_matrix =
              albatross::subset_cols(pre_perm, permuted_indices);
          return sub_matrix.transpose().lazyProduct(sub_matrix);
        },
        pool);
  }

  /*
   * The diagonal of the inverse of the matrix this LDLT
   * decomposition represents in O(n^2) operations.
   *
   * A = P^T L D L^T P, so the inverse cholesky factor is
   *     R^{-1} = D^{-1/2} L^{-1} P
   * and A^{-1} = R^{-T} R^{-1}. The diagonal of A^{-1} is the squared
   * column norm of R^{-1}.
   */
  Eigen::VectorXd inverse_diagonal() const {
    const Eigen::Index n = this->rows();
    // Solve L X = I to get X = L^{-1}; then scale rows by D^{-1/2}.
    // We deliberately skip the column permutation by P that the full inverse
    // cholesky would have, because column squared-norms of (D^{-1/2} L^{-1} P)
    // are just a permutation of the squared-norms of (D^{-1/2} L^{-1}).
    // Permuting the n-vector at the end is O(n) instead of an extra n^2 pass.
    Eigen::MatrixXd inverse_cholesky = Eigen::MatrixXd::Identity(n, n);
    this->matrixL().solveInPlace(inverse_cholesky);
    inverse_cholesky = diagonal_sqrt_inverse() * inverse_cholesky;
    const Eigen::VectorXd pre_perm_norms =
        inverse_cholesky.colwise().squaredNorm().transpose();
    return this->transpositionsP().transpose() * pre_perm_norms;
  }

  bool operator==(const SerializableLDLT &rhs) const {
    // Make sure the two lower triangles are the same and that
    // any permutations are identical.
    if (!this->m_isInitialized && !rhs.m_isInitialized) {
      return true;
    }
    auto this_lower =
        MatrixXd(MatrixXd(this->matrixLDLT()).triangularView<Eigen::Lower>());
    auto rhs_lower =
        MatrixXd(MatrixXd(rhs.matrixLDLT()).triangularView<Eigen::Lower>());

    return (this->m_isInitialized == rhs.m_isInitialized &&
            this_lower == rhs_lower &&
            this->transpositionsP().indices() ==
                rhs.transpositionsP().indices());
  }
};

} // namespace Eigen

#endif
