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
  SerializableLDLT() : LDLT<MatrixXd, Lower>(){};

  SerializableLDLT(const MatrixXd &x) : LDLT<MatrixXd, Lower>(x.ldlt()){};

  SerializableLDLT(const LDLT<MatrixXd, Lower> &ldlt)
      // Can we get around copying here?
      : LDLT<MatrixXd, Lower>(ldlt){};

  SerializableLDLT(const LDLT<MatrixXd, Lower> &&ldlt)
      : LDLT<MatrixXd, Lower>(std::move(ldlt)){};

  LDLT<MatrixXd, Lower>::TranspositionType &mutable_transpositions() {
    return this->m_transpositions;
  }

  LDLT<MatrixXd, Lower>::MatrixType &mutable_matrix() { return this->m_matrix; }

  LDLT<MatrixXd, Lower>::MatrixType matrix() const { return this->m_matrix; }

  bool &mutable_is_initialized() { return this->m_isInitialized; }

  bool is_initialized() const { return this->m_isInitialized; }

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
  template <class Rhs>
  Eigen::MatrixXd sqrt_solve(const MatrixBase<Rhs> &rhs) const {
    return diagonal_sqrt_inverse() *
           this->matrixL().solve(this->transpositionsP() * rhs);
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
  inverse_blocks(const std::vector<std::vector<std::size_t>> &blocks) const {
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

    // P
    Eigen::MatrixXd inverse_cholesky =
        this->transpositionsP() * Eigen::MatrixXd::Identity(n, n);
    // L^-1 P
    this->matrixL().solveInPlace(inverse_cholesky);

    // D^-1/2 L^-1 P
    inverse_cholesky = diagonal_sqrt_inverse() * inverse_cholesky;

    assert(!inverse_cholesky.hasNaN());

    std::vector<Eigen::MatrixXd> output;
    for (const auto &block_indices : blocks) {
      Eigen::MatrixXd sub_matrix =
          albatross::subset_cols(inverse_cholesky, block_indices);
      Eigen::MatrixXd one_block =
          sub_matrix.transpose().lazyProduct(sub_matrix);
      output.push_back(one_block);
    }
    return output;
  }

  /*
   * The diagonal of the inverse of the matrix this LDLT
   * decomposition represents in O(n^2) operations.
   */
  Eigen::VectorXd inverse_diagonal() const {
    Eigen::Index n = this->rows();

    std::size_t size_n = static_cast<std::size_t>(n);
    std::vector<std::vector<std::size_t>> block_indices(size_n);
    for (Eigen::Index i = 0; i < n; i++) {
      block_indices[i] = {static_cast<std::size_t>(i)};
    }

    Eigen::VectorXd inv_diag(n);
    const auto blocks = inverse_blocks(block_indices);
    for (std::size_t i = 0; i < size_n; i++) {
      assert(blocks[i].rows() == 1);
      assert(blocks[i].cols() == 1);
      inv_diag[i] = blocks[i](0, 0);
    }

    return inv_diag;
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
