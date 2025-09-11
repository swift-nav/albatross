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

#ifndef ALBATROSS_SRC_LINALG_BLOCK_SYMMETRIC_HPP
#define ALBATROSS_SRC_LINALG_BLOCK_SYMMETRIC_HPP

namespace albatross {

/*
 * Stores a covariance matrix which takes the form:
 *
 *   X = |A   B|
 *       |B.T C|
 *
 * It is assumes that both A and C - B.T A^-1 B are invertible and is
 * designed for the situation where A is larger than C.  The primary
 * use case is for a situation where you have a pre computed LDLT of
 * a submatrix (A) and you'd like to perform a solve of the larger
 * matrix (X)
 *
 * To do so the rules for block inversion are used:
 *
 * https://en.wikipedia.org/wiki/Block_matrix#Block_matrix_inversion
 *
 * which leads to:
 *
 *   X^-1 = |A   B|^-1
 *          |B.T C|
 *
 *        = |A^-1 + Ai_B S^-1 Ai_B^T    -Ai_B S^-1|
 *          |-S^-1 Ai_B^T                    S^-1  |
 *
 * where Ai_B = A^-1 B  and S = C - B^T A^-1 B.
 *
 * In this particular implementation Ai_B and S^-1 are pre-computed.
 */
template <typename Solver>
struct BlockSymmetric {
  BlockSymmetric() {}

  BlockSymmetric(const Solver &A_, const Eigen::MatrixXd &B_,
                 const Eigen::SerializableLDLT &S_)
      : A(A_), Ai_B(A_.solve(B_)), S(S_) {}

  BlockSymmetric(const Solver &A_, const Eigen::MatrixXd &B_,
                 const Eigen::MatrixXd &C)
      : BlockSymmetric(
            A_, B_,
            Eigen::SerializableLDLT(C - B_.transpose() * A_.solve(B_))) {}

  template <class _Scalar, int _Rows, int _Cols>
  Eigen::Matrix<_Scalar, _Rows, _Cols> solve(
      const Eigen::Matrix<_Scalar, _Rows, _Cols> &rhs) const;

  bool operator==(const BlockSymmetric &rhs) const;

  Eigen::Index rows() const;

  Eigen::Index cols() const;

  Solver A;
  Eigen::MatrixXd Ai_B;
  Eigen::SerializableLDLT S;
};

/*
 * BlockSymmetric
 *
 */

template <typename Solver>
template <class _Scalar, int _Rows, int _Cols>
inline Eigen::Matrix<_Scalar, _Rows, _Cols> BlockSymmetric<Solver>::solve(
    const Eigen::Matrix<_Scalar, _Rows, _Cols> &rhs) const {
  // https://en.wikipedia.org/wiki/Block_matrix#Block_matrix_inversion
  Eigen::Index n = A.rows() + S.rows();
  ALBATROSS_ASSERT(rhs.rows() == n);

  const Eigen::MatrixXd rhs_a = rhs.topRows(A.rows());
  const Eigen::MatrixXd rhs_b = rhs.bottomRows(S.rows());

  const auto Bt_Ai_rhs = Ai_B.transpose() * rhs_a;

  const auto Si_Bt_Ai_rhs = S.solve(Bt_Ai_rhs);
  const auto upper_left = A.solve(rhs_a) + Ai_B * Si_Bt_Ai_rhs;

  Eigen::Matrix<_Scalar, _Rows, _Cols> output(n, rhs.cols());
  output.topRows(A.rows()) = upper_left - Ai_B * S.solve(rhs_b);
  output.bottomRows(S.rows()) = S.solve(rhs_b) - Si_Bt_Ai_rhs;

  return output;
}

template <typename Solver>
inline bool BlockSymmetric<Solver>::operator==(
    const BlockSymmetric &rhs) const {
  return (A == rhs.A && Ai_B == rhs.Ai_B && S == rhs.S);
}

template <typename Solver>
inline Eigen::Index BlockSymmetric<Solver>::rows() const {
  return A.rows() + S.rows();
}

template <typename Solver>
inline Eigen::Index BlockSymmetric<Solver>::cols() const {
  return A.cols() + S.cols();
}

template <typename Solver>
BlockSymmetric<Solver> build_block_symmetric(const Solver &A,
                                             const Eigen::MatrixXd &B,
                                             const Eigen::SerializableLDLT &S) {
  return BlockSymmetric<Solver>(A, B, S);
}

template <typename Solver>
BlockSymmetric<Solver> build_block_symmetric(const Solver &A,
                                             const Eigen::MatrixXd &B,
                                             const Eigen::MatrixXd &C) {
  return BlockSymmetric<Solver>(A, B, C);
}

}  // namespace albatross

#endif  // ALBATROSS_SRC_LINALG_BLOCK_SYMMETRIC_HPP
