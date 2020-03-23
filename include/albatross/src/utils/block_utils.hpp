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

#ifndef INCLUDE_ALBATROSS_UTILS_BLOCK_UTILS_H_
#define INCLUDE_ALBATROSS_UTILS_BLOCK_UTILS_H_

namespace albatross {

/*
 * One approach to dealing with block linear algebra is to cluster everything
 * into groups these subsequent methods make those representations easier to
 * work with.
 */
template <typename MatrixType>
inline Eigen::MatrixXd block_sum(const std::vector<MatrixType> &xs) {
  MatrixType output = xs[0];
  for (std::size_t i = 1; i < xs.size(); ++i) {
    // Eigen internally asserts that the results are the same size.
    output += xs[i];
  }
  return output;
}

template <typename GroupKey, typename MatrixType>
inline MatrixType block_sum(const Grouped<GroupKey, MatrixType> &xs) {
  return block_sum(xs.values());
}

/*
 * Patchwork GP works by clustering all the data into groups which
 * results in several Grouped objects containing block matrix representations.
 *
 * These subsequent methods make those representations easier to work with.
 */
template <typename GroupKey, typename X, typename Y, typename ApplyFunction>
inline Eigen::MatrixXd block_accumulate(const Grouped<GroupKey, X> &lhs,
                                        const Grouped<GroupKey, Y> &rhs,
                                        const ApplyFunction &apply_function) {
  // block_accumulate takes two different grouped objects and returns
  // the sum of that function applied to each pair of values.  Another
  // way of writing this could be something like:
  //
  //   sum_i ( apply_function(lhs.at(key_i), rhs.at(key_i)) )
  //
  // The result of apply_function is expected to be an Eigen::MatrixXd
  static_assert(
      std::is_same<Eigen::MatrixXd,
                   typename invoke_result<ApplyFunction, X, Y>::type>::value,
      "apply_function needs to return an Eigen::MatrixXd type");

  assert(lhs.size() == rhs.size());
  assert(lhs.size() > 0);

  auto one_group = [&](const GroupKey &key) {
    assert(map_contains(lhs, key) && map_contains(rhs, key));
    return apply_function(lhs.at(key), rhs.at(key));
  };

  return block_sum(apply(lhs.keys(), one_group));
}

template <typename GroupKey, typename ApplyFunction>
inline Eigen::MatrixXd
block_product(const Grouped<GroupKey, Eigen::MatrixXd> &lhs,
              const Grouped<GroupKey, Eigen::MatrixXd> &rhs) {
  // This performs a block matrix product operation where if you aligned the
  // lhs into horizontal blocks and the right into vertical blocks by ordering
  // their keys:
  //
  //   lhs = [x_0, ..., x_n]
  //   rhs = [y_0, ..., y_n]
  //
  // the output corresponds to:
  //
  //   lhs * rhs = [x_0, ..., x_2] * [y_0
  //                                  ...
  //                                  y_2]
  //
  auto matrix_product = [&](const auto &x, const auto &y) {
    return (x * y).eval();
  };

  return block_accumulate(lhs, rhs, matrix_product);
}

template <typename GroupKey>
inline Eigen::MatrixXd
block_inner_product(const Grouped<GroupKey, Eigen::MatrixXd> &lhs,
                    const Grouped<GroupKey, Eigen::MatrixXd> &rhs) {
  // This performs a block matrix inner product operation where if you aligned
  // the lhs into horizontal blocks and the right into vertical blocks by
  // ordering their keys:
  //
  //   lhs = [x_0, ..., x_n]
  //   rhs = [y_0, ..., y_n]
  //
  // the output corresponds to:
  //
  //   lhs.T * rhs = [x_0.T, ..., x_n.T] * [y_0
  //                                        ...
  //                                        y_n]
  //
  auto matrix_inner_product = [&](const auto &x, const auto &y) {
    return (x.transpose() * y).eval();
  };

  return block_accumulate(lhs, rhs, matrix_inner_product);
}

template <typename GroupKey, typename Solver, typename Rhs>
inline auto block_diag_solve(const Grouped<GroupKey, Solver> &lhs,
                             const Grouped<GroupKey, Rhs> &rhs) {
  // Here we have the equivalent to a block diagonal matrix solve
  // in which the inverse of each group in the lhs is applied to
  // the corresponding group in rhs.
  //
  //   lhs = [x_0, ..., x_n]
  //   rhs = [y_0, ..., y_n]
  //
  // the output corresponds to:
  //
  //   lhs.T * rhs = [x_0^-1, ..., x_n^-1] * [y_0
  //                                          ...
  //                                          y_n]
  //
  auto solve_one_block = [&](const auto &key, const auto &x) {
    return Eigen::MatrixXd(lhs.at(key).solve(x));
  };

  return rhs.apply(solve_one_block);
};

template <typename GroupKey>
inline Grouped<GroupKey, Eigen::MatrixXd>
block_subtract(const Grouped<GroupKey, Eigen::MatrixXd> &lhs,
               const Grouped<GroupKey, Eigen::MatrixXd> &rhs) {

  assert(lhs.size() == rhs.size());
  auto matrix_subtract = [&](const auto &key_i, const auto &rhs_i) {
    return (lhs.at(key_i) - rhs_i).eval();
  };

  return rhs.apply(matrix_subtract);
}

template <typename MatrixType, unsigned int Mode = Eigen::Lower>
struct BlockTriangularView;

struct BlockDiagonalLLT;
struct BlockDiagonal;

struct BlockDiagonalLLT {
  std::vector<Eigen::LLT<Eigen::MatrixXd>> blocks;

  template <class _Scalar, int _Rows, int _Cols>
  Eigen::Matrix<_Scalar, _Rows, _Cols>
  solve(const Eigen::Matrix<_Scalar, _Rows, _Cols> &rhs) const;

  BlockTriangularView<const Eigen::MatrixXd> matrixL() const;

  double log_determinant() const;

  Eigen::Index rows() const;

  Eigen::Index cols() const;
};

template <typename MatrixType, unsigned int Mode> struct BlockTriangularView {
  std::vector<Eigen::TriangularView<MatrixType, Mode>> blocks;

  template <class _Scalar, int _Rows, int _Cols>
  Eigen::Matrix<_Scalar, _Rows, _Cols>
  operator*(const Eigen::Matrix<_Scalar, _Rows, _Cols> &rhs) const;

  template <class _Scalar, int _Rows, int _Cols>
  Eigen::Matrix<_Scalar, _Rows, _Cols>
  solve(const Eigen::Matrix<_Scalar, _Rows, _Cols> &rhs) const;

  Eigen::Index rows() const;

  Eigen::Index cols() const;

  Eigen::MatrixXd toDense() const;
};

struct BlockDiagonal {
  std::vector<Eigen::MatrixXd> blocks;

  template <class _Scalar, int _Rows, int _Cols>
  Eigen::Matrix<_Scalar, _Rows, _Cols>
  operator*(const Eigen::Matrix<_Scalar, _Rows, _Cols> &rhs) const;

  BlockDiagonal operator-(const BlockDiagonal &rhs) const;

  Eigen::VectorXd diagonal() const;

  BlockDiagonalLLT llt() const;

  Eigen::Index rows() const;

  Eigen::Index cols() const;

  Eigen::MatrixXd toDense() const;
};

template <typename Solver> struct BlockSymmetric {

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
  BlockSymmetric(){};

  BlockSymmetric(const Solver &A_, const Eigen::MatrixXd &B_,
                 const Eigen::SerializableLDLT &S_)
      : A(A_), Ai_B(A_.solve(B_)), S(S_) {}

  BlockSymmetric(const Solver &A_, const Eigen::MatrixXd &B_,
                 const Eigen::MatrixXd &C)
      : BlockSymmetric(
            A_, B_,
            Eigen::SerializableLDLT(C - B_.transpose() * A_.solve(B_))){};

  template <class _Scalar, int _Rows, int _Cols>
  Eigen::Matrix<_Scalar, _Rows, _Cols>
  solve(const Eigen::Matrix<_Scalar, _Rows, _Cols> &rhs) const;

  bool operator==(const BlockSymmetric &rhs) const;

  Eigen::Index rows() const;

  Eigen::Index cols() const;

  Solver A;
  Eigen::MatrixXd Ai_B;
  Eigen::SerializableLDLT S;
};

/*
 * BlockDiagonalLLT
 */
template <class _Scalar, int _Rows, int _Cols>
inline Eigen::Matrix<_Scalar, _Rows, _Cols>
BlockDiagonalLLT::solve(const Eigen::Matrix<_Scalar, _Rows, _Cols> &rhs) const {
  assert(cols() == rhs.rows());
  Eigen::Index i = 0;
  Eigen::Matrix<_Scalar, _Rows, _Cols> output(rows(), rhs.cols());
  for (const auto &b : blocks) {
    const auto rhs_chunk = rhs.block(i, 0, b.rows(), rhs.cols());
    output.block(i, 0, b.rows(), rhs.cols()) = b.solve(rhs_chunk);
    i += b.rows();
  }
  return output;
}

inline BlockTriangularView<const Eigen::MatrixXd>
BlockDiagonalLLT::matrixL() const {
  BlockTriangularView<const Eigen::MatrixXd> output;
  for (const auto &b : blocks) {
    Eigen::TriangularView<const Eigen::MatrixXd, Eigen::Lower> L = b.matrixL();
    output.blocks.push_back(L);
  }
  return output;
}

inline double BlockDiagonalLLT::log_determinant() const {
  double output = 0.;
  for (const auto &b : blocks) {
    output += 2. * log(b.matrixL().determinant());
  }
  return output;
}

inline Eigen::Index BlockDiagonalLLT::rows() const {
  Eigen::Index n = 0;
  for (const auto &b : blocks) {
    n += b.rows();
  }
  return n;
}

inline Eigen::Index BlockDiagonalLLT::cols() const {
  Eigen::Index n = 0;
  for (const auto &b : blocks) {
    n += b.cols();
  }
  return n;
}

/*
 * BlockTriangularView
 */
template <typename MatrixType, unsigned int Mode>
template <class _Scalar, int _Rows, int _Cols>
inline Eigen::Matrix<_Scalar, _Rows, _Cols>
BlockTriangularView<MatrixType, Mode>::solve(
    const Eigen::Matrix<_Scalar, _Rows, _Cols> &rhs) const {
  assert(cols() == rhs.rows());
  Eigen::Index i = 0;
  Eigen::Matrix<_Scalar, _Rows, _Cols> output(rows(), rhs.cols());
  for (const auto &b : blocks) {
    const auto rhs_chunk = rhs.block(i, 0, b.rows(), rhs.cols());
    output.block(i, 0, b.rows(), rhs.cols()) = b.solve(rhs_chunk);
    i += b.rows();
  }
  return output;
}

template <typename MatrixType, unsigned int Mode>
template <class _Scalar, int _Rows, int _Cols>
inline Eigen::Matrix<_Scalar, _Rows, _Cols>
    BlockTriangularView<MatrixType, Mode>::
    operator*(const Eigen::Matrix<_Scalar, _Rows, _Cols> &rhs) const {
  assert(cols() == rhs.rows());
  Eigen::Index i = 0;
  Eigen::Matrix<_Scalar, _Rows, _Cols> output =
      Eigen::Matrix<_Scalar, _Rows, _Cols>::Zero(rows(), rhs.cols());
  for (const auto &b : blocks) {
    const auto rhs_chunk = rhs.block(i, 0, b.rows(), rhs.cols());
    output.block(i, 0, b.rows(), rhs.cols()) = b * rhs_chunk;
    i += b.rows();
  }
  return output;
}

template <typename MatrixType, unsigned int Mode>
inline Eigen::Index BlockTriangularView<MatrixType, Mode>::rows() const {
  Eigen::Index n = 0;
  for (const auto &b : blocks) {
    n += b.rows();
  }
  return n;
}

template <typename MatrixType, unsigned int Mode>
inline Eigen::Index BlockTriangularView<MatrixType, Mode>::cols() const {
  Eigen::Index n = 0;
  for (const auto &b : blocks) {
    n += b.cols();
  }
  return n;
}

template <typename MatrixType, unsigned int Mode>
inline Eigen::MatrixXd BlockTriangularView<MatrixType, Mode>::toDense() const {
  Eigen::MatrixXd output = Eigen::MatrixXd::Zero(rows(), cols());

  Eigen::Index i = 0;
  Eigen::Index j = 0;
  for (const auto &b : blocks) {
    output.block(i, j, b.rows(), b.cols()) = b;
    i += b.rows();
    j += b.cols();
  }
  return output;
}

/*
 * Block Diagonal
 */

template <class _Scalar, int _Rows, int _Cols>
inline Eigen::Matrix<_Scalar, _Rows, _Cols> BlockDiagonal::
operator*(const Eigen::Matrix<_Scalar, _Rows, _Cols> &rhs) const {
  assert(cols() == rhs.rows());
  Eigen::Index i = 0;
  Eigen::Matrix<_Scalar, _Rows, _Cols> output =
      Eigen::Matrix<_Scalar, _Rows, _Cols>::Zero(rows(), rhs.cols());
  for (const auto &b : blocks) {
    const auto rhs_chunk = rhs.block(i, 0, b.rows(), rhs.cols());
    output.block(i, 0, b.rows(), rhs.cols()) = b * rhs_chunk;
    i += b.rows();
  }
  return output;
}

inline BlockDiagonal BlockDiagonal::operator-(const BlockDiagonal &rhs) const {
  assert(cols() == rhs.rows());
  assert(blocks.size() == rhs.blocks.size());

  BlockDiagonal output;
  for (std::size_t i = 0; i < blocks.size(); ++i) {
    assert(blocks[i].size() == rhs.blocks[i].size());
    output.blocks.emplace_back(blocks[i] - rhs.blocks[i]);
  }
  return output;
}

inline Eigen::MatrixXd BlockDiagonal::toDense() const {
  Eigen::MatrixXd output = Eigen::MatrixXd::Zero(rows(), cols());

  Eigen::Index i = 0;
  Eigen::Index j = 0;
  for (const auto &b : blocks) {
    output.block(i, j, b.rows(), b.cols()) = b;
    i += b.rows();
    j += b.cols();
  }
  return output;
}

inline Eigen::VectorXd BlockDiagonal::diagonal() const {
  assert(rows() == cols());
  Eigen::VectorXd output(rows());

  Eigen::Index i = 0;
  for (const auto b : blocks) {
    output.block(i, 0, b.rows(), 1) = b.diagonal();
    i += b.rows();
  }
  return output;
}

inline BlockDiagonalLLT BlockDiagonal::llt() const {
  BlockDiagonalLLT output;
  for (const auto &b : blocks) {
    output.blocks.emplace_back(b.llt());
  }
  return output;
}

inline Eigen::Index BlockDiagonal::rows() const {
  Eigen::Index n = 0;
  for (const auto &b : blocks) {
    n += b.rows();
  }
  return n;
}

inline Eigen::Index BlockDiagonal::cols() const {
  Eigen::Index n = 0;
  for (const auto &b : blocks) {
    n += b.cols();
  }
  return n;
}

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
  assert(rhs.rows() == n);

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
inline bool BlockSymmetric<Solver>::
operator==(const BlockSymmetric &rhs) const {
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

} // namespace albatross

#endif /* INCLUDE_ALBATROSS_UTILS_BLOCK_UTILS_H_ */
