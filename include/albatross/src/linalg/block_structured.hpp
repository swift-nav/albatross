
/*
 * Copyright (C) 2023 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef ALBATROSS_SRC_LINALG_struct_HPP
#define ALBATROSS_SRC_LINALG_struct_HPP

namespace albatross {

using PermInds = Eigen::Matrix<int, Eigen::Dynamic, 1>;
using PermutationMatrixXd = Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>;

struct DenseR {
  Eigen::MatrixXd R;
  PermutationMatrixXd P;

  Eigen::Index rows() const {
    assert(R.rows() == R.cols());
    assert(P.size() == R.rows());
    return R.rows();
  }

  Eigen::Index cols() const {
    assert(R.rows() == R.cols());
    assert(P.size() == R.rows());
    return R.cols();
  }
};

struct BlockR {
  std::vector<DenseR> blocks;

  Eigen::Index rows() const {
    Eigen::Index output = 0;
    for (const auto &block : blocks) {
      output += block.rows();
    }
    return output;
  }

  Eigen::Index cols() const {
    Eigen::Index output = 0;
    for (const auto &block : blocks) {
      output += block.cols();
    }
    return output;
  }
};

/*
 * Represents a product of two orthogonal matrices
 *
 * |Q_0   0    0   N_0   0 | |I  0 |
 * | 0   Q_1   0    0   N_1| |0 Q_n|
 * | 0    0    I    0    0 |
 *
 */
struct StructuredQ {
  std::vector<Eigen::MatrixXd> Q_blocks;
  std::vector<Eigen::MatrixXd> N_blocks;
  Eigen::MatrixXd corner;

  Eigen::Index rows() const {
    Eigen::Index output = corner.rows();
    for (std::size_t i = 0; i < Q_blocks.size(); ++i) {
      assert(Q_blocks[i].rows() == N_blocks[i].rows());
      output += Q_blocks[i].rows();
      output -= N_blocks[i].cols();
    }
    return output;
  }

  Eigen::Index cols() const {
    return this->rows();
  }
};

struct StructuredR {
  BlockR upper_left;
  Eigen::MatrixXd upper_right;
  DenseR lower_right;

  Eigen::Index rows() const {
    return upper_left.rows() + lower_right.rows();
  }
  Eigen::Index cols() const {
    return upper_left.cols() + upper_right.cols();
  }
};

struct StructuredQR {
  StructuredQ Q;
  StructuredR R;
};

namespace detail {

/*
 * Holds the precomputations required for a single independent block
 * in a structured QR, namely given
 *
 *   A_i = [B_i C_i]
 *
 * this holds, Q and,
 *
 *   |R_i U_i|
 *   | 0  L_i|
 *
 * such that
 *
 *   A_i = Q |R_i U_i|
 *           | 0  L_i|
 *
 * where
 *
 *   cross = |U_i|
 *           |L_i|
 */
struct PartialQR {
  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr;
  Eigen::MatrixXd cross;
};
}

/* Given:
 *  A = |  B_0    0     0     C_O |
 *      |   0    B_1    0     C_1 |
 *      |   0     0    B_N^T  C_N |
 *      |   0     0     0     C_X |
 *
 * computes:
 *
 *  A = Q R Z P^T
 *
 */
static StructuredQR create_structured_qr(const BlockDiagonal &left,
                                         const Eigen::MatrixXd &right) {
  const auto n_blocks = left.blocks.size();
  const auto common_cols = right.cols();
  assert(right.rows() == left.rows() + common_cols);

  std::vector<Eigen::Index> inds;
  std::vector<Eigen::Index> offsets;
  Eigen::Index offset = 0;
  for (std::size_t i = 0; i < n_blocks; ++i) {
    inds.push_back(i);
    offsets.push_back(offset);
    offset += left.blocks[i].rows();
  }

  auto compute_partial_qr = [&](const auto &i) {
    // TODO: Avoid copying in here?
    detail::PartialQR partial;
    partial.qr = left.blocks[i].colPivHouseholderQr();
    assert(partial.qr.rank() == left.blocks[i].cols());
    partial.cross =
        partial.qr.matrixQ().transpose() *
        right.block(offsets[i], 0, left.blocks[i].rows(), common_cols);
    return partial;
  };

  // TODO: async
  const auto partial_qrs = apply(inds, compute_partial_qr);

  StructuredQ Q;
  StructuredR R;
  R.upper_right = Eigen::MatrixXd::Zero(left.cols(), common_cols);
  Eigen::Index augmented_rows = common_cols + left.rows() - left.cols();
  Eigen::MatrixXd X(augmented_rows, common_cols);
  // Fill in C_X
  X.topRows(common_cols) = right.bottomRows(common_cols);
  Eigen::Index upper_offset = 0;
  Eigen::Index X_offset = common_cols;
  for (std::size_t i = 0; i < n_blocks; ++i) {
    // TODO: actually use rank? but that might require a switch to COD?
    const Eigen::Index rank = partial_qrs[i].qr.rank();
    assert(rank == partial_qrs[i].qr.cols());
    const Eigen::Index null_rank = partial_qrs[i].qr.rows() - rank;

    R.upper_right.block(upper_offset, 0, rank, common_cols) =
        partial_qrs[i].cross.topRows(rank);
    upper_offset += rank;

    X.block(X_offset, 0, null_rank, common_cols) =
        partial_qrs[i].cross.bottomRows(null_rank);
    X_offset += null_rank;

    // TODO: avoid copying to Q?
    Eigen::MatrixXd Qi = partial_qrs[i].qr.matrixQ();
    Q.Q_blocks.emplace_back(Qi.leftCols(rank));
    Q.N_blocks.emplace_back(Qi.rightCols(null_rank));
    R.upper_left.blocks.emplace_back(
        DenseR{get_R(partial_qrs[i].qr), partial_qrs[i].qr.colsPermutation()});
  }

  // const Eigen::MatrixXd UR = R.upper_right.transpose() * R.upper_right;
  const Eigen::MatrixXd XX = X.transpose() * X;
  const Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(X);
  Q.corner = qr.matrixQ();
  R.lower_right = DenseR{get_R(qr), qr.colsPermutation()};
  return StructuredQR{Q, R};
}

/*
 * A P = QR
 * A = Q R P^T
 * A^T A        = P R^T R P^T
 *              = (P R^T) (P R^T)^T
 * (A^T A)^-1/2 = (P R^T)^-1
 *              = R^-T P^T
 */
template <typename MatrixType>
inline Eigen::MatrixXd
sqrt_solve(const BlockR &block_R,
           const MatrixType &rhs) {

  Eigen::MatrixXd sqrt = Eigen::MatrixXd::Zero(rhs.rows(), rhs.cols());
  assert(rhs.rows() == block_R.rows());
  std::cout << "RHS:" << std::endl;
  std::cout << rhs << std::endl;
  Eigen::Index offset = 0;
  for (const auto &block : block_R.blocks) {
    const auto rhs_i = rhs.block(offset, 0, block.R.rows(), rhs.cols());
    assert(block.P.rows() == rhs_i.rows());
    const auto R_i = block.R.template triangularView<Eigen::Upper>();
    const auto &P_i = block.P;
    auto sqrt_i = sqrt.block(offset, 0, block.R.rows(), rhs.cols());
    sqrt_i = R_i.transpose().solve(P_i.transpose() * rhs_i);
    offset += block.R.rows();
  }
  return sqrt;
}

/*
 * A P = QR
 * A = Q R P^T
 * A^T A        = P R^T R P^T
 *              = (P R^T) (P R^T)^T
 * (A^T A)^-1/2 = (P R^T)^-1
 *              = R^-T P^T
 */
template <typename MatrixType>
inline Eigen::MatrixXd
sqrt_solve(const StructuredR &structured,
           const MatrixType &rhs) {

  // We're trying to invert the following for x, y
  //   |B 0| |x = |rhs_a
  //   |C D| |y   |rhs_b
  // Which can be broken into two operations,
  //
  //   B x = rhs_a
  //   C x + D y = rhs_b
  //
  // First we solve for x:
  Eigen::MatrixXd sqrt = Eigen::MatrixXd::Zero(rhs.rows(), rhs.cols());
  const auto block_rows = structured.upper_left.rows();
  auto x = sqrt.topRows(block_rows);
  auto rhs_a = rhs.topRows(block_rows);
  x = sqrt_solve(structured.upper_left, rhs_a);

  // At this point we've inverted the block diagonal upper section, so we know
  // x and need y.
  //
  //   C x + D y = rhs_b
  //   D y = rhs_b - C x
  //
  // We have D = P R^T so want
  //
  //   P R^T y = rhs_b - C x
  //   y = R^-T P^T (rhs_b - C x)
  auto rhs_b = rhs.bottomRows(structured.lower_right.R.rows());
  Eigen::MatrixXd corner = rhs_b - structured.upper_right.transpose() * x;

  const Eigen::Index offset = structured.upper_left.rows();
  const Eigen::Index rows = corner.rows();

  auto y = sqrt.bottomRows(rows);

  const auto R = structured.lower_right.R.template triangularView<Eigen::Upper>();
  const auto P = structured.lower_right.P;
  y = R.transpose().solve(P.transpose() * corner);

  return sqrt;
}

template <typename RhsType>
inline Eigen::MatrixXd
solve(const DenseR &RP,
      const RhsType &rhs) {
  const auto R = RP.R.template triangularView<Eigen::Upper>();
  return RP.P * R.solve(rhs);
}

/*
 * Computes x such that,
 *
 *   x = R^-1 rhs
 */
template <typename MatrixType>
inline Eigen::MatrixXd
solve(const BlockR &block_R,
      const MatrixType &rhs) {
  Eigen::MatrixXd output = Eigen::MatrixXd::Zero(rhs.rows(), rhs.cols());
  assert(rhs.rows() == block_R.rows());
  Eigen::Index offset = 0;
  for (const auto &block : block_R.blocks) {
    const auto rhs_i = rhs.block(offset, 0, block.R.rows(), rhs.cols());
    auto solve_i = output.block(offset, 0, block.R.rows(), rhs.cols());
    solve_i = solve(block, rhs_i);
    offset += block.R.rows();
  }
  return output;
}

template <typename MatrixType>
inline Eigen::MatrixXd
solve(const StructuredR &R,
      const MatrixType &rhs) {
  assert(R.rows() == R.cols());
  assert(R.cols() == rhs.rows());
  // We're trying to invert the following for x, y
  //   |D C| |x = |rhs_a
  //   |0 B| |y   |rhs_b
  //
  // Which can be broken into two operations,
  //
  //   D x + C y = rhs_a
  //   B y = rhs_b
  //
  // First we solve for y:
  Eigen::MatrixXd output = Eigen::MatrixXd::Zero(R.rows(), rhs.cols());
  auto y = output.bottomRows(R.lower_right.rows());
  const Eigen::MatrixXd rhs_b = rhs.bottomRows(R.lower_right.rows());
  y = solve(R.lower_right, rhs_b);

  // At this point we've inverted the block diagonal upper section, so we know
  // y and need x.
  //
  //   D x + C y = rhs_a
  //   D x = rhs_a - C y
  //   x = D^-1 (rhs_a - C y)
  const auto block_rows = R.upper_left.rows();
  auto x = output.topRows(block_rows);
  auto rhs_a = rhs.topRows(block_rows);
  const Eigen::MatrixXd tmp = rhs_a - R.upper_right * y;
  x = solve(R.upper_left, tmp);

  return output;
}

template <typename MatrixType>
inline
Eigen::MatrixXd dot_transpose(const StructuredQ &Q, const MatrixType &rhs) {
  assert(Q.rows() == rhs.rows());
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  Eigen::MatrixXd output = Eigen::MatrixXd::Zero(Q.cols(), rhs.cols());
  /*
   * Q^T = Q_1^T Q_0^T
   *
   * First we do the Q_0^T step
   *
   *  |Q_0^T rhs_0|   |Q_0^T   0      0| |rhs_0|
   *  |Q_1^T rhs_1|   | 0     Q_1^T   0| |rhs_1|
   *  |      rhs_k| = | 0      0      I| |rhs_k|
   *  |N_0^T rhs_0|   |N_0^T   0      0|
   *  |N_1^T rhs_1|   | 0    N_1^T    0|
   */
  Eigen::Index row = 0;
  Eigen::Index col = 0;
  for (const auto &Q_block : Q.Q_blocks) {
    output.block(row, 0, Q_block.cols(), rhs.cols()) = Q_block.transpose() * rhs.block(col, 0, Q_block.rows(), rhs.cols());
    row += Q_block.cols();
    col += Q_block.rows();
  }
  const Eigen::Index remaining = Q.rows() - col;
  output.block(row, 0, remaining, rhs.cols()) = rhs.bottomRows(remaining);
  row += remaining;

  col = 0;
  for (const auto &N_block : Q.N_blocks) {
    output.block(row, 0, N_block.cols(), rhs.cols()) = N_block.transpose() * rhs.block(col, 0, N_block.rows(), rhs.cols());
    row += N_block.cols();
    col += N_block.rows();
  }

  /*
   * Then we follow with the Q_1^T step
   *
   *
   *  |Q_0^T rhs_0|    |I  0   | |Q_0^T rhs_0|
   *  |Q_1^T rhs_1| =  |0 Q_c^T| |Q_1^T rhs_1|
   *  |Q_k^T rhs_k|              |Q_k^T rhs_k|
   *  |N_0^T rhs_0|              |N_0^T rhs_0|
   *  |N_1^T rhs_1|              |N_1^T rhs_1|
   */
  // TODO: avoid copying here
  const Eigen::MatrixXd tmp = output.bottomRows(Q.corner.rows());
  output.bottomRows(Q.corner.cols()) = Q.corner.transpose() * tmp;
  return output;
}

/*
 * A P = QR
 * A = Q R P^T
 *
 * A x = b
 * (A^T A) x = A^T b
 * x = (A^T A)^-1 A^T b
 *   = (P R^T R P^T)^-1 P R^T Q^T b
 *   = P R^-1 R^-T P^T P R^T Q^T b
 *   = P R^-1 R^-T R^T Q^T b
 *   = P R^-1 Q^T b
 */
template <typename MatrixType>
inline Eigen::MatrixXd
solve(const StructuredQR &structured,
      const MatrixType &rhs) {
  // TODO: only compute the top rows, instead of computing them all and chopping
  Eigen::MatrixXd tmp = dot_transpose(structured.Q, rhs).topRows(structured.R.cols());
  return solve(structured.R, tmp);
}

}

#endif