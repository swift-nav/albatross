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
};

struct StructuredQ {
  BlockDiagonal upper_left;
  Eigen::MatrixXd lower_right;
};

struct StructuredR {
  BlockR upper_left;
  Eigen::MatrixXd upper_right;
  DenseR lower_right;
};

struct StructuredQR {
  StructuredQ Q;
  StructuredR R;
}

// static
// Eigen::MatrixXd get_Q(const Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> &cod) {
//   const Eigen::MatrixXd Q_full = cod.matrixQ();
//   return Q_full.leftCols(cod.rank());
// }

// static
// Eigen::MatrixXd nullspace(const Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> &cod) {
//   const Eigen::MatrixXd Z_null = cod.matrixZ().bottomRows(cod.cols() - cod.rank());
//   return (cod.colsPermutation() * Z_null.transpose());
// }

namespace detail {

/*
 * Holds the precomputations required for a single independent block
 * in a structured QR, namely given
 *
 *   A_i = [B_i C_i]
 *
 * this holds,
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
struct IndependentQR {
  Eigen::MatrixXd Q;
  DenseR R;
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
static
StructuredR create_structured_R(const BlockDiagonal &left,
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

  auto precompute_one_block = [&](const auto &i) {
    // TODO: Avoid copying in here?
    const Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(left.blocks[i]);
    assert(qr.rank() == left.blocks[i].cols());
    const Eigen::MatrixXd Q = qr.matrixQ();
    const Eigen::MatrixXd cross = Q.transpose() * right.block(offsets[i], 0, left.blocks[i].rows(), common_cols);
    DenseR dense;
    dense.R = get_R(qr);
    dense.P = qr.colsPermutation();
    return detail::IndependentQR{Q, dense, cross};
  };

  // TODO: async
  const auto precomputed = apply(inds, precompute_one_block);

  StructuredR output;
  output.upper_right = Eigen::MatrixXd::Zero(left.cols(), common_cols);
  Eigen::Index augmented_rows = common_cols + left.rows() - left.cols();
  Eigen::MatrixXd X(augmented_rows, common_cols);
  // Fill in C_X
  X.topRows(common_cols) = right.bottomRows(common_cols);
  Eigen::Index upper_offset = 0;
  Eigen::Index X_offset = common_cols;
  for (std::size_t i = 0; i < n_blocks; ++i) {
    const Eigen::Index rows_i = precomputed[i].R.R.rows();
    output.upper_right.block(upper_offset, 0, rows_i, common_cols) = precomputed[i].cross.topRows(rows_i);
    upper_offset += rows_i;

    const Eigen::Index remaining_rows = precomputed[i].cross.rows() - rows_i;
    X.block(X_offset, 0, remaining_rows, common_cols) = precomputed[i].cross.bottomRows(remaining_rows);
    X_offset += remaining_rows;

    output.upper_left.blocks.emplace_back(std::move(precomputed[i].R));
  }

  const Eigen::MatrixXd UR = output.upper_right.transpose() * output.upper_right;
  const Eigen::MatrixXd XX = X.transpose() * X;
  const Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(X);
  output.lower_right = DenseR{get_R(qr), qr.colsPermutation().indices()};
  return output;
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

  Eigen::Index offset = 0;
  for (const auto &block : block_R.blocks) {
    for (Eigen::Index i = 0; i < block.P.size(); ++i) {
      sqrt.row(offset + i) = rhs.row(offset + block.P.coeff(i));
    }

    const Eigen::MatrixXd foo = sqrt.block(offset, 0, block.R.rows(), rhs.cols());
    const Eigen::MatrixXd Rti_foo = block.R.template triangularView<Eigen::Upper>().transpose().solve(foo);
    // std::cout << "R : " << std::endl;
    // std::cout << R << std::endl;
    // std::cout << "FOO : " << std::endl;
    // std::cout << foo << std::endl;
    // std::cout << "Rti_foo : " << std::endl;
    // std::cout << Rti_foo << std::endl;
    sqrt.block(offset, 0, block.R.rows(), rhs.cols()) = Rti_foo;
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
  x = sqrt_solve(structured.upper_left, rhs.topRows(block_rows));

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

  Eigen::MatrixXd corner = rhs.bottomRows(structured.lower_right.R.rows());
  corner = corner - structured.upper_right.transpose() * x;

  const Eigen::Index offset = structured.upper_left.rows();
  for (Eigen::Index i = 0; i < structured.lower_right.P.size(); ++i) {
    sqrt.row(offset + i) = corner.row(structured.lower_right.P.coeff(i));
  }

  sqrt.bottomRows(corner.rows()) = structured.lower_right.R.template triangularView<Eigen::Upper>().transpose().solve(sqrt.bottomRows(structured.lower_right.R.rows()));

  return sqrt;
}

}

#endif