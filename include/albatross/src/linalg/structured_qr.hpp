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

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace albatross {

using PermutationIndices = Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1>;

struct DenseR {
  Eigen::MatrixXd R;
  // Z is unitary
  Eigen::MatrixXd Z;
};

struct StructuredQR {
  std::vector<DenseQR> upper_left;
  Eigen::MatrixXd upper_right;
  DenseQR lower_right;
};

Eigen::MatrixXd get_Q(const Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> &cod) {
  return cod.matrixQ().leftCols(cod.rank());
}

Eigen::MatrixXd get_Q(const Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> &cod) {
  return cod.matrixQ().leftCols(cod.rank());
}

/*
 * Computes the QR decomposition of a matrix given by,
 *
 *  A = |       L_upper        |
 *      | ll_0  0     0   lr_0 |
 *      |   0  ll_i   0   lr_i |
 *      |   0   0   ll_n  lr_n |
 *
 * where L_upper is an upper triangular cholesky transposed
 * ll_i are block diagonal chunks and lr is a dense matrix
 */
StructuredQR create_structured_qr(const BlockSymmetricArrowLDLT &upper,
                                  const BlockDiagonal &lower_left,
                                  const Eigen::MatrixXd &lower_right) {
  const auto n_blocks = lower_left.blocks.size();
  assert(upper.blocks.size() == n_blocks);

  for (std::size_t i = 0; i < n_blocks; ++i) {
    const Eigen::Index block_rows = upper.blocks[i].rows() + lower_left.blocks[i].rows();
    const Eigen::Index block_cols = upper.clocks[i].cols();
    assert(lower_left.blocks[i].cols() == block_cols);
    Eigen::MatrixXd block = Eigen::MatrixXd::Zero(block_rows, block_cols);
    block.topRows(upper.blocks[i].rows()) = upper.blocks[i].sqrt_transpose();

    
    
    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(block);

    
    
  }
  
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

}