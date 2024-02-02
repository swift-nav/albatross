/*
 * Copyright (C) 2024 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef INCLUDE_ALBATROSS_LINALG_PERMUTATION_UTILS_H_
#define INCLUDE_ALBATROSS_LINALG_PERMUTATION_UTILS_H_

namespace albatross {
namespace permute {

/*
 * Permuation operations assume that a permutation matrix is given by
 * a vector of indices, where the operation P X would map row X[i] to
 * row indices[i]. In other words,
 *
 *   (P X).row(indices[i]) == X[i];
 *
 * Another way to look at this is that a permuation matrix which looks
 * like,
 *
 *       |0 0 1 0|
 *   P = |0 1 0 0|
 *       |0 0 0 1|
 *       |1 0 0 0|
 *
 * would be encoded using indices as:
 *
 * inds = [2, 1, 3, 0]
 */

/*
 * Given a permuation matrix, P, defined by indices, this computes
 * P^T * X
 */
template<typename PermutationIndicesType>
Eigen::MatrixXd transpose_from_left(const PermutationIndicesType &indices,
                                    const Eigen::MatrixXd &rhs) {

  Eigen::MatrixXd output(rhs.rows(), rhs.cols());
  assert(indices.size() == rhs.rows() && "indices size must match matrix size");
  for (Eigen::Index i = 0; i < indices.size(); ++i) {
    assert(indices.coeff(i) >= 0 && "permuation indices must be non negative");
    assert(indices.coeff(i) < output.rows() &&
           "permuation indices exceed size of the matrix");
    output.row(i) = rhs.row(indices.coeff(i));
  }
  return output;
}

/*
 * Given a permuation matrix, P, defined by indices, this computes
 * P * X
 */
template <typename PermutationIndicesType>
Eigen::MatrixXd from_left(const PermutationIndicesType &indices,
                          const Eigen::MatrixXd &rhs) {
  Eigen::MatrixXd output(rhs.rows(), rhs.cols());
  assert(indices.size() == rhs.rows() && "indices size must match matrix size");
  for (Eigen::Index i = 0; i < indices.size(); ++i) {
    assert(indices.coeff(i) >= 0 && "permuation indices must be non negative");
    assert(indices.coeff(i) < output.rows() &&
           "permuation indices exceed size of the matrix");
    output.row(indices.coeff(i)) = rhs.row(i);
  }
  return output;
}

/*
 * Given a permuation matrix, P, defined by indices, this computes
 * X * P^T
 */
template <typename PermutationIndicesType>
Eigen::MatrixXd transpose_from_right(const Eigen::MatrixXd &rhs,
                                     const PermutationIndicesType &indices) {
  Eigen::MatrixXd output(rhs.rows(), rhs.cols());
  assert(indices.size() == rhs.rows() && "indices size must match matrix size");
  for (Eigen::Index i = 0; i < indices.size(); ++i) {
    assert(indices.coeff(i) >= 0 && "permuation indices must be non negative");
    assert(indices.coeff(i) < output.rows() &&
           "permuation indices exceed size of the matrix");
    output.col(indices.coeff(i)) = rhs.col(i);
  }
  return output;
}

/*
 * Given a permuation matrix, P, defined by indices, this computes
 * X * P
 */
template <typename PermutationIndicesType>
Eigen::MatrixXd from_right(const Eigen::MatrixXd &rhs,
                           const PermutationIndicesType &indices) {
  Eigen::MatrixXd output(rhs.rows(), rhs.cols());
  assert(indices.size() == rhs.rows() && "indices size must match matrix size");
  for (Eigen::Index i = 0; i < indices.size(); ++i) {
    assert(indices.coeff(i) >= 0 && "permuation indices must be non negative");
    assert(indices.coeff(i) < output.rows() &&
           "permuation indices exceed size of the matrix");
    output.col(i) = rhs.col(indices.coeff(i));
  }
  return output;
}

} // namespace permute
}

#endif