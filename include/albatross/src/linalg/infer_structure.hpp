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

#ifndef ALBATROSS_SRC_LINALG_INFER_STRUCTURE_HPP
#define ALBATROSS_SRC_LINALG_INFER_STRUCTURE_HPP

namespace albatross {

namespace linalg {

namespace detail {

using VectorXb = Eigen::Matrix<bool, Eigen::Dynamic, 1>;

void inline connected_indices(const Eigen::MatrixXd &x, const Eigen::Index &i,
                              VectorXb *visited,
                              std::set<Eigen::Index> *active_set) {
  assert(visited != nullptr);
  assert(active_set != nullptr);

  for (Eigen::Index j = 0; j < x.cols(); ++j) {
    if (x(i, j) != 0. && !(*visited)[j]) {
      active_set->insert(j);
      (*visited)[j] = true;
      connected_indices(x, j, visited, active_set);
    }
  }
}

} // namespace detail

std::vector<std::set<Eigen::Index>> inline infer_diagonal_blocks(
    const Eigen::MatrixXd &x) {
  assert(x.rows() == x.cols());
  if (x.rows() == 0) {
    return {};
  }
  std::vector<std::set<Eigen::Index>> connected_sets;
  detail::VectorXb visited(x.rows());
  visited.fill(false);

  for (Eigen::Index i = 0; i < x.rows(); ++i) {
    if (!visited[i]) {
      std::set<Eigen::Index> active_set = {i};
      detail::connected_indices(x, i, &visited, &active_set);
      connected_sets.emplace_back(active_set);
    }
  }

  return connected_sets;
}

Eigen::PermutationMatrix<Eigen::Dynamic> inline to_permutation_matrix(
    const std::vector<std::set<Eigen::Index>> &blocks) {
  Eigen::Index n = 0;
  for (const auto &block : blocks) {
    n += cast::to_index(block.size());
  }

  Eigen::VectorXi permutation_inds(n);
  int i = 0;
  for (const auto &block : blocks) {
    for (const auto &ind : block) {
      permutation_inds[ind] = i;
      ++i;
    }
  }
  return Eigen::PermutationMatrix<Eigen::Dynamic>(permutation_inds);
}

Structured<BlockDiagonal> inline infer_block_diagonal_matrix(
    const Eigen::MatrixXd &x) {
  const auto block_inds = infer_diagonal_blocks(x);
  const auto P = to_permutation_matrix(block_inds);

  Structured<BlockDiagonal> output;
  output.P_rows = P.transpose();
  output.P_cols = P;

  // D = P * x * P.T
  // x = P.T * D * P
  for (const auto &block : block_inds) {
    std::vector<Eigen::Index> inds(block.begin(), block.end());
    output.matrix.blocks.emplace_back(subset(x, inds, inds));
  }
  return output;
}

} // namespace linalg

} // namespace albatross

#endif // ALBATROSS_SRC_LINALG_INFER_STRUCTURE_HPP
