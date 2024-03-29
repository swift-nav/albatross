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

#ifndef ALBATROSS_INDEXING_SUBSET_H
#define ALBATROSS_INDEXING_SUBSET_H

namespace albatross {

/*
 * Extract a subset of a standard vector.
 */
template <typename SizeType, typename X>
inline std::vector<X> subset(const std::vector<X> &v,
                             const std::vector<SizeType> &indices) {
  std::vector<X> out(indices.size());
  for (std::size_t i = 0; i < indices.size(); i++) {
    ALBATROSS_ASSERT(indices[i] >= 0 && "Invalid indices provided to subset");
    ALBATROSS_ASSERT(indices[i] < v.size() &&
                     "Invalid indices provided to subset");
    out[i] = v[static_cast<std::size_t>(indices[i])];
  }
  return out;
}

/*
 * Extract a subset of an Eigen::Vector
 */
template <typename SizeType>
inline Eigen::VectorXd subset(const Eigen::VectorXd &v,
                              const std::vector<SizeType> &indices) {
  Eigen::VectorXd out(cast::to_index(indices.size()));
  for (std::size_t i = 0; i < indices.size(); i++) {
    const Eigen::Index ind_i = static_cast<Eigen::Index>(indices[i]);
    ALBATROSS_ASSERT(ind_i >= 0 && "Invalid indices provided to subset");
    ALBATROSS_ASSERT(ind_i < v.size() && "Invalid indices provided to subset");
    out[cast::to_index(i)] = v[ind_i];
  }
  return out;
}

/*
 * Extracts a subset of columns from an Eigen::Matrix
 */
template <typename SizeType>
inline Eigen::MatrixXd subset_cols(const Eigen::MatrixXd &v,
                                   const std::vector<SizeType> &col_indices) {
  Eigen::MatrixXd out(v.rows(), cast::to_index(col_indices.size()));
  for (std::size_t i = 0; i < col_indices.size(); i++) {
    auto ii = cast::to_index(i);
    auto col_index = static_cast<Eigen::Index>(col_indices[i]);
    ALBATROSS_ASSERT(col_index >= 0 && "Invalid indices provided to subset");
    ALBATROSS_ASSERT(col_index < v.cols() &&
                     "Invalid indices provided to subset");
    out.col(ii) = v.col(col_index);
  }
  return out;
}

/*
 * Extracts a subset of rows from an Eigen::Matrix
 */
template <typename SizeType>
inline Eigen::MatrixXd subset_rows(const Eigen::MatrixXd &v,
                                   const std::vector<SizeType> &row_indices) {
  Eigen::MatrixXd out(cast::to_index(row_indices.size()), v.cols());
  for (std::size_t i = 0; i < row_indices.size(); i++) {
    auto ii = cast::to_index(i);
    auto row_index = static_cast<Eigen::Index>(row_indices[i]);
    ALBATROSS_ASSERT(row_index >= 0 && "Invalid indices provided to subset");
    ALBATROSS_ASSERT(row_index < v.rows() &&
                     "Invalid indices provided to subset");
    out.row(ii) = v.row(row_index);
  }
  return out;
}

/*
 * Extracts a subset of an Eigen::Matrix for the given row and column
 * indices.
 */
template <typename SizeType>
inline Eigen::MatrixXd subset(const Eigen::MatrixXd &v,
                              const std::vector<SizeType> &row_indices,
                              const std::vector<SizeType> &col_indices) {
  Eigen::MatrixXd out(cast::to_index(row_indices.size()),
                      cast::to_index(col_indices.size()));
  for (std::size_t i = 0; i < row_indices.size(); i++) {
    for (std::size_t j = 0; j < col_indices.size(); j++) {
      auto ii = cast::to_index(i);
      auto jj = cast::to_index(j);
      auto row_index = static_cast<Eigen::Index>(row_indices[i]);
      auto col_index = static_cast<Eigen::Index>(col_indices[j]);
      ALBATROSS_ASSERT(row_index >= 0 && "Invalid indices provided to subset");
      ALBATROSS_ASSERT(row_index < v.rows() &&
                       "Invalid indices provided to subset");
      ALBATROSS_ASSERT(col_index >= 0 && "Invalid indices provided to subset");
      ALBATROSS_ASSERT(col_index < v.cols() &&
                       "Invalid indices provided to subset");
      out(ii, jj) = v(row_index, col_index);
    }
  }
  return out;
}

/*
 * Takes a symmetric subset of an Eigen::Matrix.  Ie, it'll index the same rows
 * and columns.
 */
template <typename SizeType>
inline Eigen::MatrixXd symmetric_subset(const Eigen::MatrixXd &v,
                                        const std::vector<SizeType> &indices) {
  ALBATROSS_ASSERT(v.rows() == v.cols());
  return subset(v, indices, indices);
}

/*
 * Extract a subset of an Eigen::DiagonalMatrix
 */
template <typename SizeType, typename Scalar, int Size>
inline Eigen::DiagonalMatrix<Scalar, Size>
symmetric_subset(const Eigen::DiagonalMatrix<Scalar, Size> &v,
                 const std::vector<SizeType> &indices) {
  return subset(v.diagonal(), indices).asDiagonal();
}

/*
 * Set a subset of an Eigen::Vector.  If this worked it'd be
 * the equivalent of:
 *
 *     to[indices] = from;
 */
template <typename SizeType>
inline void set_subset(const Eigen::VectorXd &from,
                       const std::vector<SizeType> &indices,
                       Eigen::VectorXd *to) {
  ALBATROSS_ASSERT(cast::to_index(indices.size()) == from.size());
  for (std::size_t i = 0; i < indices.size(); ++i) {
    const Eigen::Index ind_i = static_cast<Eigen::Index>(indices[i]);
    ALBATROSS_ASSERT(ind_i >= 0 && "Invalid indices provided to subset");
    ALBATROSS_ASSERT(ind_i < to->size() &&
                     "Invalid indices provided to subset");
    (*to)[ind_i] = from[cast::to_index(i)];
  }
}

/*
 * Set a subset of an Eigen::Vector.  If this worked it'd be
 * the equivalent of:
 *
 *     to[indices] = from;
 */
template <typename SizeType, typename Scalar, int Size>
inline void set_subset(const Eigen::DiagonalMatrix<Scalar, Size> &from,
                       const std::vector<SizeType> &indices,
                       Eigen::DiagonalMatrix<Scalar, Size> *to) {
  ALBATROSS_ASSERT(cast::to_index(indices.size()) == from.diagonal().size());
  for (std::size_t i = 0; i < indices.size(); i++) {
    const Eigen::Index ind_i = static_cast<Eigen::Index>(indices[i]);
    ALBATROSS_ASSERT(ind_i >= 0 && "Invalid indices provided to subset");
    ALBATROSS_ASSERT(ind_i < to->size() &&
                     "Invalid indices provided to subset");
    to->diagonal()[ind_i] = from.diagonal()[cast::to_index(i)];
  }
}

template <typename X>
inline std::set<X> set_difference(const std::set<X> &x, const std::set<X> &y) {
  std::set<X> diff;
  std::set_difference(x.begin(), x.end(), y.begin(), y.end(),
                      std::inserter(diff, diff.begin()));
  return diff;
}

template <typename X>
inline std::set<X> vector_set_difference(const std::vector<X> &x,
                                         const std::vector<X> &y) {
  std::set<X> sorted_x(x.begin(), x.end());
  std::set<X> sorted_y(y.begin(), y.end());
  return set_difference(sorted_x, sorted_y);
}

/*
 * Computes the indices between 0 and n - 1 which are NOT contained
 * in `indices`.  Here complement is the mathematical interpretation
 * of the word meaning "the part required to make something whole".
 * In other words, indices and indices_complement(indices) should
 * contain all the numbers between 0 and n-1
 */
inline std::vector<std::size_t>
indices_complement(const std::vector<std::size_t> &indices,
                   const std::size_t n) {
  std::vector<std::size_t> all_indices(n);
  std::iota(all_indices.begin(), all_indices.end(), 0);
  const auto complement = vector_set_difference(all_indices, indices);
  return std::vector<std::size_t>(complement.begin(), complement.end());
}

template <typename GroupType>
inline GroupIndices indices_from_groups(const GroupIndexer<GroupType> &indexer,
                                        const std::vector<GroupType> &keys) {
  GroupIndices output;
  for (const auto &key : keys) {
    output.insert(output.end(), indexer.at(key).begin(), indexer.at(key).end());
  }
  return output;
}

} // namespace albatross

#endif
