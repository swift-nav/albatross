/*
 * Copyright (C) 2022 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef ALBATROSS_BLOCK_SPARSE_MATRIX_H
#define ALBATROSS_BLOCK_SPARSE_MATRIX_H

namespace albatross {

inline std::vector<Eigen::Index>
sizes_to_starts(const std::vector<Eigen::Index> &sizes) {
  Eigen::Index start = 0;
  std::vector<Eigen::Index> starts;
  for (const auto &size : sizes) {
    starts.push_back(start);
    start += size;
  }
  return starts;
}

template <typename BlockType> class BlockSparseMatrix {
public:
  BlockSparseMatrix(const std::vector<Eigen::Index> &block_row_sizes,
                    const std::vector<Eigen::Index> &block_col_sizes)
      : row_sizes_(block_row_sizes),
        row_starts_(sizes_to_starts(block_row_sizes)),
        col_sizes_(block_col_sizes),
        col_starts_(sizes_to_starts(block_col_sizes)), blocks_() {}

  friend std::ostream &operator<<(std::ostream &s, const BlockSparseMatrix &m) {
    for (const auto &row : m.blocks_) {
      for (const auto &block : row.second) {
        s << "block[" << row.first << ", " << block.first
          << "] : " << std::endl;
        s << block.second << "    " << std::endl;
      }
      s << std::endl;
    }
    return s;
  }

  const BlockType &get_block(Eigen::Index i, Eigen::Index j) const {
    assert(i < row_sizes_.size());
    assert(i >= 0);
    assert(j < col_sizes_.size());
    assert(j >= 0);
    assert(block.rows() == row_sizes_[i]);
    assert(block.cols() == col_sizes_[j]);
    const auto row_iter = blocks_.find(i);
    assert(row_iter != blocks_.end());
    return row_iter->second[j];
  }

  template <typename Rhs>
  void set_block(Eigen::Index i, Eigen::Index j, const Rhs &block) {
    assert(i < row_sizes_.size());
    assert(i >= 0);
    assert(j < col_sizes_.size());
    assert(j >= 0);
    assert(block.rows() == row_sizes_[i]);
    assert(block.cols() == col_sizes_[j]);
    const auto row_iter = blocks_.find(i);
    if (row_iter == blocks_.end()) {
      blocks_[i] = {{j, block}};
    } else {
      row_iter->second[j] = block;
    }
  }

  Eigen::Index rows() const {
    return row_starts_[row_starts_.size() - 1] +
           row_sizes_[row_sizes_.size() - 1];
  }

  Eigen::Index cols() const {
    return col_starts_[col_starts_.size() - 1] +
           col_sizes_[col_sizes_.size() - 1];
  }

  template <typename Rhs,
            std::enable_if_t<is_eigen_plain_object<Rhs>::value, int> = 0>
  auto operator*(const Rhs &rhs) const {
    // decltype((std::declval<BlockType>() * rhs).eval()) output(rows(),
    //                                                           rhs.cols());
    Eigen::MatrixXd output = Eigen::MatrixXd::Zero(rows(), rhs.cols());

    assert(rhs.rows() == this->cols());

    for (const auto &row : blocks_) {
      const auto &i = row.first;
      const auto &si = cast::to_size(i);
      for (const auto &block : row.second) {
        const auto &j = block.first;
        const auto &sj = cast::to_size(j);
        output.block(row_starts_[si], 0, row_sizes_[si], rhs.cols()) +=
            block.second *
            rhs.block(col_starts_[sj], 0, col_sizes_[sj], rhs.cols());
      }
    }
    return output;
  }

private:
  std::vector<Eigen::Index> row_sizes_;
  std::vector<Eigen::Index> row_starts_;
  std::vector<Eigen::Index> col_sizes_;
  std::vector<Eigen::Index> col_starts_;
  std::map<Eigen::Index, std::map<Eigen::Index, BlockType>> blocks_;
};

} // namespace albatross

#endif