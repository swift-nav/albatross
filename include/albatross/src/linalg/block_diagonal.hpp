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

#ifndef ALBATROSS_SRC_LINALG_BLOCK_DIAGONAL_HPP
#define ALBATROSS_SRC_LINALG_BLOCK_DIAGONAL_HPP

namespace albatross {

template <typename MatrixType, unsigned int Mode = Eigen::Lower>
struct BlockTriangularView;

struct BlockDiagonalLDLT;
struct BlockDiagonal;

struct BlockDiagonalLDLT {
  using RealScalar = Eigen::SerializableLDLT::RealScalar;
  using Scalar = Eigen::SerializableLDLT::Scalar;
  using MatrixType = Eigen::SerializableLDLT::MatrixType;
  std::vector<Eigen::SerializableLDLT> blocks;

  template <typename Derived>
  inline Eigen::MatrixXd solve(const Eigen::MatrixBase<Derived> &rhs) const;

  template <typename Derived>
  inline Eigen::MatrixXd
  sqrt_solve(const Eigen::MatrixBase<Derived> &rhs) const;

  template <class _Scalar, int _Options, typename _StorageIndex>
  Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic>
  solve(const Eigen::SparseMatrix<_Scalar, _Options, _StorageIndex> &rhs) const;

  template <class _Scalar, int _Options, typename _StorageIndex>
  Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic> sqrt_solve(
      const Eigen::SparseMatrix<_Scalar, _Options, _StorageIndex> &rhs) const;

  template <class _Scalar, int _Rows, int _Cols>
  Eigen::Matrix<_Scalar, _Rows, _Cols>
  solve(const Eigen::Matrix<_Scalar, _Rows, _Cols> &rhs,
        ThreadPool *pool) const;

  template <typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
  inline Eigen::MatrixXd
  solve(const Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel> &rhs_orig)
      const;

  template <class _Scalar, int _Rows, int _Cols>
  Eigen::Matrix<_Scalar, _Rows, _Cols>
  sqrt_solve(const Eigen::Matrix<_Scalar, _Rows, _Cols> &rhs,
             ThreadPool *pool) const;

  const BlockDiagonalLDLT &adjoint() const;

  std::map<size_t, Eigen::Index> block_to_row_map() const;

  double log_determinant() const;

  double rcond() const;

  Eigen::Index rows() const;

  Eigen::Index cols() const;

  bool operator==(const BlockDiagonalLDLT &other) const;
};

struct BlockDiagonal {
  std::vector<Eigen::MatrixXd> blocks;

  template <class _Scalar, int _Rows, int _Cols>
  Eigen::Matrix<_Scalar, _Rows, _Cols>
  operator*(const Eigen::Matrix<_Scalar, _Rows, _Cols> &rhs) const;

  BlockDiagonal operator-(const BlockDiagonal &rhs) const;

  Eigen::VectorXd diagonal() const;

  BlockDiagonalLDLT ldlt() const;

  Eigen::Index rows() const;

  Eigen::Index cols() const;

  Eigen::MatrixXd toDense() const;
};

/*
 * BlockDiagonalLDLT
 */
template <typename Derived>
inline Eigen::MatrixXd
BlockDiagonalLDLT::solve(const Eigen::MatrixBase<Derived> &rhs) const {
  ALBATROSS_ASSERT(cols() == rhs.rows());
  Eigen::Index i = 0;
  Eigen::MatrixXd output(rows(), rhs.cols());
  for (const auto &b : blocks) {
    const auto rhs_chunk = rhs.block(i, 0, b.rows(), rhs.cols());
    output.block(i, 0, b.rows(), rhs.cols()) = b.solve(rhs_chunk);
    i += b.rows();
  }
  return output;
}

template <typename Derived>
inline Eigen::MatrixXd
BlockDiagonalLDLT::sqrt_solve(const Eigen::MatrixBase<Derived> &rhs) const {
  ALBATROSS_ASSERT(cols() == rhs.rows());
  Eigen::Index i = 0;
  Eigen::MatrixXd output(rows(), rhs.cols());
  for (const auto &b : blocks) {
    const auto rhs_chunk = rhs.block(i, 0, b.rows(), rhs.cols());
    output.block(i, 0, b.rows(), rhs.cols()) = b.sqrt_solve(rhs_chunk);
    i += b.rows();
  }
  return output;
}

template <class _Scalar, int _Options, typename _StorageIndex>
inline Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic>
BlockDiagonalLDLT::solve(
    const Eigen::SparseMatrix<_Scalar, _Options, _StorageIndex> &rhs) const {
  ALBATROSS_ASSERT(cols() == rhs.rows());
  Eigen::Index i = 0;
  Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic> output(rows(),
                                                                rhs.cols());
  for (const auto &b : blocks) {
    const auto rhs_chunk = rhs.block(i, 0, b.rows(), rhs.cols());
    output.block(i, 0, b.rows(), rhs.cols()) = b.solve(rhs_chunk);
    i += b.rows();
  }
  return output;
}

template <typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
inline Eigen::MatrixXd BlockDiagonalLDLT::solve(
    const Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel> &rhs_orig)
    const {
  ALBATROSS_ASSERT(cols() == rhs_orig.rows());
  Eigen::MatrixXd rhs{rhs_orig};
  return solve(rhs);
}

template <class _Scalar, int _Options, typename _StorageIndex>
inline Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic>
BlockDiagonalLDLT::sqrt_solve(
    const Eigen::SparseMatrix<_Scalar, _Options, _StorageIndex> &rhs) const {
  ALBATROSS_ASSERT(cols() == rhs.rows());
  Eigen::Index i = 0;
  Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic> output(rows(),
                                                                rhs.cols());
  for (const auto &b : blocks) {
    const auto rhs_chunk = rhs.block(i, 0, b.rows(), rhs.cols());
    output.block(i, 0, b.rows(), rhs.cols()) = b.sqrt_solve(rhs_chunk);
    i += b.rows();
  }
  return output;
}

inline std::map<size_t, Eigen::Index>
BlockDiagonalLDLT::block_to_row_map() const {
  Eigen::Index row = 0;
  std::map<size_t, Eigen::Index> block_to_row;

  for (size_t i = 0; i < blocks.size(); ++i) {
    block_to_row[i] = row;
    row += blocks[i].rows();
  }
  ALBATROSS_ASSERT(row == cols());

  return block_to_row;
}

template <class _Scalar, int _Rows, int _Cols>
inline Eigen::Matrix<_Scalar, _Rows, _Cols>
BlockDiagonalLDLT::solve(const Eigen::Matrix<_Scalar, _Rows, _Cols> &rhs,
                         ThreadPool *pool) const {
  ALBATROSS_ASSERT(cols() == rhs.rows());
  Eigen::Matrix<_Scalar, _Rows, _Cols> output(rows(), rhs.cols());
  auto solve_and_fill_one_block = [&](const size_t i, const Eigen::Index row) {
    const auto rhs_chunk = rhs.block(row, 0, blocks[i].rows(), rhs.cols());
    output.block(row, 0, blocks[i].rows(), rhs.cols()) =
        blocks[i].solve(rhs_chunk);
  };

  apply_map(block_to_row_map(), solve_and_fill_one_block, pool);
  return output;
}

template <class _Scalar, int _Rows, int _Cols>
inline Eigen::Matrix<_Scalar, _Rows, _Cols>
BlockDiagonalLDLT::sqrt_solve(const Eigen::Matrix<_Scalar, _Rows, _Cols> &rhs,
                              ThreadPool *pool) const {
  ALBATROSS_ASSERT(cols() == rhs.rows());
  Eigen::Matrix<_Scalar, _Rows, _Cols> output(rows(), rhs.cols());

  auto solve_and_fill_one_block = [&](const size_t i, const Eigen::Index row) {
    const auto rhs_chunk = rhs.block(row, 0, blocks[i].rows(), rhs.cols());
    output.block(row, 0, blocks[i].rows(), rhs.cols()) =
        blocks[i].sqrt_solve(rhs_chunk);
  };

  apply_map(block_to_row_map(), solve_and_fill_one_block, pool);
  return output;
}

inline double BlockDiagonalLDLT::log_determinant() const {
  double output = 0.;
  for (const auto &b : blocks) {
    output += b.log_determinant();
  }
  return output;
}

inline double BlockDiagonalLDLT::rcond() const {
  // L1 induced norm is just the maximum absolute column sum.
  // Therefore the L1 induced norm of a block-diagonal matrix is the
  // maximum of the L1 induced norms of the individual blocks.
  double l1_norm = -INFINITY;
  for (const auto &b : blocks) {
    l1_norm = std::max(l1_norm, b.l1_norm());
  }
  return Eigen::internal::rcond_estimate_helper(l1_norm, *this);
}

inline Eigen::Index BlockDiagonalLDLT::rows() const {
  Eigen::Index n = 0;
  for (const auto &b : blocks) {
    n += b.rows();
  }
  return n;
}

inline Eigen::Index BlockDiagonalLDLT::cols() const {
  Eigen::Index n = 0;
  for (const auto &b : blocks) {
    n += b.cols();
  }
  return n;
}

inline const BlockDiagonalLDLT &BlockDiagonalLDLT::adjoint() const {
  return *this;
}

inline bool
BlockDiagonalLDLT::operator==(const BlockDiagonalLDLT &other) const {
  return blocks == other.blocks;
}

/*
 * Block Diagonal
 */

template <class _Scalar, int _Rows, int _Cols>
inline Eigen::Matrix<_Scalar, _Rows, _Cols> BlockDiagonal::operator*(
    const Eigen::Matrix<_Scalar, _Rows, _Cols> &rhs) const {
  ALBATROSS_ASSERT(cols() == rhs.rows());
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
  ALBATROSS_ASSERT(cols() == rhs.rows());
  ALBATROSS_ASSERT(blocks.size() == rhs.blocks.size());

  BlockDiagonal output;
  for (std::size_t i = 0; i < blocks.size(); ++i) {
    ALBATROSS_ASSERT(blocks[i].size() == rhs.blocks[i].size());
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
  ALBATROSS_ASSERT(rows() == cols());
  Eigen::VectorXd output(rows());

  Eigen::Index i = 0;
  for (const auto &b : blocks) {
    output.block(i, 0, b.rows(), 1) = b.diagonal();
    i += b.rows();
  }
  return output;
}

inline BlockDiagonalLDLT BlockDiagonal::ldlt() const {
  BlockDiagonalLDLT output;
  for (const auto &b : blocks) {
    output.blocks.emplace_back(b.ldlt());
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

} // namespace albatross

#endif // ALBATROSS_SRC_LINALG_BLOCK_DIAGONAL_HPP
