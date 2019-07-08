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

struct BlockDiagonalLLT;
struct BlockDiagonal;

struct BlockDiagonalLLT {
  std::vector<Eigen::LLT<Eigen::MatrixXd>> blocks;

  template <class _Scalar, int _Rows, int _Cols>
  Eigen::Matrix<_Scalar, _Rows, _Cols>
  solve(const Eigen::Matrix<_Scalar, _Rows, _Cols> &rhs) const;

  BlockDiagonal matrixL() const;

  Eigen::Index rows() const;

  Eigen::Index cols() const;
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

struct BlockSymmetric {

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
   */

  BlockSymmetric(){};

  BlockSymmetric(const Eigen::SerializableLDLT &A_, const Eigen::MatrixXd &B_,
                 const Eigen::SerializableLDLT &S_)
      : A(A_), Ai_B(A_.solve(B_)), S(S_) {}

  BlockSymmetric(const Eigen::SerializableLDLT &A_, const Eigen::MatrixXd &B_,
                 const Eigen::MatrixXd &C)
      : BlockSymmetric(
            A_, B_,
            Eigen::SerializableLDLT(C - B_.transpose() * A_.solve(B_))){};

  /*
   * https://en.wikipedia.org/wiki/Block_matrix#Block_matrix_inversion
   */
  template <class _Scalar, int _Rows, int _Cols>
  Eigen::Matrix<_Scalar, _Rows, _Cols>
  solve(const Eigen::Matrix<_Scalar, _Rows, _Cols> &rhs) const;

  bool operator==(const BlockSymmetric &rhs) const;

  template <typename Archive>
  void serialize(Archive &archive, const std::uint32_t);

  Eigen::SerializableLDLT A;
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

inline BlockDiagonal BlockDiagonalLLT::matrixL() const {
  BlockDiagonal output;
  for (const auto &b : blocks) {
    Eigen::MatrixXd L = b.matrixL();
    output.blocks.push_back(L);
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

template <class _Scalar, int _Rows, int _Cols>
inline Eigen::Matrix<_Scalar, _Rows, _Cols>
BlockSymmetric::solve(const Eigen::Matrix<_Scalar, _Rows, _Cols> &rhs) const {
  // https://en.wikipedia.org/wiki/Block_matrix#Block_matrix_inversion
  Eigen::Index n = A.rows() + S.rows();
  assert(rhs.rows() == n);

  const auto rhs_a = rhs.topRows(A.rows());
  const auto rhs_b = rhs.bottomRows(S.rows());

  const auto Bt_Ai_rhs = Ai_B.transpose() * rhs_a;

  const auto Si_Bt_Ai_rhs = S.solve(Bt_Ai_rhs);
  const auto upper_left = A.solve(rhs_a) + Ai_B * Si_Bt_Ai_rhs;

  Eigen::Matrix<_Scalar, _Rows, _Cols> output(n, rhs.cols());
  output.topRows(A.rows()) = upper_left - Ai_B * S.solve(rhs_b);
  output.bottomRows(S.rows()) = S.solve(rhs_b) - Si_Bt_Ai_rhs;

  return output;
}

inline bool BlockSymmetric::operator==(const BlockSymmetric &rhs) const {
  return (A == rhs.A && Ai_B == rhs.Ai_B && S == rhs.S);
}

template <typename Archive>
inline void BlockSymmetric::serialize(Archive &archive, const std::uint32_t) {
  archive(cereal::make_nvp("A", A), cereal::make_nvp("Ai_B", Ai_B),
          cereal::make_nvp("S", S));
}

} // namespace albatross

#endif /* INCLUDE_ALBATROSS_UTILS_BLOCK_UTILS_H_ */
