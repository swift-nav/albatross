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

#ifndef ALBATROSS_EIGEN_SERIALIZABLE_LDLT_H
#define ALBATROSS_EIGEN_SERIALIZABLE_LDLT_H

namespace Eigen {

template <class Archive, typename _Scalar, int _Rows, int _Cols>
inline void save_lower_triangle(Archive &archive,
                                const Eigen::Matrix<_Scalar, _Rows, _Cols> &v) {
  Eigen::Index storage_size = (v.rows() * v.rows() + v.rows()) / 2;
  Eigen::VectorXd data(storage_size);

  Eigen::Index cnt = 0;
  for (Eigen::Index i = 0; i < v.rows(); i++) {
    for (Eigen::Index j = 0; j <= i; j++) {
      data[cnt++] = v(i, j);
    }
  }
  archive(cereal::make_nvp("lower_triangle", data));
}

template <class Archive, typename _Scalar, int _Rows, int _Cols>
inline void load_lower_triangle(Archive &archive,
                                Eigen::Matrix<_Scalar, _Rows, _Cols> &v) {

  Eigen::VectorXd data;
  archive(cereal::make_nvp("lower_triangle", data));
  // We assume the matrix is square and compute the number of rows from the
  // storage size using the quadratic formula.
  //     rows^2 + rows - 2 * storage_size = 0
  double a = 1;
  double b = 1;
  double c = -2. * static_cast<double>(data.size());
  double rows_as_double = (std::sqrt(b * b - 4 * a * c) - b) / (2 * a);
  assert(rows_as_double -
                 static_cast<Eigen::Index>(std::lround(rows_as_double)) ==
             0. &&
         "inferred a non integer number of rows");
  Eigen::Index rows = static_cast<Eigen::Index>(std::lround(rows_as_double));

  v.resize(rows, rows);
  Eigen::Index cnt = 0;
  for (Eigen::Index i = 0; i < rows; i++) {
    for (Eigen::Index j = 0; j <= i; j++) {
      v(i, j) = data[cnt++];
    }
  }
}

class SerializableLDLT : public LDLT<MatrixXd, Lower> {
public:
  SerializableLDLT() : LDLT<MatrixXd, Lower>(){};

  SerializableLDLT(const LDLT<MatrixXd, Lower> &ldlt)
      // Can we get around copying here?
      : LDLT<MatrixXd, Lower>(ldlt){};

  template <typename Archive>
  void save(Archive &archive, const std::uint32_t) const {
    save_lower_triangle(archive, this->m_matrix);
    archive(cereal::make_nvp("transpositions", this->m_transpositions),
            cereal::make_nvp("is_initialized", this->m_isInitialized));
  }

  template <typename Archive> void load(Archive &archive, const std::uint32_t) {
    load_lower_triangle(archive, this->m_matrix);
    archive(cereal::make_nvp("transpositions", this->m_transpositions),
            cereal::make_nvp("is_initialized", this->m_isInitialized));
  }

  std::vector<Eigen::MatrixXd>
  inverse_blocks(const std::vector<std::vector<std::size_t>> &blocks) const {

    /*
     * The LDLT decomposition is stored such that,
     *
     *     A = P^T LDL^T P
     *
     * We first need to compute the inverse of the cholesky, R^-1 such that
     *
     *     A^{-1} = R^-T R^-1
     *     R^{-1} = D^1/2 L^-1 P
     *
     * we can then pull out sub blocks of A^{-1} by dot products of
     * corresponding
     * columns of R^{-1}.
     */
    Eigen::Index n = this->matrixL().rows();

    // P
    Eigen::MatrixXd inverse_cholesky =
        this->transpositionsP() * Eigen::MatrixXd::Identity(n, n);
    // L^-1 P
    this->matrixL().solveInPlace(inverse_cholesky);
    // D^-1/2 L^-1 P
    const auto sqrt_diag =
        this->vectorD().array().sqrt().inverse().matrix().asDiagonal();
    inverse_cholesky = sqrt_diag * inverse_cholesky;

    assert(!inverse_cholesky.hasNaN());

    std::vector<Eigen::MatrixXd> output;
    for (const auto &block_indices : blocks) {
      Eigen::MatrixXd sub_matrix =
          albatross::subset_cols(inverse_cholesky, block_indices);
      Eigen::MatrixXd one_block =
          sub_matrix.transpose().lazyProduct(sub_matrix);
      output.push_back(one_block);
    }
    return output;
  }

  /*
   * The diagonal of the inverse of the matrix this LDLT
   * decomposition represents in O(n^2) operations.
   */
  Eigen::VectorXd inverse_diagonal() const {
    Eigen::Index n = this->rows();

    std::size_t size_n = static_cast<std::size_t>(n);
    std::vector<std::vector<std::size_t>> block_indices(size_n);
    for (Eigen::Index i = 0; i < n; i++) {
      block_indices[i] = {static_cast<std::size_t>(i)};
    }

    Eigen::VectorXd inv_diag(n);
    const auto blocks = inverse_blocks(block_indices);
    for (std::size_t i = 0; i < size_n; i++) {
      assert(blocks[i].rows() == 1);
      assert(blocks[i].cols() == 1);
      inv_diag[i] = blocks[i](0, 0);
    }

    return inv_diag;
  }

  bool operator==(const SerializableLDLT &rhs) const {
    // Make sure the two lower triangles are the same and that
    // any permutations are identical.
    if (!this->m_isInitialized && !rhs.m_isInitialized) {
      return true;
    }
    auto this_lower =
        MatrixXd(MatrixXd(this->matrixLDLT()).triangularView<Eigen::Lower>());
    auto rhs_lower =
        MatrixXd(MatrixXd(rhs.matrixLDLT()).triangularView<Eigen::Lower>());

    return (this->m_isInitialized == rhs.m_isInitialized &&
            this_lower == rhs_lower &&
            this->transpositionsP().indices() ==
                rhs.transpositionsP().indices());
  }
};

} // namespace Eigen

#endif
