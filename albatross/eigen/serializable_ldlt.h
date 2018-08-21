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

#include "Eigen/Cholesky"
#include "Eigen/Dense"
#include "cereal/cereal.hpp"
#include "core/indexing.h"
#include <math.h>

namespace Eigen {

template <class Archive, typename _Scalar, int _Rows, int _Cols>
inline void save_lower_triangle(Archive &archive,
                                const Eigen::Matrix<_Scalar, _Rows, _Cols> &v) {
  cereal::size_type rows = static_cast<cereal::size_type>(v.rows());
  cereal::size_type storage_size = (rows * rows + rows) / 2;
  archive(cereal::make_size_tag(storage_size));
  for (cereal::size_type i = 0; i < rows; i++) {
    for (cereal::size_type j = 0; j <= i; j++) {
      archive(v(i, j));
    }
  }
}

inline void adjust_storage_size(__attribute__((unused))
                                cereal::JSONInputArchive &archive,
                                cereal::size_type *storage_size) {
  // TODO: understand why the storage size upon load is always augmented by two
  // when using a JSON archive.
  *storage_size -= 2;
}

template <class Archive>
inline void adjust_storage_size(__attribute__((unused)) Archive &archive,
                                __attribute__((unused))
                                cereal::size_type *storage_size) {}

template <class Archive, typename _Scalar, int _Rows, int _Cols>
inline void load_lower_triangle(Archive &archive,
                                Eigen::Matrix<_Scalar, _Rows, _Cols> &v) {
  cereal::size_type storage_size;
  archive(cereal::make_size_tag(storage_size));
  adjust_storage_size(archive, &storage_size);
  // We assume the matrix is square and compute the number of rows from the
  // storage size using the quadratic formula.
  //     rows^2 + rows - 2 * storage_size = 0
  double a = 1;
  double b = 1;
  double c = -2. * static_cast<double>(storage_size);
  double rows_as_double = (std::sqrt(b * b - 4 * a * c) - b) / (2 * a);
  assert(rows_as_double - static_cast<Eigen::Index>(rows_as_double) == 0. &&
         "inferred a non integer number of rows");
  cereal::size_type rows = static_cast<cereal::size_type>(rows_as_double);
  v.resize(rows, rows);
  for (cereal::size_type i = 0; i < rows; i++) {
    for (cereal::size_type j = 0; j <= i; j++) {
      archive(v(i, j));
    }
  }
}

class SerializableLDLT : public LDLT<MatrixXd, Lower> {
public:
  SerializableLDLT() : LDLT<MatrixXd, Lower>(){};

  SerializableLDLT(const LDLT<MatrixXd, Lower> &ldlt)
      // Can we get around copying here?
      : LDLT<MatrixXd, Lower>(ldlt){};

  template <typename Archive> void save(Archive &archive) const {
    save_lower_triangle(archive, this->m_matrix);
    archive(this->m_transpositions, this->m_isInitialized);
  }

  template <typename Archive> void load(Archive &archive) {
    load_lower_triangle(archive, this->m_matrix);
    archive(this->m_transpositions, this->m_isInitialized);
  }

  std::vector<Eigen::MatrixXd>
  inverse_blocks(const std::vector<albatross::FoldIndices> &blocks) const {

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

    std::vector<Eigen::MatrixXd> output;
    for (const auto &block_indices : blocks) {
      Eigen::MatrixXd sub_matrix =
          albatross::subset_cols(block_indices, inverse_cholesky);
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
    std::vector<albatross::FoldIndices> block_indices(size_n);
    for (Eigen::Index i = 0; i < n; i++) {
      block_indices[i] = {static_cast<albatross::FoldIndices::value_type>(i)};
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

} // namesapce Eigen

#endif
