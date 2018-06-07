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

template <class Archive, typename _Scalar, int _Rows, int _Cols>
inline void load_lower_triangle(Archive &archive,
                                Eigen::Matrix<_Scalar, _Rows, _Cols> &v) {
  cereal::size_type storage_size;
  archive(cereal::make_size_tag(storage_size));
  // TODO: understand why the storage size upon load is always augmented by two.
  storage_size -= 2;
  // We assume the matrix is square and compute the number of rows from the
  // storage size.
  double drows = (std::sqrt(static_cast<double>(storage_size * 8 + 1)) - 1) / 2;
  cereal::size_type rows = static_cast<cereal::size_type>(drows);
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

  /*
   * The diagonal of the inverse of the matrix this LDLT
   * decomposition represents in O(n^2) operations.
   */
  Eigen::VectorXd inverse_diagonal() const {
    Eigen::Index n = this->rows();
    Eigen::MatrixXd inverse_cholesky =
        this->transpositionsP() * Eigen::MatrixXd::Identity(n, n);
    this->matrixL().solveInPlace(inverse_cholesky);

    Eigen::VectorXd sqrt_diag = this->vectorD();
    for (Eigen::Index i = 0; i < n; i++) {
      sqrt_diag[i] = 1. / std::sqrt(sqrt_diag[i]);
    }

    Eigen::VectorXd inv_diag(n);
    for (Eigen::Index i = 0; i < n; i++) {
      const Eigen::VectorXd col_i =
          inverse_cholesky.col(i).cwiseProduct(sqrt_diag);
      ;
      inv_diag[i] = col_i.dot(col_i);
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
