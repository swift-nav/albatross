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

template <typename MatrixType = MatrixXd>
class SerializableLDLT : public LDLT<MatrixType, Lower> {
public:
  SerializableLDLT() : LDLT<MatrixType, Lower>(){};

  SerializableLDLT(const LDLT<MatrixType, Lower> &ldlt)
      : LDLT<MatrixType, Lower>(ldlt){};

  template <typename Archive> void save(Archive &archive) const {
    save_lower_triangle(archive, this->m_matrix);
    archive(this->m_transpositions, this->m_isInitialized);
  }

  template <typename Archive> void load(Archive &archive) {
    load_lower_triangle(archive, this->m_matrix);
    archive(this->m_transpositions, this->m_isInitialized);
  }

  bool operator==(const SerializableLDLT &rhs) const {
    // Make sure the two lower triangles are the same and that
    // any permutations are identical.
    auto this_lower =
        MatrixXd(MatrixXd(this->matrixLDLT()).triangularView<Eigen::Lower>());
    auto rhs_lower =
        MatrixXd(MatrixXd(rhs.matrixLDLT()).triangularView<Eigen::Lower>());
    return (this_lower == rhs_lower &&
            this->transpositionsP().indices() ==
                rhs.transpositionsP().indices());
  }
};

} // namesapce Eigen

#endif
