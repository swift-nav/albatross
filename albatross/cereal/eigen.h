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

#ifndef ALBATROSS_CEREAL_EIGEN_H
#define ALBATROSS_CEREAL_EIGEN_H

#include "Eigen/Dense"
#include "cereal/cereal.hpp"
#include <cereal/types/map.hpp>
#include <cereal/types/vector.hpp>

namespace cereal {

/*
 * The subsequent save/load methods catch the serialization methods
 * for arbitrary Eigen::Matrix* types.  The general idea is that each
 * row is saved as a Eigen::Vector.
 */
template <class Archive, typename _Scalar, int _Rows, int _Cols>
inline void save(Archive &archive,
                 const Eigen::Matrix<_Scalar, _Rows, _Cols> &v) {
  size_type rows = static_cast<size_type>(v.rows());
  archive(cereal::make_size_tag(rows));
  for (size_type i = 0; i < rows; i++) {
    Eigen::Matrix<_Scalar, _Cols, 1> row = v.row(i);
    archive(row);
  }
};

template <class Archive, typename _Scalar, int _Rows, int _Cols>
inline void load(Archive &archive, Eigen::Matrix<_Scalar, _Rows, _Cols> &v) {
  size_type rows;
  archive(cereal::make_size_tag(rows));
  /*
   * In order to determine the size of a matrix, we have to first determine
   * how many rows, then inspect the size of the first row to get the
   * number of columns.
   */
  if (rows > 0) {
    Eigen::Matrix<_Scalar, _Rows, 1> first;
    archive(first);
    size_type cols = first.rows();
    v.resize(rows, cols);
    v.row(0) = first;

    for (size_type i = 1; i < rows; i++) {
      Eigen::Matrix<_Scalar, _Cols, 1> row;
      archive(row);
      v.row(i) = row;
    }
  } else {
    // Serialized matrix is empty.
    v.resize(0, 0);
  }
};

/*
 * The subsequent save/load methods catch the serialization methods
 * for arbitrary Eigen::Vector* types through template specialization.
 * In this case each scalar value is serialized.
 */
template <class Archive, typename _Scalar, int _Rows>
inline void save(Archive &archive, const Eigen::Matrix<_Scalar, _Rows, 1> &v) {
  size_type rows = static_cast<size_type>(v.rows());
  archive(cereal::make_size_tag(rows));
  for (size_type i = 0; i < rows; i++) {
    archive(v(i));
  }
};

template <class Archive, typename _Scalar, int _Rows>
inline void load(Archive &archive, Eigen::Matrix<_Scalar, _Rows, 1> &v) {
  size_type rows;
  archive(cereal::make_size_tag(rows));
  v.resize(rows);
  for (size_type i = 0; i < rows; i++) {
    archive(v(i));
  }
};

template <class Archive, int SizeAtCompileTime, int MaxSizeAtCompileTime,
          typename _StorageIndex>
inline void
serialize(Archive &archive,
          Eigen::Transpositions<SizeAtCompileTime, MaxSizeAtCompileTime,
                                _StorageIndex> &v) {
  archive(v.indices());
}

} // namespace cereal

#endif
