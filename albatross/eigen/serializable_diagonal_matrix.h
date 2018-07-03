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

#ifndef ALBATROSS_EIGEN_SERIALIZABLE_DIAGONAL_MATRIX_H
#define ALBATROSS_EIGEN_SERIALIZABLE_DIAGONAL_MATRIX_H

/*
 * The Eigen::DiagonalMatrix doesn't provide the public methods
 * required to reliably serialize the `m_diagonal` private
 * member.  In order to make the DiagonalMatrix serializable
 * we instead inherit from it, giving private access to the
 * diagonal elements which in turn allows us to serialize it.
 */

#include "Eigen/Cholesky"
#include "Eigen/Dense"
#include "cereal/cereal.hpp"
#include <math.h>

namespace Eigen {

template <typename _Scalar, int SizeAtCompileTime>
class SerializableDiagonalMatrix
    : public Eigen::DiagonalMatrix<_Scalar, SizeAtCompileTime> {
  using BaseClass = Eigen::DiagonalMatrix<_Scalar, SizeAtCompileTime>;

public:
  SerializableDiagonalMatrix() : BaseClass(){};

  template <typename OtherDerived>
  inline SerializableDiagonalMatrix(const DiagonalBase<OtherDerived> &other)
      : BaseClass(other){};

  template <typename Archive> void serialize(Archive &archive) {
    archive(cereal::make_nvp("diagonal", this->m_diagonal));
  }

  bool operator==(const BaseClass &other) const {
    return (this->m_diagonal == other.diagonal());
  }
};

} // namesapce Eigen

#endif
