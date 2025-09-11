/*
 * Copyright (C) 2023 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef ALBATROSS_EIGEN_SERIALIZABLE_SPQR_H
#define ALBATROSS_EIGEN_SERIALIZABLE_SPQR_H

namespace Eigen {

template <typename _MatrixType>
class SerializableSPQR : public SPQR<_MatrixType> {
  using Base = SPQR<_MatrixType>;

public:
  using Base::cols;
  using Base::rows;
  using StorageIndex = typename Base::StorageIndex;
  enum { ColsAtCompileTime = Dynamic, MaxColsAtCompileTime = Dynamic };
  SerializableSPQR() : Base() {
    // Eigen does not initialise these, but it will happily call
    // `SPQR_free()` on itself in its destructor.  Try initializing
    // one and destructing it without passing it a matrix.
    this->m_cR = nullptr;
    this->m_E = nullptr;
    this->m_H = nullptr;
    this->m_HPinv = nullptr;
    this->m_HTau = nullptr;
  }

  template <class Archive>
  void save(Archive &ar, const std::uint32_t version ALBATROSS_UNUSED) const;

  template <class Archive>
  void load(Archive &ar, const std::uint32_t version ALBATROSS_UNUSED);
};

} // namespace Eigen

#endif // ALBATROSS_EIGEN_SERIALIZABLE_SPQR_H
