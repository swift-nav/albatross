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

#ifndef ALBATROSS_CEREAL_SERIALIZABLE_SPQR_H
#define ALBATROSS_CEREAL_SERIALIZABLE_SPQR_H

namespace Eigen {

template <typename _MatrixType>
template <class Archive>
void SerializableSPQR<_MatrixType>::save(
    Archive &ar, const std::uint32_t version ALBATROSS_UNUSED) const {
  ar(CEREAL_NVP(Base::m_isInitialized));
  ar(CEREAL_NVP(Base::m_analysisIsOk));
  ar(CEREAL_NVP(Base::m_factorizationIsOk));
  ar(CEREAL_NVP(Base::m_isRUpToDate));
  ar(CEREAL_NVP(Base::m_info));
  ar(CEREAL_NVP(Base::m_ordering));
  ar(CEREAL_NVP(Base::m_allow_tol));
  ar(CEREAL_NVP(Base::m_tolerance));
  ar(CEREAL_NVP(Base::m_R));
  ar(CEREAL_NVP(Base::m_rank));
  ar(CEREAL_NVP(Base::m_cc));
  ar(CEREAL_NVP(Base::m_useDefaultThreshold));
  ar(CEREAL_NVP(Base::m_rows));

  const std::size_t integer_size_bytes = sizeof(StorageIndex);

  ar(::cereal::make_nvp("m_cR", *this->m_cR));
  ar(::cereal::make_nvp("m_H", *this->m_H));
  ar(::cereal::make_nvp("m_HTau", *this->m_HTau));
  encode_array(ar, "m_E", this->m_E,
               albatross::cast::to_size(cols()) * integer_size_bytes);
  encode_array(ar, "m_HPinv", this->m_HPinv,
               albatross::cast::to_size(rows()) * integer_size_bytes);
}

template <typename _MatrixType>
template <class Archive>
void SerializableSPQR<_MatrixType>::load(
    Archive &ar, const std::uint32_t version ALBATROSS_UNUSED) {
  ar(CEREAL_NVP(Base::m_isInitialized));
  ar(CEREAL_NVP(Base::m_analysisIsOk));
  ar(CEREAL_NVP(Base::m_factorizationIsOk));
  ar(CEREAL_NVP(Base::m_isRUpToDate));
  ar(CEREAL_NVP(Base::m_info));
  ar(CEREAL_NVP(Base::m_ordering));
  ar(CEREAL_NVP(Base::m_allow_tol));
  ar(CEREAL_NVP(Base::m_tolerance));
  ar(CEREAL_NVP(Base::m_R));
  ar(CEREAL_NVP(Base::m_rank));
  ar(CEREAL_NVP(Base::m_cc));
  ar(CEREAL_NVP(Base::m_useDefaultThreshold));
  ar(CEREAL_NVP(Base::m_rows));

  const std::size_t integer_size_bytes = sizeof(StorageIndex);
  this->SPQR_free();
  this->m_cR =
      cholmod_l_allocate_sparse(1, 1, 1, 1, 1, 0, CHOLMOD_REAL, &this->m_cc);
  ar(::cereal::make_nvp("m_cR", *this->m_cR));
  this->m_H =
      cholmod_l_allocate_sparse(1, 1, 1, 1, 1, 0, CHOLMOD_REAL, &this->m_cc);
  ar(::cereal::make_nvp("m_H", *this->m_H));
  this->m_HTau = cholmod_l_allocate_dense(1, 1, 1, CHOLMOD_REAL, &this->m_cc);
  ar(::cereal::make_nvp("m_HTau", *this->m_HTau));
  this->m_E = static_cast<StorageIndex *>(cholmod_l_malloc(
      albatross::cast::to_size(cols()), integer_size_bytes, &this->m_cc));
  decode_array(ar, "m_E", this->m_E,
               albatross::cast::to_size(cols()) * integer_size_bytes);
  this->m_HPinv = static_cast<StorageIndex *>(cholmod_l_malloc(
      albatross::cast::to_size(rows()), integer_size_bytes, &this->m_cc));
  decode_array(ar, "m_HPinv", this->m_HPinv,
               albatross::cast::to_size(rows()) * integer_size_bytes);
}

} // namespace Eigen

#endif // ALBATROSS_CEREAL_SERIALIZABLE_SPQR_H
