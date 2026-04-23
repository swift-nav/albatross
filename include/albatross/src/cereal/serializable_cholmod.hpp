/*
 * Copyright (C) 2026 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef ALBATROSS_CEREAL_SERIALIZABLE_CHOLMOD_H
#define ALBATROSS_CEREAL_SERIALIZABLE_CHOLMOD_H

// The three serializable-cholmod wrappers all share the same persistent
// state: a couple of flags inherited from SparseSolverBase / CholmodBase,
// the numeric shift, the cholmod_common configuration, and the factor
// itself.  They differ only in what CHOLMOD mode flags the derived ctor's
// init() sets -- that's captured by the `cholmod_common` save/load.
//
// Because the relevant fields are protected in the CHOLMOD bases, the
// save/load bodies have to live inside the derived class, so they're
// spelled out once per variant via these macros.
// Base-qualified access bypasses the `using Base::m_cholmod` declaration
// that Eigen's Cholmod derived classes have in their private section; the
// underlying CholmodBase member is protected so our derived-class
// save/load can reach it with explicit qualification.
#define ALBATROSS_SERIALIZABLE_CHOLMOD_SAVE_BODY()                             \
  const bool is_initialized = this->SolverBase::m_isInitialized;               \
  ar(::cereal::make_nvp("m_isInitialized", is_initialized));                   \
  if (!is_initialized) {                                                       \
    return;                                                                    \
  }                                                                            \
  ar(::cereal::make_nvp("m_factorizationIsOk",                                 \
                        this->SolverBase::m_factorizationIsOk));               \
  ar(::cereal::make_nvp("m_analysisIsOk", this->SolverBase::m_analysisIsOk));  \
  ar(::cereal::make_nvp("m_info", this->SolverBase::m_info));                  \
  ar(::cereal::make_nvp("m_shiftOffset0",                                      \
                        this->SolverBase::m_shiftOffset[0]));                  \
  ar(::cereal::make_nvp("m_shiftOffset1",                                      \
                        this->SolverBase::m_shiftOffset[1]));                  \
  ar(::cereal::make_nvp("m_cholmod", this->SolverBase::m_cholmod));            \
  ALBATROSS_ASSERT(this->SolverBase::m_cholmodFactor != nullptr);              \
  ar(::cereal::make_nvp("m_cholmodFactor", *this->SolverBase::m_cholmodFactor));

#define ALBATROSS_SERIALIZABLE_CHOLMOD_LOAD_BODY()                             \
  bool is_initialized = false;                                                 \
  ar(::cereal::make_nvp("m_isInitialized", is_initialized));                   \
  if (this->SolverBase::m_cholmodFactor != nullptr) {                          \
    cholmod_free_factor(&this->SolverBase::m_cholmodFactor,                    \
                        &this->SolverBase::m_cholmod);                         \
    this->SolverBase::m_cholmodFactor = nullptr;                               \
  }                                                                            \
  if (!is_initialized) {                                                       \
    this->SolverBase::m_isInitialized = false;                                 \
    this->SolverBase::m_factorizationIsOk = 0;                                 \
    this->SolverBase::m_analysisIsOk = 0;                                      \
    return;                                                                    \
  }                                                                            \
  ar(::cereal::make_nvp("m_factorizationIsOk",                                 \
                        this->SolverBase::m_factorizationIsOk));               \
  ar(::cereal::make_nvp("m_analysisIsOk", this->SolverBase::m_analysisIsOk));  \
  ar(::cereal::make_nvp("m_info", this->SolverBase::m_info));                  \
  ar(::cereal::make_nvp("m_shiftOffset0",                                      \
                        this->SolverBase::m_shiftOffset[0]));                  \
  ar(::cereal::make_nvp("m_shiftOffset1",                                      \
                        this->SolverBase::m_shiftOffset[1]));                  \
  ar(::cereal::make_nvp("m_cholmod", this->SolverBase::m_cholmod));            \
  this->SolverBase::m_cholmodFactor =                                          \
      cholmod_allocate_factor(1, &this->SolverBase::m_cholmod);                \
  ALBATROSS_ASSERT(this->SolverBase::m_cholmodFactor != nullptr);              \
  ar(::cereal::make_nvp("m_cholmodFactor",                                     \
                        *this->SolverBase::m_cholmodFactor));                  \
  this->SolverBase::m_isInitialized = true;

namespace Eigen {

template <typename _MatrixType, int _UpLo>
template <class Archive>
void SerializableCholmodSupernodalLLT<_MatrixType, _UpLo>::save(
    Archive &ar, const std::uint32_t version ALBATROSS_UNUSED) const {
  ALBATROSS_SERIALIZABLE_CHOLMOD_SAVE_BODY();
}

template <typename _MatrixType, int _UpLo>
template <class Archive>
void SerializableCholmodSupernodalLLT<_MatrixType, _UpLo>::load(
    Archive &ar, const std::uint32_t version ALBATROSS_UNUSED) {
  ALBATROSS_SERIALIZABLE_CHOLMOD_LOAD_BODY();
}

template <typename _MatrixType, int _UpLo>
template <class Archive>
void SerializableCholmodSimplicialLLT<_MatrixType, _UpLo>::save(
    Archive &ar, const std::uint32_t version ALBATROSS_UNUSED) const {
  ALBATROSS_SERIALIZABLE_CHOLMOD_SAVE_BODY();
}

template <typename _MatrixType, int _UpLo>
template <class Archive>
void SerializableCholmodSimplicialLLT<_MatrixType, _UpLo>::load(
    Archive &ar, const std::uint32_t version ALBATROSS_UNUSED) {
  ALBATROSS_SERIALIZABLE_CHOLMOD_LOAD_BODY();
}

template <typename _MatrixType, int _UpLo>
template <class Archive>
void SerializableCholmodSimplicialLDLT<_MatrixType, _UpLo>::save(
    Archive &ar, const std::uint32_t version ALBATROSS_UNUSED) const {
  ALBATROSS_SERIALIZABLE_CHOLMOD_SAVE_BODY();
}

template <typename _MatrixType, int _UpLo>
template <class Archive>
void SerializableCholmodSimplicialLDLT<_MatrixType, _UpLo>::load(
    Archive &ar, const std::uint32_t version ALBATROSS_UNUSED) {
  ALBATROSS_SERIALIZABLE_CHOLMOD_LOAD_BODY();
}

} // namespace Eigen

#undef ALBATROSS_SERIALIZABLE_CHOLMOD_SAVE_BODY
#undef ALBATROSS_SERIALIZABLE_CHOLMOD_LOAD_BODY

#endif // ALBATROSS_CEREAL_SERIALIZABLE_CHOLMOD_H
