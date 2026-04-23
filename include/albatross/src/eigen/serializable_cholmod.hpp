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

#ifndef ALBATROSS_EIGEN_SERIALIZABLE_CHOLMOD_H
#define ALBATROSS_EIGEN_SERIALIZABLE_CHOLMOD_H

namespace Eigen {

template <typename _MatrixType, int _UpLo = Lower>
class SerializableCholmodSupernodalLLT
    : public CholmodSupernodalLLT<_MatrixType, _UpLo> {
  using Base = CholmodSupernodalLLT<_MatrixType, _UpLo>;

public:
  using Base::cols;
  using Base::rows;
  using MatrixType = _MatrixType;
  using StorageIndex = typename MatrixType::StorageIndex;
  using SolverBase = CholmodBase<_MatrixType, _UpLo, Base>;
  enum { UpLo = _UpLo };

  SerializableCholmodSupernodalLLT() : Base() {}

  // Public access to otherwise protected CHOLMOD state, for downstream
  // code (e.g. CholmodCovariance) that needs to call `cholmod_solve`,
  // `cholmod_spsolve`, or inspect the factor directly.  `m_cholmod` is
  // declared mutable in CholmodBase; Eigen's derived classes re-import
  // it into their private section with `using Base::m_cholmod`, so we
  // reach it through explicit CholmodBase qualification.  The const
  // accessors yield non-const pointers because CHOLMOD's solve API takes
  // non-const pointers but treats the factor / common as state, and
  // `m_cholmod` is mutable anyway.
  cholmod_factor *factorPtr() const {
    return this->SolverBase::m_cholmodFactor;
  }
  cholmod_common *cholmodCommonPtr() const {
    return &this->SolverBase::m_cholmod;
  }

  template <class Archive>
  void save(Archive &ar, const std::uint32_t version ALBATROSS_UNUSED) const;

  template <class Archive>
  void load(Archive &ar, const std::uint32_t version ALBATROSS_UNUSED);
};

template <typename _MatrixType, int _UpLo = Lower>
class SerializableCholmodSimplicialLLT
    : public CholmodSimplicialLLT<_MatrixType, _UpLo> {
  using Base = CholmodSimplicialLLT<_MatrixType, _UpLo>;

public:
  using Base::cols;
  using Base::rows;
  using MatrixType = _MatrixType;
  using StorageIndex = typename MatrixType::StorageIndex;
  using SolverBase = CholmodBase<_MatrixType, _UpLo, Base>;
  enum { UpLo = _UpLo };

  SerializableCholmodSimplicialLLT() : Base() {}

  cholmod_factor *factorPtr() const {
    return this->SolverBase::m_cholmodFactor;
  }
  cholmod_common *cholmodCommonPtr() const {
    return &this->SolverBase::m_cholmod;
  }

  template <class Archive>
  void save(Archive &ar, const std::uint32_t version ALBATROSS_UNUSED) const;

  template <class Archive>
  void load(Archive &ar, const std::uint32_t version ALBATROSS_UNUSED);
};

template <typename _MatrixType, int _UpLo = Lower>
class SerializableCholmodSimplicialLDLT
    : public CholmodSimplicialLDLT<_MatrixType, _UpLo> {
  using Base = CholmodSimplicialLDLT<_MatrixType, _UpLo>;

public:
  using Base::cols;
  using Base::rows;
  using MatrixType = _MatrixType;
  using StorageIndex = typename MatrixType::StorageIndex;
  using SolverBase = CholmodBase<_MatrixType, _UpLo, Base>;
  enum { UpLo = _UpLo };

  SerializableCholmodSimplicialLDLT() : Base() {}

  cholmod_factor *factorPtr() const {
    return this->SolverBase::m_cholmodFactor;
  }
  cholmod_common *cholmodCommonPtr() const {
    return &this->SolverBase::m_cholmod;
  }

  template <class Archive>
  void save(Archive &ar, const std::uint32_t version ALBATROSS_UNUSED) const;

  template <class Archive>
  void load(Archive &ar, const std::uint32_t version ALBATROSS_UNUSED);
};

} // namespace Eigen

#endif // ALBATROSS_EIGEN_SERIALIZABLE_CHOLMOD_H
