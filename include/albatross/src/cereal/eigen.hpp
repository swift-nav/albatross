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

namespace cereal {

template <class Archive, class _Scalar, int _Rows, int _Cols>
inline void save(Archive &ar, Eigen::Matrix<_Scalar, _Rows, _Cols> const &m,
                 const std::uint32_t) {
  Eigen::Index rows = m.rows();
  Eigen::Index cols = m.cols();
  std::size_t size = albatross::cast::to_size(rows * cols);

  std::string payload = albatross::zstd::compress(m.data(), size);

  if (::cereal::traits::is_text_archive<Archive>::value) {
    payload =
        base64::encode(reinterpret_cast<const unsigned char *>(payload.data()),
                       payload.size());
  }

  ar(CEREAL_NVP(rows));
  ar(CEREAL_NVP(cols));
  ar(CEREAL_NVP(payload));
}

template <class Archive, class _Scalar, int _Rows, int _Cols>
inline void load(Archive &ar, Eigen::Matrix<_Scalar, _Rows, _Cols> &m,
                 const std::uint32_t) {
  Eigen::Index rows;
  Eigen::Index cols;
  std::string payload;

  ar(CEREAL_NVP(rows));
  ar(CEREAL_NVP(cols));
  ar(CEREAL_NVP(payload));

  std::size_t size = albatross::cast::to_size(rows * cols);

  if (size == 0) {
    m = Eigen::Matrix<_Scalar, _Rows, _Cols>(rows, cols);
    return;
  }

  if (::cereal::traits::is_text_archive<Archive>::value) {
    payload = base64::decode(payload);
  }

  m.resize(rows, cols);
  if (!albatross::zstd::maybe_decompress(payload, m.data(), size)) {
    const std::string decompressed =
        gzip::decompress(payload.data(), payload.size());
    ALBATROSS_ASSERT(decompressed.size() == size * sizeof(_Scalar));
    m = Eigen::Map<Eigen::Matrix<_Scalar, _Rows, _Cols>>(
        static_cast<_Scalar *>(
            static_cast<void *>(const_cast<char *>(decompressed.data()))),
        rows, cols);
  }
}

template <class Archive, int SizeAtCompileTime, int MaxSizeAtCompileTime,
          typename _StorageIndex>
inline void
save(Archive &archive,
     const Eigen::Transpositions<SizeAtCompileTime, MaxSizeAtCompileTime,
                                 _StorageIndex> &v,
     const std::uint32_t) {
  archive(cereal::make_nvp("indices", v.indices()));
}

template <class Archive, int SizeAtCompileTime, int MaxSizeAtCompileTime,
          typename _StorageIndex>
inline void load(Archive &archive,
                 Eigen::Transpositions<SizeAtCompileTime, MaxSizeAtCompileTime,
                                       _StorageIndex> &v,
                 const std::uint32_t) {
  typename Eigen::Transpositions<SizeAtCompileTime, MaxSizeAtCompileTime,
                                 _StorageIndex>::IndicesType indices;
  archive(cereal::make_nvp("indices", indices));
  v.indices() = indices;
}

template <typename Archive, typename _Scalar, int SizeAtCompileTime>
inline void serialize(Archive &archive,
                      Eigen::DiagonalMatrix<_Scalar, SizeAtCompileTime> &matrix,
                      const std::uint32_t) {
  archive(matrix.diagonal());
}

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
  const double a = 1.;
  const double b = 1.;
  const double c = -2. * albatross::cast::to_double(data.size());
  const double rows_as_double = (std::sqrt(b * b - 4 * a * c) - b) / (2 * a);
  ALBATROSS_ASSERT(
      rows_as_double - static_cast<double>(std::lround(rows_as_double)) == 0. &&
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

template <class Archive, class _Scalar, int _Options, typename _StorageIndex>
inline void save(Archive &ar,
                 Eigen::SparseMatrix<_Scalar, _Options, _StorageIndex> const &m,
                 const std::uint32_t version ALBATROSS_UNUSED) {
  const _StorageIndex rows = m.rows();
  const _StorageIndex cols = m.cols();
  const std::size_t nnz = albatross::cast::to_size(m.nonZeros());
  const std::size_t size_bytes = nnz * sizeof(_Scalar);

  std::vector<Eigen::Triplet<_Scalar>> elems;
  elems.reserve(nnz);
  for (_StorageIndex col = 0; col < m.outerSize(); ++col) {
    for (typename Eigen::SparseMatrix<_Scalar, _Options,
                                      _StorageIndex>::InnerIterator it(m, col);
         it; ++it) {
      elems.emplace_back(it.col(), it.row(), it.value());
    }
  }

  std::string payload =
      gzip::compress(reinterpret_cast<const char *>(elems.data()), size_bytes);
  if (::cereal::traits::is_text_archive<Archive>::value) {
    payload =
        base64::encode(reinterpret_cast<const unsigned char *>(payload.data()),
                       payload.size());
  }

  ar(CEREAL_NVP(rows));
  ar(CEREAL_NVP(cols));
  ar(CEREAL_NVP(nnz));
  ar(CEREAL_NVP(payload));
}

template <class Archive, class _Scalar, int _Options, typename _StorageIndex>
inline void load(Archive &ar,
                 Eigen::SparseMatrix<_Scalar, _Options, _StorageIndex> &m,
                 const std::uint32_t version ALBATROSS_UNUSED) {
  _StorageIndex rows = 0;
  _StorageIndex cols = 0;
  std::size_t nnz = 0;
  std::string payload;
  ar(CEREAL_NVP(rows));
  ar(CEREAL_NVP(cols));
  ar(CEREAL_NVP(nnz));
  ar(CEREAL_NVP(payload));
  const std::size_t size_bytes = nnz * sizeof(_Scalar);

  if (::cereal::traits::is_text_archive<Archive>::value) {
    payload = base64::decode(payload);
  }

  const std::string decompressed =
      gzip::decompress(payload.data(), payload.size());

  ALBATROSS_ASSERT(size_bytes == decompressed.size());
  m.resize(rows, cols);

  if (size_bytes == 0) {
    m.setZero();
  } else {
    std::vector<Eigen::Triplet<_Scalar>> elems(nnz);
    std::memcpy(elems.data(), decompressed.data(), size_bytes);
    m.setFromTriplets(elems.begin(), elems.end());
  }

  m.makeCompressed();
}

} // namespace cereal

#endif
