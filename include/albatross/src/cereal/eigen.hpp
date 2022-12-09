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
  std::size_t size_in_bytes = size * sizeof(_Scalar);

  // Turn the Eigen::Matrix in to an array of characters
  std::vector<_Scalar> data(size);
  Eigen::Map<Eigen::Matrix<_Scalar, _Rows, _Cols>>(data.data(), rows, cols) = m;
  char *char_data = reinterpret_cast<char *>(data.data());

  std::string payload = gzip::compress(char_data, size_in_bytes);

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
  std::size_t size_in_bytes = size * sizeof(_Scalar);

  if (::cereal::traits::is_text_archive<Archive>::value) {
    payload = base64::decode(payload);
  }

  const std::string decompressed =
      gzip::decompress(payload.data(), payload.size());

  ALBATROSS_ASSERT(size_in_bytes == decompressed.size());

  if (size_in_bytes == 0) {
    m = Eigen::Matrix<_Scalar, _Rows, _Cols>(rows, cols);
    return;
  }

  std::vector<_Scalar> decoded_data(size);
  std::memcpy(decoded_data.data(), decompressed.data(), size_in_bytes);

  m = Eigen::Map<Eigen::Matrix<_Scalar, _Rows, _Cols>>(decoded_data.data(),
                                                       rows, cols);
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

} // namespace cereal

#endif
