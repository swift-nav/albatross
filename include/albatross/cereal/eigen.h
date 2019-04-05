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
#include <gzip/compress.hpp>
#include <gzip/decompress.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/vector.hpp>

namespace cereal {

template <class Archive, class _Scalar, int _Rows, int _Cols>
inline
void
  save(Archive & ar, Eigen::Matrix<_Scalar, _Rows, _Cols> const & m, const std::uint32_t )
  {
    Eigen::Index rows = m.rows();
    Eigen::Index cols = m.cols();
    std::size_t size_in_bytes = static_cast<std::size_t>(rows * cols * sizeof(_Scalar));

    // Turn the Eigen::Matrix in to an array of characters
    _Scalar *data = static_cast<_Scalar*>(std::malloc(size_in_bytes));
    Eigen::Map<Eigen::Matrix<_Scalar, _Rows, _Cols>>(data, rows, cols) = m;
    char *char_data = reinterpret_cast<char *>(data);

    std::string payload = gzip::compress(char_data, size_in_bytes);

    if (::cereal::traits::is_text_archive<Archive>::value) {
      payload = base64::encode(reinterpret_cast<const unsigned char *>(payload.data()),
                               payload.size());
    }

    ar(CEREAL_NVP(rows));
    ar(CEREAL_NVP(cols));
    ar(CEREAL_NVP(payload));
  }

template <class Archive, class _Scalar, int _Rows, int _Cols> inline
void
  load(Archive & ar, Eigen::Matrix<_Scalar, _Rows, _Cols> & m, const std::uint32_t)
  {
    Eigen::Index rows;
    Eigen::Index cols;
    std::string payload;

    ar(CEREAL_NVP(rows));
    ar(CEREAL_NVP(cols));
    ar(CEREAL_NVP(payload));

    std::size_t size_in_bytes = static_cast<std::size_t>(rows * cols * sizeof(_Scalar));

    if (::cereal::traits::is_text_archive<Archive>::value) {
      payload = base64::decode(payload);
    }

    const std::string decompressed = gzip::decompress(payload.data(), payload.size());

    assert(size_in_bytes == decompressed.size());

    _Scalar *decoded_data = static_cast<_Scalar*>(std::malloc(size_in_bytes));
    std::memcpy(decoded_data, decompressed.data(), size_in_bytes);

    m = Eigen::Map<Eigen::Matrix<_Scalar, _Rows, _Cols>>(decoded_data, rows, cols);
  }

template <class Archive, int SizeAtCompileTime, int MaxSizeAtCompileTime,
          typename _StorageIndex>
inline void
serialize(Archive &archive,
          Eigen::Transpositions<SizeAtCompileTime, MaxSizeAtCompileTime,
                                _StorageIndex> &v,
          const std::uint32_t) {
  archive(cereal::make_nvp("indices", v.indices()));
}

} // namespace cereal

#endif
