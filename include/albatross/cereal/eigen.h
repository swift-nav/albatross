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

    ar(rows);
    ar(cols);

    _Scalar *data = static_cast<_Scalar*>(std::malloc(size_in_bytes));
    Eigen::Map<Eigen::Matrix<_Scalar, _Rows, _Cols>>(data, m.rows(), m.cols()) = m;
    char *char_data = reinterpret_cast<char *>(data);

    const std::string compressed = gzip::compress(char_data, size_in_bytes);
    auto base64string = base64::encode(reinterpret_cast<const unsigned char *>(compressed.data()), compressed.size());

    ar(base64string);
  }

template <class Archive, class _Scalar, int _Rows, int _Cols> inline
void
  load(Archive & ar, Eigen::Matrix<_Scalar, _Rows, _Cols> & m, const std::uint32_t)
  {
    Eigen::Index rows;
    Eigen::Index cols;
    ar(rows);
    ar(cols);

    std::size_t size_in_bytes = static_cast<std::size_t>(rows * cols * sizeof(_Scalar));

    std::string base64string;
    ar(base64string);

    auto decoded = base64::decode(base64string);

    const std::string decompressed = gzip::decompress(decoded.data(), decoded.size());

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
  archive(v.indices());
}

} // namespace cereal

#endif
