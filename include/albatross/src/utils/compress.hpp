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

#ifndef ALBATROSS_UTILS_COMPRESS_H
#define ALBATROSS_UTILS_COMPRESS_H

#include <string>

#include <blosc2.h>
#include <blosc2/filters-registry.h>
#include <zfp.h>
#include <zstd.h>

namespace albatross {

inline void *string_data_to_void(std::string *string) {
  return static_cast<void *>(const_cast<char *>(string->data()));   // NOLINT
}

inline void *string_data_to_void(const std::string *string) {
  return static_cast<void *>(const_cast<char *>(string->data()));   // NOLINT
}

inline const void *string_data_to_const_void(std::string *string) {
  return static_cast<const void *>(string->data());
}

inline const void *string_data_to_const_void(const std::string *string) {
  return static_cast<const void *>(string->data());
}

namespace zstd {

inline std::string compress(const void *data, std::size_t size_bytes,
                            int compression_level = ZSTD_CLEVEL_DEFAULT) {
  const std::size_t compressed_bound = ZSTD_compressBound(size_bytes);

  std::string compressed{};
  compressed.resize(compressed_bound);

  const std::size_t compressed_size =
      ZSTD_compress(string_data_to_void(&compressed), compressed.size(), data,
                    size_bytes, compression_level);

  compressed.resize(compressed_size);
  compressed.shrink_to_fit();

  return compressed;
}

inline std::string compress_string(const std::string &input) {
  return compress(static_cast<const void *>(input.data()), input.size());
}

inline void decompress(const std::string &input, void *output,
                       std::size_t output_size) {
  const auto decompressed_size = ZSTD_getFrameContentSize(
      static_cast<const void *>(input.data()), input.size());

  if (ZSTD_CONTENTSIZE_UNKNOWN == decompressed_size ||
      ZSTD_CONTENTSIZE_ERROR == decompressed_size) {
    ALBATROSS_ASSERT(false &&
                     "zstd error getting decompressed size from input buffer.");
  }

  ALBATROSS_ASSERT(decompressed_size == output_size &&
                   "Requested object size != incoming zstd decompressed size.");

  const std::size_t result_size = ZSTD_decompress(
      output, output_size, string_data_to_const_void(&input), input.size());

  ALBATROSS_ASSERT(
      result_size == output_size &&
      "Requested object size != resulting zstd decompressed size.");
}

inline std::string decompress(const std::string &input) {
  std::string output;
  const auto decompressed_size = ZSTD_getFrameContentSize(
      static_cast<const void *>(input.data()), input.size());

  if (ZSTD_CONTENTSIZE_UNKNOWN == decompressed_size ||
      ZSTD_CONTENTSIZE_ERROR == decompressed_size) {
    ALBATROSS_ASSERT(false &&
                     "zstd error getting decompressed size from input buffer.");
  }
  decompress(input,
             const_cast<void *>(static_cast<const void *>(output.data())),
             decompressed_size);
  return output;
}

}  // namespace zstd

namespace zfp {

namespace internal {

inline std::string compress(const void *data, std::size_t rows,
                            std::size_t columns, zfp_type element_type) {
  zfp_field *field =
      zfp_field_2d(const_cast<void *>(data), element_type, rows, columns);
  zfp_stream *stream = zfp_stream_open(nullptr);
  zfp_stream_set_reversible(stream);

  std::string compressed{};
  compressed.resize(zfp_stream_maximum_size(stream, field));
  bitstream *bitstream =
      stream_open(string_data_to_void(&compressed), compressed.size());
  zfp_stream_set_bit_stream(stream, bitstream);
  zfp_stream_rewind(stream);
  const std::size_t compressed_size = zfp_compress(stream, field);
  ALBATROSS_ASSERT(compressed_size > 0 && "error compressing with zfp");
  compressed.resize(compressed_size);
  compressed.shrink_to_fit();
  return compressed;
}

inline void decompress(const std::string &input, void *output,
                       std::size_t output_rows, std::size_t output_columns,
                       zfp_type element_type) {
  zfp_field *field =
      zfp_field_2d(output, element_type, output_rows, output_columns);
  zfp_stream *stream = zfp_stream_open(nullptr);
  zfp_stream_set_reversible(stream);
  bitstream *bitstream = stream_open(string_data_to_void(&input), input.size());
  zfp_stream_set_bit_stream(stream, bitstream);
  zfp_stream_rewind(stream);
  const std::size_t decompressed_size = zfp_decompress(stream, field);
  ALBATROSS_ASSERT(decompressed_size > 0 && "error decompressing with zfp");

  const std::size_t expected_size_bytes =
      output_rows * output_columns * zfp_type_size(element_type);
  ALBATROSS_ASSERT(decompressed_size == expected_size_bytes &&
                   "zfp decompression size did not match expected size");
}

}  // namespace internal

inline std::string compress(const double *data, std::size_t rows,
                            std::size_t columns) {
  return internal::compress(static_cast<const void *>(data), rows, columns,
                            zfp_type_double);
}

inline std::string compress(const std::int32_t *data, std::size_t rows,
                            std::size_t columns) {
  return internal::compress(static_cast<const void *>(data), rows, columns,
                            zfp_type_int32);
}

inline std::string compress(const std::int64_t *data, std::size_t rows,
                            std::size_t columns) {
  return internal::compress(static_cast<const void *>(data), rows, columns,
                            zfp_type_int64);
}

inline void decompress(const std::string &input, double *output,
                       std::size_t output_rows, std::size_t output_columns) {
  internal::decompress(input, static_cast<void *>(output), output_rows,
                       output_columns, zfp_type_double);
}

inline void decompress(const std::string &input, std::int32_t *output,
                       std::size_t output_rows, std::size_t output_columns) {
  internal::decompress(input, static_cast<void *>(output), output_rows,
                       output_columns, zfp_type_int32);
}

inline void decompress(const std::string &input, std::int64_t *output,
                       std::size_t output_rows, std::size_t output_columns) {
  internal::decompress(input, static_cast<void *>(output), output_rows,
                       output_columns, zfp_type_int64);
}

}  // namespace zfp

namespace blosc2 {

namespace internal {

constexpr std::size_t kBloscBufferPad = 64;

inline std::string compress(const void *data, std::size_t n_elements,
                            std::size_t element_size, std::uint8_t compression_level,
                            std::int16_t n_threads) {
  ALBATROSS_ASSERT(nullptr != data && "must provide some data to compress");
  blosc2_schunk schunk;
  schunk.typesize = static_cast<int>(element_size);
  blosc2_cparams compress_params = BLOSC2_CPARAMS_DEFAULTS;
  compress_params.typesize = static_cast<int>(element_size);
  compress_params.compcode = BLOSC_ZSTD;
  compress_params.clevel = compression_level;
  compress_params.nthreads = n_threads;
  compress_params.schunk = &schunk;
  // compress_params.filters[BLOSC2_MAX_FILTERS - 2] = BLOSC_SHUFFLE;
  // compress_params.filters[BLOSC2_MAX_FILTERS - 1] = BLOSC_FILTER_BYTEDELTA;
  compress_params.filters[BLOSC2_MAX_FILTERS - 1] = BLOSC_BITSHUFFLE;

  const std::size_t size_bytes = n_elements * element_size;
  std::string compressed{};
  compressed.resize(size_bytes + kBloscBufferPad);
  blosc2_context *context = blosc2_create_cctx(compress_params);
  const auto compressed_size = blosc2_compress_ctx(
      context, data, static_cast<int>(size_bytes),
      string_data_to_void(&compressed), static_cast<int>(compressed.size()));
  blosc2_free_ctx(context);

  ALBATROSS_ASSERT(compressed_size >= 0 && "error compressing with blosc2");
  ALBATROSS_ASSERT(compressed_size > 0 &&
                   "not enough room to compress with blosc2");
  compressed.resize(albatross::cast::to_size(compressed_size));
  compressed.shrink_to_fit();
  return compressed;
}

inline void decompress(const std::string &input, void *output,
                       std::size_t n_elements, std::size_t element_size,
                       std::int16_t n_threads) {
  ALBATROSS_ASSERT(nullptr != output &&
                   "must provide a buffer into which to decompress");
  blosc2_schunk schunk;
  schunk.typesize = static_cast<int>(element_size);
  blosc2_dparams decompress_params = BLOSC2_DPARAMS_DEFAULTS;
  decompress_params.schunk = &schunk;
  decompress_params.nthreads = n_threads;
  blosc2_context *context = blosc2_create_dctx(decompress_params);
  const std::size_t expected_size_bytes = n_elements * element_size;
  const auto decompressed_size =
      blosc2_decompress_ctx(context, string_data_to_const_void(&input),
                            static_cast<int>(input.size()), output,
                            static_cast<int>(expected_size_bytes));
  blosc2_free_ctx(context);
  ALBATROSS_ASSERT(albatross::cast::to_size(decompressed_size) ==
                       expected_size_bytes &&
                   "error decompressing with blosc");
}

}  // namespace internal

constexpr int kDefaultCompressionLevel = 3;
constexpr int kDefaultNumThreads = 1;

template <typename Element>
inline std::string compress(
    const Element *data, std::size_t rows, std::size_t cols,
    std::uint8_t compression_level = kDefaultCompressionLevel,
    std::int16_t n_threads = kDefaultNumThreads) {
  return internal::compress(static_cast<const void *>(data), rows * cols,
                            sizeof(Element), compression_level, n_threads);
}

template <typename Element>
inline void decompress(const std::string &input, Element *output,
                       std::size_t rows, std::size_t cols,
                       std::int16_t n_threads = kDefaultNumThreads) {
  internal::decompress(input, static_cast<void *>(output), rows * cols,
                       sizeof(Element), n_threads);
}

}  // namespace blosc2

namespace blosc {

inline void init() {
  blosc2_init();
  blosc1_set_compressor(BLOSC_ZSTD_COMPNAME);
}

inline void cleanup() { blosc2_destroy(); }

namespace internal {

constexpr std::size_t kBloscBufferPad = 64;

inline std::string compress(const void *data, std::size_t n_elements,
                            std::size_t element_size, int compression_level) {
  ALBATROSS_ASSERT(nullptr != data);
  const std::size_t size_bytes = n_elements * element_size;
  std::string compressed{};
  compressed.resize(size_bytes + kBloscBufferPad);
  const int compressed_size = blosc1_compress(
      compression_level, BLOSC_BITSHUFFLE, element_size, size_bytes, data,
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      const_cast<void *>(static_cast<const void *>(compressed.data())),
      compressed.size());
  ALBATROSS_ASSERT(compressed_size >= 0 && "error compressing with blosc");
  compressed.resize(albatross::cast::to_size(compressed_size));
  compressed.shrink_to_fit();
  return compressed;
}

inline void decompress(const std::string &input, void *output,
                       std::size_t n_elements, std::size_t element_size) {
  ALBATROSS_ASSERT(nullptr != output);
  const std::size_t expected_size_bytes = n_elements * element_size;
  const int decompressed_size = blosc1_decompress(
      static_cast<const void *>(input.data()), output, expected_size_bytes);
  ALBATROSS_ASSERT(albatross::cast::to_size(decompressed_size) ==
                       expected_size_bytes &&
                   "error decompressing with blosc");
}

}  // namespace internal

constexpr int kDefaultCompressionLevel = 1;

inline std::string compress(const double *data, std::size_t rows,
                            std::size_t cols,
                            int compression_level = kDefaultCompressionLevel) {
  return internal::compress(static_cast<const void *>(data), rows * cols,
                            sizeof(double), compression_level);
}

inline void decompress(const std::string &input, double *output,
                       std::size_t rows, std::size_t cols) {
  internal::decompress(input, static_cast<void *>(output), rows * cols,
                       sizeof(double));
}

inline std::string compress(const std::int64_t *data, std::size_t rows,
                            std::size_t cols,
                            int compression_level = kDefaultCompressionLevel) {
  return internal::compress(static_cast<const void *>(data), rows * cols,
                            sizeof(std::int64_t), compression_level);
}

inline void decompress(const std::string &input, std::int64_t *output,
                       std::size_t rows, std::size_t cols) {
  internal::decompress(input, static_cast<void *>(output), rows * cols,
                       sizeof(std::int64_t));
}

inline std::string compress(const std::int32_t *data, std::size_t rows,
                            std::size_t cols,
                            int compression_level = kDefaultCompressionLevel) {
  return internal::compress(static_cast<const void *>(data), rows * cols,
                            sizeof(std::int32_t), compression_level);
}

inline void decompress(const std::string &input, std::int32_t *output,
                       std::size_t rows, std::size_t cols) {
  internal::decompress(input, static_cast<void *>(output), rows * cols,
                       sizeof(std::int32_t));
}

}  // namespace blosc

}  // namespace albatross

#endif  // ALBATROSS_UTILS_COMPRESS_H
