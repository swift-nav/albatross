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

namespace albatross {

namespace zstd {

namespace internal {

inline std::string compress(const void *data, std::size_t size_bytes,
                            int compression_level) {
  const std::size_t compressed_bound = ZSTD_compressBound(size_bytes);

  std::string compressed{};
  compressed.resize(compressed_bound);

  const std::size_t compressed_size =
      ZSTD_compress(
          static_cast<void *>(const_cast<char *>(compressed.data())), // NOLINT
          compressed.size(), data, size_bytes, compression_level);

  compressed.resize(compressed_size);
  compressed.shrink_to_fit();

  return compressed;
}

enum class DecompressResult {
  kOK,
  kZstdContentSizeUnknown,
  kZstdContentSizeError,
  kZstdExpectedRequestedSizeMismatch,
  kZstdResultingRequestedSizeMismatch
};

inline bool failed(DecompressResult result) {
  if (result == DecompressResult::kOK) {
    return false;
  }

  return true;
}

inline void assert_on_error(DecompressResult result) {
  switch (result) {
  case DecompressResult::kOK:
    return;
  case DecompressResult::kZstdContentSizeUnknown:
    ALBATROSS_ASSERT(
        false && "zstd couldn't determine decompressed size of input buffer");
    break;
  case DecompressResult::kZstdContentSizeError:
    ALBATROSS_ASSERT(
        false && "zstd error determining decompressed size of input buffer");
    break;
  case DecompressResult::kZstdExpectedRequestedSizeMismatch:
    ALBATROSS_ASSERT(
        false && "requested object size != zstd expected decompressed size");
    break;
  case DecompressResult::kZstdResultingRequestedSizeMismatch:
    ALBATROSS_ASSERT(
        false && "requested object size != zstd resulting decompressed size");
    break;
  default:
    ALBATROSS_ASSERT(
        false &&
        "albatross internal bug (unexpected decompression result code)");
    break;
  }
}

inline auto get_decompressed_string_size(const std::string &input) {
  return ZSTD_getFrameContentSize(static_cast<const void *>(input.data()),
                                  input.size());
}

inline auto check_decompressed_size(unsigned long long decompressed_size) {
  if (ZSTD_CONTENTSIZE_UNKNOWN == decompressed_size) {
    return DecompressResult::kZstdContentSizeUnknown;
  }

  if (ZSTD_CONTENTSIZE_ERROR == decompressed_size) {
    return DecompressResult::kZstdContentSizeError;
  }

  return DecompressResult::kOK;
}

inline DecompressResult __attribute__((warn_unused_result))
maybe_decompress(const std::string &input, void *output,
                 std::size_t output_size) {
  const auto decompressed_size = get_decompressed_string_size(input);

  const auto check_result = check_decompressed_size(decompressed_size);
  if (failed(check_result)) {
    return check_result;
  }

  if (decompressed_size != output_size) {
    return DecompressResult::kZstdExpectedRequestedSizeMismatch;
  }

  const std::size_t result_size =
      ZSTD_decompress(output, output_size,
                      static_cast<const void *>(input.data()), input.size());

  if (result_size != output_size) {
    return DecompressResult::kZstdResultingRequestedSizeMismatch;
  }

  return DecompressResult::kOK;
}

inline void decompress(const std::string &input, void *output,
                       std::size_t output_size) {
  assert_on_error(maybe_decompress(input, output, output_size));
}

} // namespace internal

constexpr int kDefaultCompressionLevel = ZSTD_CLEVEL_DEFAULT;

// Compress the first `n_elements` pointed to by `data` using `zstd`.
template <typename Element>
inline std::string compress(const Element *data, std::size_t n_elements,
                            int compression_level = kDefaultCompressionLevel) {
  return internal::compress(static_cast<const void *>(data),
                            n_elements * sizeof(Element), compression_level);
}

// Compress the contents of the string (treated as binary data) using
// `zstd`.
inline std::string compress(const std::string &input,
                            int compression_level = kDefaultCompressionLevel) {
  return internal::compress(static_cast<const void *>(input.data()),
                            input.size(), compression_level);
}

// Decompress the `zstd`-compressed data in `input` into `data`.
// `n_elements` must match the number of elements actually contained
// in the compressed data, or an assertion will be triggered.
template <typename Element>
inline void decompress(const std::string &input, Element *data,
                       std::size_t n_elements) {
  return internal::decompress(input, static_cast<void *>(data),
                              n_elements * sizeof(Element));
}

// Decompress the `zstd`-compressed data in `input` into `output`.
// Returns `true` on successful decompression, in which case `output`
// has been populated / overwritten.  If decompression fails, `output`
// may or may not have been modified.
template <typename Element>
inline bool __attribute__((warn_unused_result))
maybe_decompress(const std::string &input, Element *output,
                 std::size_t n_elements) {
  const auto result = internal::maybe_decompress(
      input, static_cast<void *>(output), sizeof(Element) * n_elements);

  if (internal::failed(result)) {
    return false;
  }

  return true;
}
// Decompress the `zstd`-compressed data in `input` into `output`.
// Returns `true` on successful decompression, in which case `output`
// has been populated / overwritten.  If decompression fails, `output`
// may or may not have been modified.
inline bool __attribute__((warn_unused_result))
maybe_decompress(const std::string &input, std::string *output) {
  const auto decompressed_size = internal::get_decompressed_string_size(input);

  if (internal::failed(internal::check_decompressed_size(decompressed_size))) {
    return false;
  }

  output->resize(decompressed_size);
  const auto result = internal::maybe_decompress(
      input, const_cast<void *>(static_cast<const void *>(output->data())),
      decompressed_size);

  if (internal::failed(result)) {
    return false;
  }

  return true;
}

// Decompress the `zstd`-compressed data in `input` into `output`.  If
// anything goes wrong, an assertion will be triggered.
inline std::string decompress(const std::string &input) {
  std::string output;
  const auto decompressed_size = internal::get_decompressed_string_size(input);
  assert_on_error(internal::check_decompressed_size(decompressed_size));

  output.resize(decompressed_size);
  internal::decompress(
      input, const_cast<void *>(static_cast<const void *>(output.data())),
      decompressed_size);
  return output;
}

} // namespace zstd

} // namespace albatross

#endif // ALBATROSS_UTILS_COMPRESS_H
