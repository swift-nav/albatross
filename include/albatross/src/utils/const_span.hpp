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

#ifndef ALBATROSS_UTILS_CONST_SPAN_HPP_
#define ALBATROSS_UTILS_CONST_SPAN_HPP_

namespace albatross {

/*
 * A minimal C++17 backport of std::span<const T>.
 *
 * Provides a non-owning view over a contiguous sequence of const T.
 * Implicitly constructible from std::vector<T> and supports zero-copy
 * sub-range slicing via subspan().
 */
template <typename T> class const_span {
  const T *data_;
  std::size_t size_;

public:
  const_span() : data_(nullptr), size_(0) {}
  const_span(const T *d, std::size_t n) : data_(d), size_(n) {}

  // NOLINTNEXTLINE(google-explicit-constructor)
  const_span(const std::vector<T> &v) : data_(v.data()), size_(v.size()) {}

  const_span subspan(std::size_t offset, std::size_t count) const {
    ALBATROSS_ASSERT(offset + count <= size_);
    return {data_ + offset, count};
  }

  const T &operator[](std::size_t i) const {
    ALBATROSS_ASSERT(i < size_);
    return data_[i];
  }

  std::size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }
  const T *data() const { return data_; }
  const T *begin() const { return data_; }
  const T *end() const { return data_ + size_; }
};

} // namespace albatross

#endif // ALBATROSS_UTILS_CONST_SPAN_HPP_
