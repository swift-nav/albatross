/*
 * Copyright (C) 2019 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef ALBATROSS_INDEXING_APPLY_HPP_
#define ALBATROSS_INDEXING_APPLY_HPP_

namespace albatross {

template <typename ToKeepFunction, typename ValueType,
          typename std::enable_if<details::is_valid_value_only_filter_function<
                                      ToKeepFunction, ValueType>::value,
                                  int>::type = 0>
inline auto filter(const std::vector<ValueType> &values,
                   const ToKeepFunction &to_keep) {
  std::vector<ValueType> output;
  for (const auto &v : values) {
    if (to_keep(v)) {
      output.emplace_back(v);
    }
  }
  return output;
}

} // namespace albatross

#endif /* ALBATROSS_INDEXING_APPLY_HPP_ */
