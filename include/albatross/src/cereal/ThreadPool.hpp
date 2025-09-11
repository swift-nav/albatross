/*
 * Copyright (C) 2022 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef ALBATROSS_CEREAL_THREADPOOL_H
#define ALBATROSS_CEREAL_THREADPOOL_H

namespace cereal {

// ThreadPool objects are serialized by simply recording how many
// threads they contain and creating a new thread pool object with the
// deserialized thread count.

template <typename Archive>
inline void save(Archive &archive, const std::shared_ptr<ThreadPool> &pool) {
  std::size_t n_workers = 0;
  if (nullptr != pool) {
    n_workers = pool->thread_count();
  }
  archive(n_workers);
}

template <typename Archive>
inline void load(Archive &archive, std::shared_ptr<ThreadPool> &pool) {
  std::size_t n_workers = 0;
  archive(n_workers);
  if (n_workers == 0 || n_workers == 1) {
    pool = nullptr;
  } else {
    pool = std::make_shared<ThreadPool>(n_workers);
  }
}

}  // namespace cereal

#endif  // ALBATROSS_CEREAL_THREADPOOL_H
