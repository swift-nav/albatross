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

#ifndef ALBATROSS_INDEXING_BLOCK_HPP_
#define ALBATROSS_INDEXING_BLOCK_HPP_

namespace albatross {

namespace detail {

// Partition a square, triangular matrix into blocks of columns
// (upper-triangular) or rows (lower-triangular) with approximately
// equal areas.
//
// Returns pairs of indices: [start, end) for each block.
inline auto partition_triangular(Eigen::Index size,
                                 Eigen::Index partition_count) {
  std::vector<std::pair<Eigen::Index, Eigen::Index>> results{};
  double area = 0;
  Eigen::Index start_index = 0;
  for (Eigen::Index block = 0; block < partition_count; ++block) {
    // idx_n+1 = sqrt( 1 / k + A_n )
    const auto end_fraction = sqrt(1 / cast::to_double(partition_count) + area);
    area = end_fraction * end_fraction;
    const auto end_index =
        static_cast<Eigen::Index>(rint(cast::to_double(size) * end_fraction));
    results.emplace_back(start_index, end_index);
    start_index = end_index;
  }

  // Due to the ceiling, this can be one over in many cases.
  results.back().second = std::min(results.back().second, size);

  return results;
}

}  // namespace detail

}  // namespace albatross

#endif /* ALBATROSS_INDEXING_BLOCK_HPP_ */
