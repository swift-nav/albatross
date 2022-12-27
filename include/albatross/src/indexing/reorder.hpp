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

#ifndef ALBATROSS_INDEXING_REORDER_HPP_
#define ALBATROSS_INDEXING_REORDER_HPP_

namespace albatross {

template <typename T, typename GroupKey>
std::vector<T> reorder(const std::vector<T> &x,
                       GroupIndexer<GroupKey> indexer) {
  std::vector<T> output(x.size());
  std::size_t i = 0;
  for (const auto &pair : indexer) {
    for (const auto &ind : pair.second) {
      output[i++] = x[ind];
    }
  }
  return output;
}

template <typename T, typename GrouperFunc>
std::vector<T> reorder(const std::vector<T> &x, GrouperFunc &&grouper) {
  return reorder(x, std::forward<GrouperFunc>(grouper));
}

template <typename T, typename GrouperFunc>
RegressionDataset<T> reorder(const RegressionDataset<T> &x,
                             GrouperFunc grouper) {

  RegressionDataset<T> output(std::vector<T>(x.size()),
                              MarginalDistribution(Eigen::VectorXd(x.size()),
                                                   Eigen::VectorXd(x.size())));

  std::size_t i = 0;
  const auto indexer = group_by(x, grouper).indexer();
  for (const auto &pair : indexer) {
    for (const auto &ind : pair.second) {
      output.features[i] = x.features[ind];
      output.targets.mean[i] = x.target.mean[ind];
      output.targets.diagonal()[i] = x.target.get_diagonal(ind);
      ++i;
    }
  }
  return output;
}

} // namespace albatross

#endif /* ALBATROSS_INDEXING_GROUPBY_HPP_ */
