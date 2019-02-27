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

#ifndef ALBATROSS_CORE_CONCATENATE_H
#define ALBATROSS_CORE_CONCATENATE_H

namespace albatross {

template <typename FeatureType>
RegressionDataset<FeatureType> concatenate_datasets(
    const std::vector<RegressionDataset<FeatureType>> &datasets) {
  std::size_t n = 0;
  for (const auto &d : datasets) {
    n += d.features.size();
  }

  Eigen::VectorXd target_means(static_cast<Eigen::Index>(n));
  Eigen::VectorXd target_variance(static_cast<Eigen::Index>(n));
  std::vector<FeatureType> features(n);

  bool target_has_covariance = datasets[0].targets.has_covariance();

  std::size_t i = 0;
  for (const auto &d : datasets) {
    assert(target_has_covariance == d.targets.has_covariance());
    for (std::size_t j = 0; j < d.features.size(); ++j) {
      features[i] = d.features[j];
      Eigen::Index ei = static_cast<Eigen::Index>(i);
      Eigen::Index ej = static_cast<Eigen::Index>(j);
      target_means[ei] = d.targets.mean[ej];
      target_variance[ei] = d.targets.get_diagonal(ej);
      ++i;
    }
  }
  assert(i == n);
  MarginalDistribution targets;
  if (target_has_covariance) {
    targets = MarginalDistribution(target_means, target_variance.asDiagonal());
  } else {
    targets = MarginalDistribution(target_means);
  }
  return RegressionDataset<FeatureType>(features, targets);
}

template <typename CovarianceType>
MarginalDistribution concatenate_distributions(
    const std::vector<Distribution<CovarianceType>> &distributions) {

  Eigen::Index n = 0;
  for (const auto &d : distributions) {
    n += d.size();
  }

  Eigen::VectorXd mean(n);
  Eigen::VectorXd variance(static_cast<Eigen::Index>(n));

  Eigen::Index i = 0;
  for (const auto &d : distributions) {
    Eigen::Index d_size = static_cast<Eigen::Index>(d.size());
    for (Eigen::Index j = 0; j < d_size; ++j) {
      mean[i] = d.mean[j];
      variance[i] = d.get_diagonal(j);
      ++i;
    }
  }
  assert(i == n);
  MarginalDistribution output(mean, variance.asDiagonal());
  return output;
}

} // namespace albatross

#endif
