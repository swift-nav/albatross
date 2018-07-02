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

#ifndef ALBATROSS_CORE_DISTRIBUTION_H
#define ALBATROSS_CORE_DISTRIBUTION_H

#include "indexing.h"
#include <Eigen/Core>
#include <iostream>
#include <vector>

namespace albatross {

/*
 * A Distribution holds what is typically assumed to be a
 * multivariate Gaussian distribution with mean and optional
 * covariance.
 */
template <typename CovarianceType> struct Distribution {
  Eigen::VectorXd mean;
  CovarianceType covariance;

  std::size_t size() const {
    // If the covariance is defined it must have the same number
    // of rows and columns which should be the same size as the mean.
    assert_valid();
    return mean.size();
  }

  void assert_valid() const {
    if (covariance.size() > 0) {
      assert(covariance.rows() == covariance.cols());
      assert(mean.size() == static_cast<std::size_t>(covariance.rows()));
    }
  }

  bool has_covariance() const {
    assert_valid();
    return covariance.size() > 0;
  }

  Distribution() : mean(), covariance(){};
  Distribution(const Eigen::VectorXd &mean_) : mean(mean_), covariance(){};
  Distribution(const Eigen::VectorXd &mean_, const CovarianceType &covariance_)
      : mean(mean_), covariance(covariance_){};
};

using DiagonalMatrixXd = Eigen::DiagonalMatrix<double, Eigen::Dynamic>;
using DenseDistribution = Distribution<Eigen::MatrixXd>;
using DiagonalDistribution = Distribution<DiagonalMatrixXd>;

template <typename CovarianceType, typename SizeType>
Distribution<CovarianceType> subset(const std::vector<SizeType> &indices,
                                    const Distribution<CovarianceType> &dist) {
  auto mean = subset(indices, Eigen::VectorXd(dist.mean));
  if (dist.has_covariance()) {
    auto cov = symmetric_subset(indices, dist.covariance);
    return Distribution<CovarianceType>(mean, cov);
  } else {
    return Distribution<CovarianceType>(mean);
  }
}

} // namespace albatross

#endif
