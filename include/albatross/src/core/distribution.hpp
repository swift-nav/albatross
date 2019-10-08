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

inline bool operator==(const albatross::DiagonalMatrixXd &x,
                       const albatross::DiagonalMatrixXd &y) {
  return (x.diagonal() == y.diagonal());
}

namespace albatross {

/*
 * A Distribution holds what is typically assumed to be a
 * multivariate Gaussian distribution with mean and optional
 * covariance.
 */
template <typename CovarianceType> struct Distribution {
  Eigen::VectorXd mean;
  CovarianceType covariance;
  std::map<std::string, std::string> metadata;

  Distribution() : mean(), covariance(), metadata(){};
  Distribution(const Eigen::VectorXd &mean_)
      : mean(mean_), covariance(), metadata(){};
  Distribution(const Eigen::VectorXd &mean_, const CovarianceType &covariance_)
      : mean(mean_), covariance(covariance_), metadata(){};

  std::size_t size() const;

  void assert_valid() const;

  bool has_covariance() const;

  double get_diagonal(Eigen::Index i) const;

  bool operator==(const Distribution<CovarianceType> &other) const {
    return (mean == other.mean && has_covariance() == other.has_covariance() &&
            covariance == other.covariance);
  }

  template <typename OtherCovarianceType>
  typename std::enable_if<
      !std::is_same<CovarianceType, OtherCovarianceType>::value, bool>::type
  operator==(const Distribution<OtherCovarianceType> &) const {
    return false;
  }
};

template <typename CovarianceType>
std::size_t Distribution<CovarianceType>::size() const {
  // If the covariance is defined it must have the same number
  // of rows and columns which should be the same size as the mean.
  assert_valid();
  return static_cast<std::size_t>(mean.size());
}

template <typename CovarianceType>
void Distribution<CovarianceType>::assert_valid() const {
  if (covariance.size() > 0) {
    assert(covariance.rows() == covariance.cols());
    assert(mean.size() == covariance.rows());
  }
}

template <typename CovarianceType>
bool Distribution<CovarianceType>::has_covariance() const {
  assert_valid();
  return covariance.size() > 0;
}

template <typename CovarianceType>
double Distribution<CovarianceType>::get_diagonal(Eigen::Index i) const {
  return has_covariance() ? covariance.diagonal()[i] : NAN;
}

template <typename SizeType, typename CovarianceType>
Distribution<CovarianceType> subset(const Distribution<CovarianceType> &dist,
                                    const std::vector<SizeType> &indices) {
  auto subset_mean = albatross::subset(Eigen::VectorXd(dist.mean), indices);
  if (dist.has_covariance()) {
    auto subset_cov = albatross::symmetric_subset(dist.covariance, indices);
    return Distribution<CovarianceType>(subset_mean, subset_cov);
  } else {
    return Distribution<CovarianceType>(subset_mean);
  }
}

template <typename SizeType, typename CovarianceType>
void set_subset(const Distribution<CovarianceType> &from,
                const std::vector<SizeType> &indices,
                Distribution<CovarianceType> *to) {
  set_subset(from.mean, indices, &to->mean);
  assert(from.has_covariance() == to->has_covariance());
  if (from.has_covariance()) {
    set_subset(from.covariance, indices, &to->covariance);
  }
}

inline MarginalDistribution
concatenate_marginals(const MarginalDistribution &x,
                      const MarginalDistribution &y) {
  Eigen::VectorXd mean(x.mean.size() + y.mean.size());
  mean.block(0, 0, x.mean.size(), 1) = x.mean;
  mean.block(x.mean.size(), 0, y.mean.size(), 1) = y.mean;

  if (!x.has_covariance() && !y.has_covariance()) {
    return MarginalDistribution(mean);
  } else {
    Eigen::VectorXd variance(mean.size());
    variance.fill(NAN);
    if (x.has_covariance()) {
      variance.block(0, 0, x.mean.size(), 1) = x.covariance.diagonal();
    }
    if (y.has_covariance()) {
      variance.block(x.mean.size(), 0, y.mean.size(), 1) =
          y.covariance.diagonal();
    }

    return MarginalDistribution(mean, variance.asDiagonal());
  }
}

inline MarginalDistribution
concatenate_marginals(const std::vector<MarginalDistribution> &dists) {

  if (dists.size() == 0) {
    return MarginalDistribution();
  }

  std::size_t size = 0;
  bool any_has_covariance = false;
  for (const auto &dist : dists) {
    size += dist.size();
    any_has_covariance = any_has_covariance || dist.has_covariance();
  }

  Eigen::VectorXd mean(static_cast<Eigen::Index>(size));
  Eigen::Index i = 0;
  for (const auto &dist : dists) {
    mean.block(i, 0, dist.mean.size(), 1) = dist.mean;
    i += dist.mean.size();
  }

  if (!any_has_covariance) {
    return MarginalDistribution(mean);
  } else {
    Eigen::VectorXd variance(mean.size());
    variance.fill(NAN);
    Eigen::Index i = 0;
    for (const auto &dist : dists) {
      if (dist.has_covariance()) {
        variance.block(i, 0, dist.mean.size(), 1) = dist.covariance.diagonal();
      }
      i += dist.mean.size();
    }

    return MarginalDistribution(mean, variance.asDiagonal());
  }
}

} // namespace albatross

#endif
