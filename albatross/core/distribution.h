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

#include "cereal/cereal.hpp"
#include "core/traits.h"
#include "eigen/serializable_diagonal_matrix.h"
#include <Eigen/Core>
#include <iostream>
#include <map>
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
  // Sometimes it can be helpful to keep track of some
  // auxillary information regarding how a distribution was
  // derived, that can be stored in this map.
  std::map<std::string, std::string> metadata;

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

  double get_diagonal(Eigen::Index i) const {
    return has_covariance() ? covariance.diagonal()[i] : NAN;
  }

  /*
   * If the CovarianceType is serializable, add a serialize method.
   */
  template <class Archive>
  typename std::enable_if<
      valid_in_out_serializer<CovarianceType, Archive>::value, void>::type
  serialize(Archive &archive) {
    archive(cereal::make_nvp("mean", mean));
    archive(cereal::make_nvp("covariance", covariance));
    archive(cereal::make_nvp("metadata", metadata));
  }

  /*
   * If you try to serialize a Distribution for which the covariance
   * type is not serializable you'll get an error.
   */
  template <class Archive>
  typename std::enable_if<
      !valid_in_out_serializer<CovarianceType, Archive>::value, void>::type
  save(Archive &archive) {
    static_assert(delay_static_assert<Archive>::value,
                  "In order to serialize a Distribution the corresponding "
                  "CovarianceType must be serializable.");
  }

  bool operator==(const Distribution &other) const {
    return (mean == other.mean && covariance == other.covariance &&
            metadata == other.metadata);
  }
};

// A JointDistribution has a dense covariance matrix, which
// contains the covariance between each variable and all others.
using JointDistribution = Distribution<Eigen::MatrixXd>;

// We use a wrapper around DiagonalMatrix in order to make
// the resulting distribution serializable
using DiagonalMatrixXd =
    Eigen::SerializableDiagonalMatrix<double, Eigen::Dynamic>;
// A MarginalDistribution has only a digaonal covariance
// matrix, so in turn only describes the variance of each
// variable independent of all others.
using MarginalDistribution = Distribution<DiagonalMatrixXd>;

} // namespace albatross

#endif
