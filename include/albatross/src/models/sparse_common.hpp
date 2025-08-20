/*
 * Copyright (C) 2024 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef INCLUDE_ALBATROSS_MODELS_SPARSE_COMMON_H_
#define INCLUDE_ALBATROSS_MODELS_SPARSE_COMMON_H_

namespace albatross {

namespace details {

constexpr double DEFAULT_NUGGET = 1e-8;

inline std::string measurement_nugget_name() { return "measurement_nugget"; }

inline std::string inducing_nugget_name() { return "inducing_nugget"; }

static constexpr double cSparseRNugget = 1.e-10;

} // namespace details

struct UniformlySpacedInducingPoints {

  UniformlySpacedInducingPoints(std::size_t num_points_ = 10)
      : num_points(num_points_) {}

  template <typename CovarianceFunction>
  std::vector<double> operator()(const CovarianceFunction &cov ALBATROSS_UNUSED,
                                 const std::vector<double> &features) const {
    double min = *std::min_element(features.begin(), features.end());
    double max = *std::max_element(features.begin(), features.end());
    return linspace(min, max, num_points);
  }

  std::size_t num_points;
};

struct StateSpaceInducingPointStrategy {

  template <typename CovarianceFunction, typename FeatureType,
            std::enable_if_t<has_valid_state_space_representation<
                                 CovarianceFunction, FeatureType>::value,
                             int> = 0>
  auto operator()(const CovarianceFunction &cov,
                  const std::vector<FeatureType> &features) const {
    return cov.state_space_representation(features);
  }

  template <typename CovarianceFunction, typename FeatureType,
            std::enable_if_t<!has_valid_state_space_representation<
                                 CovarianceFunction, FeatureType>::value,
                             int> = 0>
  auto
  operator()(const CovarianceFunction &cov ALBATROSS_UNUSED,
             const std::vector<FeatureType> &features ALBATROSS_UNUSED) const
      ALBATROSS_FAIL(
          CovarianceFunction,
          "Covariance function is missing state_space_representation method, "
          "be sure _ssr_impl has been defined for the types concerned");
};

struct SPQRImplementation {
  using QRType = Eigen::SPQR<Eigen::SparseMatrix<double>>;

  static std::unique_ptr<QRType> compute(const Eigen::MatrixXd &m,
                                         ThreadPool *threads) {
    return SPQR_create(m.sparseView(), threads);
  }
};

struct DenseQRImplementation {
  using QRType = Eigen::ColPivHouseholderQR<Eigen::MatrixXd>;

  static std::unique_ptr<QRType> compute(const Eigen::MatrixXd &m,
                                         ThreadPool *threads
                                         __attribute__((unused))) {
    return std::make_unique<QRType>(m);
  }
};

} // namespace albatross

#endif // INCLUDE_ALBATROSS_MODELS_SPARSE_COMMON_H_
