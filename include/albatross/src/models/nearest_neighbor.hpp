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

#ifndef ALBATROSS_SRC_MODELS_NEAREST_NEIGHBOR_MODEL_HPP_
#define ALBATROSS_SRC_MODELS_NEAREST_NEIGHBOR_MODEL_HPP_

namespace albatross {

template <typename DistanceMetric> class NearestNeighborModel;

template <typename FeatureType> struct NearestNeighborFit;

template <typename FeatureType> struct Fit<NearestNeighborFit<FeatureType>> {

  Fit() : training_data(){};

  Fit(const RegressionDataset<FeatureType> &dataset) : training_data(dataset){};

  bool operator==(const Fit<NearestNeighborFit<FeatureType>> &other) const {
    return training_data == other.training_data;
  }

  RegressionDataset<FeatureType> training_data;
};

template <typename DistanceMetric>
class NearestNeighborModel
    : public ModelBase<NearestNeighborModel<DistanceMetric>> {

public:
  NearestNeighborModel() : distance_metric(){};

  std::string get_name() const { return "nearest_neighbor_model"; };

  template <typename FeatureType>
  Fit<NearestNeighborFit<FeatureType>>
  _fit_impl(const std::vector<FeatureType> &features,
            const MarginalDistribution &targets) const {
    return Fit<NearestNeighborFit<FeatureType>>(
        RegressionDataset<FeatureType>(features, targets));
  }

  template <typename FeatureType>
  auto fit_from_prediction(const std::vector<FeatureType> &features,
                           const JointDistribution &prediction) const {
    const NearestNeighborModel<DistanceMetric> m(*this);
    MarginalDistribution marginal_pred(
        prediction.mean, prediction.covariance.diagonal().asDiagonal());
    Fit<NearestNeighborFit<FeatureType>> fit = {
        RegressionDataset<FeatureType>(features, marginal_pred)};
    FitModel<NearestNeighborModel, Fit<NearestNeighborFit<FeatureType>>>
        fit_model(m, fit);
    return fit_model;
  }

  template <typename FeatureType>
  MarginalDistribution
  _predict_impl(const std::vector<FeatureType> &features,
                const Fit<NearestNeighborFit<FeatureType>> &fit,
                PredictTypeIdentity<MarginalDistribution> &&) const {
    const Eigen::Index n = static_cast<Eigen::Index>(features.size());
    Eigen::VectorXd mean = Eigen::VectorXd::Zero(n);
    mean.fill(NAN);
    Eigen::VectorXd variance = Eigen::VectorXd::Zero(n);
    variance.fill(NAN);

    for (std::size_t i = 0; i < features.size(); ++i) {
      const auto min_index =
          index_with_min_distance(features[i], fit.training_data.features);
      mean[i] = fit.training_data.targets.mean[min_index];
      variance[i] = fit.training_data.targets.get_diagonal(min_index);
    }

    if (fit.training_data.targets.has_covariance()) {
      return MarginalDistribution(mean, variance.asDiagonal());
    } else {
      return MarginalDistribution(mean);
    }
  }

private:
  template <typename FeatureType>
  std::size_t
  index_with_min_distance(const FeatureType &ref,
                          const std::vector<FeatureType> &features) const {
    assert(features.size() > 0);

    std::size_t min_index = 0;
    double min_distance = distance_metric(ref, features[0]);

    for (std::size_t i = 1; i < features.size(); ++i) {
      const double dist = distance_metric(ref, features[i]);
      if (dist < min_distance) {
        min_index = i;
        min_distance = dist;
      }
    }
    return min_index;
  }

  DistanceMetric distance_metric;
};

} // namespace albatross

#endif // ALBATROSS_SRC_MODELS_NEAREST_NEIGHBOR_MODEL_HPP_
