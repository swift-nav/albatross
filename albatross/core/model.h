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

#ifndef ALBATROSS_CORE_MODEL_H
#define ALBATROSS_CORE_MODEL_H

#include <Eigen/Core>
#include <map>
#include <vector>
#include "optional.hpp"
#include "map_utils.h"
#include "static_inspection.h"
#include "parameter_handling_mixin.h"

using std::experimental::optional;

namespace albatross {

struct PredictionDistribution {
  Eigen::VectorXd mean;
  Eigen::MatrixXd covariance;

  PredictionDistribution(const Eigen::VectorXd &mean_)
      : mean(mean_), covariance(){};
  PredictionDistribution(const Eigen::VectorXd &mean_,
                         const Eigen::MatrixXd &covariance_)
      : mean(mean_), covariance(covariance_){};
};

/*
 * A RegressionDataset holds two vectors of data, the features
 * where a single feature can be any class that contains the information used
 * to make predictions of the target.  This is called a RegressionDataset since
 * it is assumed that each feature is regressed to a single double typed
 * target.
 */
template <class FeatureType>
struct RegressionDataset {
  std::vector<FeatureType> features;
  Eigen::VectorXd targets;

  RegressionDataset(const std::vector<FeatureType> &features_,
                    const Eigen::VectorXd &targets_)
      : features(features_), targets(targets_) {
    // If the two inputs aren't the same size they clearly aren't
    // consistent.
    assert(static_cast<int>(features_.size()) ==
           static_cast<int>(targets_.size()));
  }
};

typedef int32_t s32;
using FoldIndices = std::vector<s32>;
using FoldName = std::string;
using FoldIndexer = std::map<FoldName, FoldIndices>;

/*
 * A combination of training and testing datasets, typically used in cross
 * validation.
 */
template <class FeatureType>
struct RegressionFold {
  RegressionDataset<FeatureType> train;
  RegressionDataset<FeatureType> test;
  FoldName name;
  FoldIndices test_indices;

  RegressionFold(const RegressionDataset<FeatureType> &train_,
                 const RegressionDataset<FeatureType> &test_,
                 const FoldName &name_, const FoldIndices &test_indices_)
      : train(train_), test(test_), name(name_), test_indices(test_indices_){};
};

/*
 * A model that uses a single Feature to estimate the value of a double typed
 * target.
 */
template <class FeatureType, class ModelFit>
class RegressionModel : public ParameterHandlingMixin {
 public:
  typedef FeatureType Feature;
  RegressionModel() : ParameterHandlingMixin(), fit_storage_() {};
  virtual ~RegressionModel(){};

  /*
   * Provides a wrapper around the implementation `fit_` which performs
   * simple size checks and makes sure the fit method is called before
   * predict.
   */
  ModelFit fit(const std::vector<FeatureType> &features,
           const Eigen::VectorXd &targets) {
    assert(static_cast<s32>(features.size()) ==
           static_cast<s32>(targets.size()));
    auto model_fit = fit_(features, targets);
    fit_storage_ = model_fit;
    return model_fit;
  }

  /*
   * Convenience function which unpacks a dataset into features and targets.
   */
  ModelFit fit(const RegressionDataset<FeatureType> &dataset) {
    return fit(dataset.features, dataset.targets);
  }

  /*
   * Similar to fit, this predict method wraps the implementation `predict_`
   * and makes simple checks to confirm the implementation is returning
   * properly sized PredictionDistributions.
   */
  PredictionDistribution predict(
      const std::vector<FeatureType> &features) const {
    assert(fit_storage_);
    PredictionDistribution preds = predict_(features);
    assert(static_cast<s32>(preds.mean.size()) ==
           static_cast<s32>(features.size()));
    return preds;
  }

  /*
   * Computes predictions for the test features given set of training
   * features and targets. In the general case this is simply a call to fit,
   * follwed by predict but overriding this method may speed up computation for
   * some models.
   */
  PredictionDistribution fit_and_predict(
      const std::vector<FeatureType> &train_features,
      const Eigen::VectorXd &train_targets,
      const std::vector<FeatureType> &test_features) {
    // Fit using the training data, then predict with the test.
    fit(train_features, train_targets);
    return predict(test_features);
  }

  /*
   * A convenience wrapper around fit_and_predict which uses the entries
   * in a RegressionFold struct
   */
  PredictionDistribution fit_and_predict(
      const RegressionFold<FeatureType> &fold) {
    return fit_and_predict(fold.train.features, fold.train.targets,
                           fold.test.features);
  }

  virtual std::string get_name() const = 0;

 protected:
  /*
   * These methods are required from an implmenting class, notice that
   * the methods are marked `const`.  Anything that needs to be stored
   * in the model should be returned in the `ModelFit` type.
   */
  virtual ModelFit fit_(const std::vector<FeatureType> &features,
                          const Eigen::VectorXd &targets) const = 0;

  virtual PredictionDistribution predict_(
      const std::vector<FeatureType> &features) const = 0;

  optional<ModelFit> fit_storage_;
};

template <class FeatureType, class ModelFit>
using RegressionModelCreator =
    std::function<std::unique_ptr<RegressionModel<FeatureType, ModelFit>>()>;
}

#endif
