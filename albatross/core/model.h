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
#include "map_utils.h"
#include "parameter_handling_mixin.h"
#include "traits.h"
#include <cereal/archives/json.hpp>

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
template <typename FeatureType>
struct RegressionDataset {
  std::vector<FeatureType> features;
  Eigen::VectorXd targets;

  RegressionDataset() {};

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
template <typename FeatureType>
struct RegressionFold {
  RegressionDataset<FeatureType> train_dataset;
  RegressionDataset<FeatureType> test_dataset;
  FoldName name;
  FoldIndices test_indices;

  RegressionFold(const RegressionDataset<FeatureType> &train_dataset_,
                 const RegressionDataset<FeatureType> &test_dataset_,
                 const FoldName &name_, const FoldIndices &test_indices_)
      : train_dataset(train_dataset_), test_dataset(test_dataset_), name(name_), test_indices(test_indices_){};
};

/*
 * A model that uses a single Feature to estimate the value of a double typed
 * target.
 */
template <typename FeatureType>
class RegressionModel : public ParameterHandlingMixin {
 public:
  using Feature = FeatureType;
  RegressionModel() : ParameterHandlingMixin(), has_been_fit_() {};
  virtual ~RegressionModel(){};

  template <typename OtherFeatureType>
  bool operator == (const RegressionModel<FeatureType> &other) const {
    return false;
  }

  virtual bool operator == (const RegressionModel<FeatureType> &other) const {
    // If the fit method has been called it's possible that some unknown
    // class members may have been modified.  As such, if a model has been
    // fit we fail hard to avoid possibly unexpected behavior.  Any
    // implementation that wants a functional equality operator after
    // having been fit will need to override this one.
    assert(!has_been_fit());
    return (get_name() == other.get_name() &&
            get_params() == other.get_params() &&
            has_been_fit() == other.has_been_fit());
  }

  /*
   * Provides a wrapper around the implementation `fit_` which performs
   * simple size checks and makes sure the fit method is called before
   * predict.
   */
  void fit(const std::vector<FeatureType> &features,
           const Eigen::VectorXd &targets) {
    assert(static_cast<s32>(features.size()) ==
           static_cast<s32>(targets.size()));
    fit_(features, targets);
    has_been_fit_ = true;
  }

  /*
   * Convenience function which unpacks a dataset into features and targets.
   */
  void fit(const RegressionDataset<FeatureType> &dataset) {
    fit(dataset.features, dataset.targets);
  }

  /*
   * Similar to fit, this predict method wraps the implementation `predict_`
   * and makes simple checks to confirm the implementation is returning
   * properly sized PredictionDistributions.
   */
  PredictionDistribution predict(
      const std::vector<FeatureType> &features) const {
    assert(has_been_fit());
    PredictionDistribution preds = predict_(features);
    assert(static_cast<s32>(preds.mean.size()) ==
           static_cast<s32>(features.size()));
    return preds;
  }

  PredictionDistribution predict(
      const FeatureType &feature) const {
    std::vector<FeatureType> features = {feature};
    return predict(features);
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

  std::string pretty_string() const {
    std::ostringstream ss;
    ss << get_name() << std::endl;
    ss << ParameterHandlingMixin::pretty_string();
    return ss.str();
  }

  virtual bool has_been_fit() const {
    return has_been_fit_;
  }

  virtual std::string get_name() const = 0;

  /*
   * Here we define the serialization routines.  Note that while in most
   * cases we could use the cereal method `serialize`, in this case we don't
   * know for sure where the parameters are stored.  The GaussianProcessRegression
   * model, for example, derives its parameters from its covariance function,
   * so it's `params_` are actually empty.  As a result we need to use the
   * save/load cereal variant and deal with parameters through the get/set
   * interface.
   */
  template<class Archive>
  void save(Archive & archive) const
  {
    auto params = get_params();
    archive(cereal::make_nvp("parameters", params));
    archive(cereal::make_nvp("has_been_fit", has_been_fit_));
  }

  template<class Archive>
  void load(Archive & archive)
  {
    auto params = get_params();
    archive(cereal::make_nvp("parameters", params));
    archive(cereal::make_nvp("has_been_fit", has_been_fit_));
    set_params(params);
  }

 protected:

  virtual void fit_(const std::vector<FeatureType> &features,
                    const Eigen::VectorXd &targets) = 0;

  virtual PredictionDistribution predict_(
      const std::vector<FeatureType> &features) const = 0;

  bool has_been_fit_;
};


template <typename FeatureType>
using RegressionModelCreator =
    std::function<std::unique_ptr<RegressionModel<FeatureType>>()>;
}

#endif
