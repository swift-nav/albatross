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

#include "core/dataset.h"
#include "core/indexing.h"
#include "core/parameter_handling_mixin.h"
#include "map_utils.h"
#include "traits.h"
#include <Eigen/Core>
#include <cereal/archives/json.hpp>
#include <map>
#include <vector>

namespace albatross {

namespace detail {
// This is effectively just a container that allows us to develop methods
// which behave different conditional on the type of predictions desired.
template <typename T> struct PredictTypeIdentity { typedef T type; };
}

// This can be used to make intentions more obvious when calling
// predict variants for which you only want the mean.
using PredictMeanOnly = Eigen::VectorXd;

using Insights = std::map<std::string, std::string>;

/*
 * A model that uses a single Feature to estimate the value of a double typed
 * target.
 */
template <typename FeatureType>
class RegressionModel : public ParameterHandlingMixin {
public:
  using Feature = FeatureType;
  RegressionModel() : ParameterHandlingMixin(), has_been_fit_(){};
  virtual ~RegressionModel(){};

  virtual bool operator==(const RegressionModel<FeatureType> &other) const {
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
           const MarginalDistribution &targets) {
    assert(features.size() > 0);
    assert(features.size() == static_cast<std::size_t>(targets.size()));
    has_been_fit_ = true;
    insights_["input_feature_count"] = std::to_string(features.size());
    fit_(features, targets);
  }

  /*
   * Convenience function which assumes zero target covariance.
   */
  void fit(const std::vector<FeatureType> &features,
           const Eigen::VectorXd &targets) {
    return fit(features, MarginalDistribution(targets));
  }

  /*
   * Convenience function which unpacks a dataset into features and targets.
   */
  void fit(const RegressionDataset<FeatureType> &dataset) {
    return fit(dataset.features, dataset.targets);
  }

  /*
   * Similar to fit, this predict methods wrap the implementation `predict_*_`
   * and makes simple checks to confirm the implementation is returning
   * properly sized Distribution.
   */
  template <typename PredictType = JointDistribution>
  PredictType predict(const std::vector<FeatureType> &features) const {
    return predict(features, detail::PredictTypeIdentity<PredictType>());
  }

  template <typename PredictType = JointDistribution>
  PredictType predict(const FeatureType &feature) const {
    std::vector<FeatureType> features = {feature};
    return predict<PredictType>(features);
  }

  template <typename PredictType = MarginalDistribution>
  std::vector<PredictType>
  cross_validated_predictions(const RegressionDataset<FeatureType> &dataset,
                              const FoldIndexer &fold_indexer) {
    return cross_validated_predictions_(
        dataset, fold_indexer, detail::PredictTypeIdentity<PredictType>());
  }

  // Because cross validation can never properly produce a full
  // joint distribution it is common to only use the marginal
  // predictions, hence the different default from predict.
  template <typename PredictType = MarginalDistribution>
  std::vector<PredictType> cross_validated_predictions(
      const std::vector<RegressionFold<FeatureType>> &folds) {
    // Iteratively make predictions and assemble the output vector
    std::vector<PredictType> predictions;
    for (std::size_t i = 0; i < folds.size(); i++) {
      fit(folds[i].train_dataset);
      predictions.push_back(
          predict<PredictType>(folds[i].test_dataset.features));
    }
    return predictions;
  }

  std::string pretty_string() const {
    std::ostringstream ss;
    ss << get_name() << std::endl;
    ss << ParameterHandlingMixin::pretty_string();
    return ss.str();
  }

  virtual bool has_been_fit() const { return has_been_fit_; }

  virtual std::string get_name() const = 0;

  Insights get_insights() const { return insights_; }

  virtual std::unique_ptr<RegressionModel<FeatureType>>
  ransac_model(double inlier_threshold, std::size_t min_inliers,
               std::size_t random_sample_size, std::size_t max_iterations) {
    static_assert(
        is_complete<GenericRansac<void, FeatureType>>::value,
        "ransac methods aren't complete yet, be sure you've included ransac.h");
    return make_generic_ransac_model<FeatureType>(
        this, inlier_threshold, min_inliers, random_sample_size, max_iterations,
        leave_one_out_indexer<FeatureType>);
  }

  /*
   * Here we define the serialization routines.  Note that while in most
   * cases we could use the cereal method `serialize`, in this case we don't
   * know for sure where the parameters are stored.  The
   * GaussianProcessRegression
   * model, for example, derives its parameters from its covariance function,
   * so it's `params_` are actually empty.  As a result we need to use the
   * save/load cereal variant and deal with parameters through the get/set
   * interface.
   */
  template <class Archive> void save(Archive &archive) const {
    auto params = get_params();
    archive(cereal::make_nvp("parameters", params));
    archive(cereal::make_nvp("has_been_fit", has_been_fit_));
  }

  template <class Archive> void load(Archive &archive) {
    auto params = get_params();
    archive(cereal::make_nvp("parameters", params));
    archive(cereal::make_nvp("has_been_fit", has_been_fit_));
    set_params(params);
  }

protected:
  virtual void fit_(const std::vector<FeatureType> &features,
                    const MarginalDistribution &targets) = 0;
  /*
   * Predict specializations
   */

  JointDistribution
  predict(const std::vector<FeatureType> &features,
          detail::PredictTypeIdentity<JointDistribution> &&identity) const {
    assert(has_been_fit());
    JointDistribution preds = predict_(features);
    assert(static_cast<std::size_t>(preds.mean.size()) == features.size());
    return preds;
  }

  MarginalDistribution
  predict(const std::vector<FeatureType> &features,
          detail::PredictTypeIdentity<MarginalDistribution> &&identity) const {
    assert(has_been_fit());
    MarginalDistribution preds = predict_marginal_(features);
    assert(static_cast<std::size_t>(preds.mean.size()) == features.size());
    return preds;
  }

  Eigen::VectorXd
  predict(const std::vector<FeatureType> &features,
          detail::PredictTypeIdentity<Eigen::VectorXd> &&identity) const {
    assert(has_been_fit());
    Eigen::VectorXd preds = predict_mean_(features);
    assert(static_cast<std::size_t>(preds.size()) == features.size());
    return preds;
  }

  /*
   * Cross validation specializations
   *
   * Note the naming here uses a trailing underscore.  This is to avoid
   * name hiding when implementing one of these methods in a derived
   * class:
   *
   * https://stackoverflow.com/questions/1628768/why-does-an-overridden-function-in-the-derived-class-hide-other-overloads-of-the
   */
  virtual std::vector<JointDistribution> cross_validated_predictions_(
      const RegressionDataset<FeatureType> &dataset,
      const FoldIndexer &fold_indexer,
      const detail::PredictTypeIdentity<JointDistribution> &identity) {
    const auto folds = folds_from_fold_indexer(dataset, fold_indexer);
    return cross_validated_predictions<JointDistribution>(folds);
  }

  virtual std::vector<MarginalDistribution> cross_validated_predictions_(
      const RegressionDataset<FeatureType> &dataset,
      const FoldIndexer &fold_indexer,
      const detail::PredictTypeIdentity<MarginalDistribution> &identity) {
    const auto folds = folds_from_fold_indexer(dataset, fold_indexer);
    return cross_validated_predictions<MarginalDistribution>(folds);
  }

  virtual std::vector<Eigen::VectorXd> cross_validated_predictions_(
      const RegressionDataset<FeatureType> &dataset,
      const FoldIndexer &fold_indexer,
      const detail::PredictTypeIdentity<PredictMeanOnly> &identity) {
    const auto folds = folds_from_fold_indexer(dataset, fold_indexer);
    return cross_validated_predictions<PredictMeanOnly>(folds);
  }

  virtual JointDistribution
  predict_(const std::vector<FeatureType> &features) const = 0;

  virtual MarginalDistribution
  predict_marginal_(const std::vector<FeatureType> &features) const {
    const auto full_distribution = predict_(features);
    return MarginalDistribution(
        full_distribution.mean,
        full_distribution.covariance.diagonal().asDiagonal());
  }

  virtual Eigen::VectorXd
  predict_mean_(const std::vector<FeatureType> &features) const {
    const auto marginal_distribution = predict_marginal_(features);
    return marginal_distribution.mean;
  }

  bool has_been_fit_;
  Insights insights_;
};

template <typename FeatureType>
using RegressionModelCreator =
    std::function<std::unique_ptr<RegressionModel<FeatureType>>()>;
} // namespace albatross

#endif
