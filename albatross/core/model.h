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

#include <Eigen/Core>
#include <map>
#include <vector>
#include "map_utils.h"
#include "parameter_handler.h"

#ifndef LIBSWIFTNAV_BASE_MODEL_H
#define LIBSWIFTNAV_BASE_MODEL_H

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
 * A RegressionDataset holds two vectors of data, the predictors
 * where a single predictor can be any class that contains the information used
 * to make predictions of the target.  This is called a RegressionDataset since
 * it is assumed that each predictor is regressed to a single double typed
 * target.
 */
template <class Predictor>
struct RegressionDataset {
  std::vector<Predictor> predictors;
  Eigen::VectorXd targets;

  RegressionDataset(const std::vector<Predictor> &predictors_,
                    const Eigen::VectorXd &targets_)
      : predictors(predictors_), targets(targets_) {
    // If the two inputs aren't the same size they clearly aren't
    // consistent.
    assert(static_cast<int>(predictors_.size()) ==
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
template <class Predictor>
struct RegressionFold {
  RegressionDataset<Predictor> train;
  RegressionDataset<Predictor> test;
  FoldName name;
  FoldIndices test_indices;

  RegressionFold(const RegressionDataset<Predictor> &train_,
                 const RegressionDataset<Predictor> &test_,
                 const FoldName &name_, const FoldIndices &test_indices_)
      : train(train_), test(test_), name(name_), test_indices(test_indices_){};
};

/*
 * A model that uses a single Predictor to estimate the value of a double typed
 * target.
 */
template <class Predictor>
class RegressionModel : public ParameterHandlingMixin {
 public:
  RegressionModel() : ParameterHandlingMixin(), has_been_fit_(false){};
  virtual ~RegressionModel(){};

  /*
   * Provides a wrapper around the implementation `fit_` which performs
   * simple size checks and makes sure the fit method is called before
   * predict.
   */
  void fit(const std::vector<Predictor> &predictors,
           const Eigen::VectorXd &targets) {
    assert(static_cast<s32>(predictors.size()) ==
           static_cast<s32>(targets.size()));
    fit_(predictors, targets);
    has_been_fit_ = true;
  }

  /*
   * Convenience function which unpacks a dataset into predictors and targets.
   */
  void fit(const RegressionDataset<Predictor> &dataset) {
    fit(dataset.predictors, dataset.targets);
  }

  /*
   * Similar to fit, this predict method wraps the implementation `predict_`
   * and makes simple checks to confirm the implementation is returning
   * properly sized PredictionDistributions.
   */
  PredictionDistribution predict(
      const std::vector<Predictor> &predictors) const {
    assert(has_been_fit_);
    PredictionDistribution preds = predict_(predictors);
    assert(static_cast<s32>(preds.mean.size()) ==
           static_cast<s32>(predictors.size()));
    return preds;
  }

  /*
   * Computes predictions for the test predictors given set of training
   * predictors and targets. In the general case this is simply a call to fit,
   * follwed by predict but overriding this method may speed up computation for
   * some models.
   */
  PredictionDistribution fit_and_predict(
      const std::vector<Predictor> &train_predictors,
      const Eigen::VectorXd &train_targets,
      const std::vector<Predictor> &test_predictors) {
    // Fit using the training data, then predict with the test.
    fit(train_predictors, train_targets);
    return predict(test_predictors);
  }

  /*
   * A convenience wrapper around fit_and_predict which uses the entries
   * in a RegressionFold struct
   */
  PredictionDistribution fit_and_predict(
      const RegressionFold<Predictor> &fold) {
    return fit_and_predict(fold.train.predictors, fold.train.targets,
                           fold.test.predictors);
  }

 protected:
  virtual void fit_(const std::vector<Predictor> &predictors,
                    const Eigen::VectorXd &targets) = 0;

  virtual PredictionDistribution predict_(
      const std::vector<Predictor> &predictors) const = 0;

  bool has_been_fit_ = false;
};

template <class Predictor>
using RegressionModelCreator =
    std::function<std::unique_ptr<RegressionModel<Predictor>>()>;
}

#endif
