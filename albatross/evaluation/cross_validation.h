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

#ifndef ALBATROSS_CROSS_VALIDATION_H
#define ALBATROSS_CROSS_VALIDATION_H

namespace albatross {

/*
 * This is a specialization of the `Prediction` class which adds some
 * cross validation specific methods, and specializes the standard
 * methods (such as mean, marginal, joint).
 */
template <typename ModelType, typename FeatureType>
class Prediction<CrossValidation<ModelType>, FeatureType, FoldIndexer> {
public:
  Prediction(const ModelType &model,
             const RegressionDataset<FeatureType> &dataset,
             const FoldIndexer &indexer)
      : model_(model), dataset_(dataset), indexer_(indexer) {}

  // MEAN

  // Cross validation specialized means().
  template <
      typename DummyType = ModelType,
      typename std::enable_if<has_valid_cv_mean<DummyType, FeatureType>::value,
                              int>::type = 0>
  std::vector<Eigen::VectorXd> means() const {
    return model_.cross_validated_predictions(
        dataset_, indexer_, PredictTypeIdentity<Eigen::VectorXd>());
  }

  // Generic means();
  template <typename DummyType = ModelType,
            typename std::enable_if<
                has_mean<Prediction<
                    DummyType, FeatureType,
                    typename fit_type<DummyType, FeatureType>::type>>::value &&
                    !has_valid_cv_mean<DummyType, FeatureType>::value,
                int>::type = 0>
  std::vector<Eigen::VectorXd> means() const {
    const auto folds = folds_from_fold_indexer(dataset_, indexer_);
    const auto predictions = albatross::get_predictions(model_, folds);
    return get_means(predictions);
  }

  // No valid method of computing the means.
  template <typename DummyType = ModelType,
            typename std::enable_if<
                !has_mean<Prediction<
                    DummyType, FeatureType,
                    typename fit_type<DummyType, FeatureType>::type>>::value &&
                    !has_valid_cv_mean<DummyType, FeatureType>::value,
                int>::type = 0>
  std::vector<Eigen::VectorXd> means() const = delete;

  template <typename DummyType = ModelType,
            typename std::enable_if<
                has_mean<Prediction<
                    DummyType, FeatureType,
                    typename fit_type<DummyType, FeatureType>::type>>::value ||
                    has_valid_cv_mean<DummyType, FeatureType>::value,
                int>::type = 0>
  Eigen::VectorXd mean() const {
    const auto folds = folds_from_fold_indexer(dataset_, indexer_);
    return concatenate_mean_predictions(folds, means());
  }

  // No valid method of computing the means.
  template <typename DummyType = ModelType,
            typename std::enable_if<
                !has_mean<Prediction<
                    DummyType, FeatureType,
                    typename fit_type<DummyType, FeatureType>::type>>::value &&
                    !has_valid_cv_mean<DummyType, FeatureType>::value,
                int>::type = 0>

  Eigen::VectorXd mean() const = delete;

  // Marginal

  // Cross validation specialized marginals()
  template <
      typename DummyType = ModelType,
      typename std::enable_if<
          has_valid_cv_marginal<DummyType, FeatureType>::value, int>::type = 0>
  std::vector<MarginalDistribution> marginals() const {
    return model_.cross_validated_predictions(
        dataset_, indexer_, PredictTypeIdentity<MarginalDistribution>());
  }

  // Generic marginals()
  template <typename DummyType = ModelType,
            typename std::enable_if<
                has_marginal<Prediction<
                    DummyType, FeatureType,
                    typename fit_type<DummyType, FeatureType>::type>>::value &&
                    !has_valid_cv_marginal<DummyType, FeatureType>::value,
                int>::type = 0>
  std::vector<MarginalDistribution> marginals() const {
    const auto folds = folds_from_fold_indexer(dataset_, indexer_);
    const auto predictions = albatross::get_predictions(model_, folds);
    return get_marginals(predictions);
  }

  // No valid way of computing marginals.
  template <typename DummyType = ModelType,
            typename std::enable_if<
                !has_marginal<Prediction<
                    DummyType, FeatureType,
                    typename fit_type<DummyType, FeatureType>::type>>::value &&
                    !has_valid_cv_marginal<DummyType, FeatureType>::value,
                int>::type = 0>
  MarginalDistribution marginals() const = delete;

  // No valid way of computing marginals.
  template <typename DummyType = ModelType,
            typename std::enable_if<
                has_marginal<Prediction<
                    DummyType, FeatureType,
                    typename fit_type<DummyType, FeatureType>::type>>::value ||
                    has_valid_cv_marginal<DummyType, FeatureType>::value,
                int>::type = 0>
  MarginalDistribution marginal() const {
    const auto folds = folds_from_fold_indexer(dataset_, indexer_);
    return concatenate_marginal_predictions(folds, marginals());
  }

  // No valid way of computing marginals.
  template <typename DummyType = ModelType,
            typename std::enable_if<
                !has_marginal<Prediction<
                    DummyType, FeatureType,
                    typename fit_type<DummyType, FeatureType>::type>>::value &&
                    !has_valid_cv_marginal<DummyType, FeatureType>::value,
                int>::type = 0>
  MarginalDistribution marginal() const = delete;

  // JOINT

  // Cross validation specific joints().
  template <
      typename DummyType = ModelType,
      typename std::enable_if<has_valid_cv_joint<DummyType, FeatureType>::value,
                              int>::type = 0>
  std::vector<JointDistribution> joints() const {
    return model_.cross_validated_predictions(
        dataset_, indexer_, PredictTypeIdentity<JointDistribution>());
  }

  // Generic joints().
  template <typename DummyType = ModelType,
            typename std::enable_if<
                has_joint<Prediction<
                    DummyType, FeatureType,
                    typename fit_type<DummyType, FeatureType>::type>>::value &&
                    !has_valid_cv_joint<DummyType, FeatureType>::value,
                int>::type = 0>
  std::vector<JointDistribution> joints() const {
    const auto folds = folds_from_fold_indexer(dataset_, indexer_);
    const auto predictions = albatross::get_predictions(model_, folds);
    return get_joints(predictions);
  }

  // No valid way of computing joints().
  template <typename DummyType = ModelType,
            typename std::enable_if<
                !has_joint<Prediction<
                    DummyType, FeatureType,
                    typename fit_type<DummyType, FeatureType>::type>>::value &&
                    !has_valid_cv_joint<DummyType, FeatureType>::value,
                int>::type = 0>
  std::vector<JointDistribution> joints() const = delete;

  template <typename DummyType = ModelType>
  JointDistribution joint() const =
      delete; // Cannot produce a full joint distribution from cross validation.

  template <typename PredictType>
  PredictType get(PredictTypeIdentity<PredictType> =
                      PredictTypeIdentity<PredictType>()) const {
    return get(get_type<PredictType>());
  }

private:
  template <typename T> struct get_type {};

  auto get(get_type<Eigen::VectorXd>) const { return this->mean(); }

  auto get(get_type<std::vector<Eigen::VectorXd>>) const {
    return this->means();
  }

  auto get(get_type<MarginalDistribution>) const { return this->marginal(); }

  auto get(get_type<std::vector<MarginalDistribution>>) const {
    return this->marginals();
  }

  auto get(get_type<JointDistribution>) const { return this->joint(); }

  auto get(get_type<std::vector<JointDistribution>>) const {
    return this->joints();
  }

  const ModelType model_;
  const RegressionDataset<FeatureType> dataset_;
  const FoldIndexer indexer_;
};

/*
 * Cross Validation
 */

template <typename ModelType, typename FeatureType>
using CVPrediction =
    Prediction<CrossValidation<ModelType>, FeatureType, FoldIndexer>;

template <typename ModelType> class CrossValidation {

  ModelType model_;

public:
  CrossValidation(const ModelType &model) : model_(model){};

  // get_predictions

  template <typename FeatureType>
  auto
  get_predictions(const std::vector<RegressionFold<FeatureType>> &folds) const {
    return albatross::get_predictions(model_, folds);
  }

  template <typename FeatureType, typename IndexFunc>
  auto get_predictions(const RegressionDataset<FeatureType> &dataset,
                       const IndexFunc &index_function) const {
    const auto indexer = index_function(dataset);
    const auto folds = folds_from_fold_indexer(dataset, indexer);
    return get_predictions(folds);
  }

  // get_prediction

  template <typename FeatureType>
  CVPrediction<ModelType, FeatureType>
  predict(const RegressionDataset<FeatureType> &dataset,
          const FoldIndexer &indexer) const {
    return CVPrediction<ModelType, FeatureType>(model_, dataset, indexer);
  }

  template <typename FeatureType, typename IndexFunc>
  auto predict(const RegressionDataset<FeatureType> &dataset,
               const IndexFunc &index_function) const {
    const auto indexer = index_function(dataset);
    return predict(dataset, indexer);
  }

  // Scores

  template <typename RequiredPredictType, typename FeatureType>
  Eigen::VectorXd
  scores(const EvaluationMetric<RequiredPredictType> &metric,
         const std::vector<RegressionFold<FeatureType>> &folds) const {
    const auto preds = get_predictions(folds);
    return cross_validated_scores(metric, folds, preds);
  }

  template <typename RequiredPredictType, typename FeatureType>
  Eigen::VectorXd scores(const EvaluationMetric<RequiredPredictType> &metric,
                         const RegressionDataset<FeatureType> &dataset,
                         const FoldIndexer &indexer) const {
    const auto folds = folds_from_fold_indexer(dataset, indexer);
    const auto prediction = predict(dataset, indexer);
    const auto predictions =
        prediction.template get<std::vector<RequiredPredictType>>();
    return cross_validated_scores(metric, folds, predictions);
  }

  template <typename RequiredPredictType, typename FeatureType,
            typename IndexFunc>
  Eigen::VectorXd scores(const EvaluationMetric<RequiredPredictType> &metric,
                         const RegressionDataset<FeatureType> &dataset,
                         const IndexFunc &index_function) const {
    const auto indexer = index_function(dataset);
    return scores(metric, dataset, indexer);
  }
};

template <typename ModelType>
CrossValidation<ModelType> ModelBase<ModelType>::cross_validate() const {
  return CrossValidation<ModelType>(derived());
}

} // namespace albatross

#endif
