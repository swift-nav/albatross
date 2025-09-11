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

template <typename ModelType, typename FeatureType>
auto predict_fold(const ModelType &model,
                  const RegressionFold<FeatureType> &fold) {
  return model.fit(fold.train_dataset).predict(fold.test_dataset.features);
};

/*
 * This is a specialization of the `Prediction` class which adds some
 * cross validation specific methods, and specializes the standard
 * methods (such as mean, marginal, joint).
 */
template <typename ModelType, typename FeatureType, typename GroupKey>
class Prediction<CrossValidation<ModelType>, FeatureType,
                 GroupIndexer<GroupKey>> {
public:
  Prediction(const ModelType &model,
             const RegressionDataset<FeatureType> &dataset,
             const GroupIndexer<GroupKey> &indexer)
      : model_(model), dataset_(dataset), indexer_(indexer) {}

  auto predictions() const {

    const auto predict_one_group = [&](const auto &,
                                       const GroupIndices &test_indices) {
      return predict_fold(model_, create_fold(test_indices, dataset_));
    };

    return indexer_.index_apply(predict_one_group);
  }

  // MEAN

  // Cross validation specialized means().
  template <typename DummyType = ModelType,
            typename std::enable_if<
                has_valid_cv_mean<DummyType, FeatureType, GroupKey>::value,
                int>::type = 0>
  Grouped<GroupKey, Eigen::VectorXd> means() const {
    return Grouped<GroupKey, Eigen::VectorXd>(
        model_.cross_validated_predictions(
            dataset_, indexer_, PredictTypeIdentity<Eigen::VectorXd>()));
  }

  // Generic means();
  template <typename DummyType = ModelType,
            typename std::enable_if<
                has_mean<Prediction<
                    DummyType, FeatureType,
                    typename fit_type<DummyType, FeatureType>::type>>::value &&
                    !has_valid_cv_mean<DummyType, FeatureType, GroupKey>::value,
                int>::type = 0>
  Grouped<GroupKey, Eigen::VectorXd> means() const {
    return get_means(predictions());
  }

  // No valid method of computing the means.
  template <typename DummyType = ModelType,
            typename std::enable_if<
                !has_mean<Prediction<
                    DummyType, FeatureType,
                    typename fit_type<DummyType, FeatureType>::type>>::value &&
                    !has_valid_cv_mean<DummyType, FeatureType, GroupKey>::value,
                int>::type = 0>
  Grouped<GroupKey, Eigen::VectorXd> means() const = delete;

  template <typename DummyType = ModelType,
            typename std::enable_if<
                has_mean<Prediction<
                    DummyType, FeatureType,
                    typename fit_type<DummyType, FeatureType>::type>>::value ||
                    has_valid_cv_mean<DummyType, FeatureType, GroupKey>::value,
                int>::type = 0>
  Eigen::VectorXd mean() const {
    return concatenate_mean_predictions(indexer_, means());
  }

  // No valid method of computing the means.
  template <typename DummyType = ModelType,
            typename std::enable_if<
                !has_mean<Prediction<
                    DummyType, FeatureType,
                    typename fit_type<DummyType, FeatureType>::type>>::value &&
                    !has_valid_cv_mean<DummyType, FeatureType, GroupKey>::value,
                int>::type = 0>

  Eigen::VectorXd mean() const = delete;

  // Marginal

  // Cross validation specialized marginals()
  template <typename DummyType = ModelType,
            typename std::enable_if<
                has_valid_cv_marginal<DummyType, FeatureType, GroupKey>::value,
                int>::type = 0>
  Grouped<GroupKey, MarginalDistribution> marginals() const {
    return model_.cross_validated_predictions(
        dataset_, indexer_, PredictTypeIdentity<MarginalDistribution>());
  }

  // Generic marginals()
  template <
      typename DummyType = ModelType,
      typename std::enable_if<
          has_marginal<Prediction<
              DummyType, FeatureType,
              typename fit_type<DummyType, FeatureType>::type>>::value &&
              !has_valid_cv_marginal<DummyType, FeatureType, GroupKey>::value,
          int>::type = 0>
  Grouped<GroupKey, MarginalDistribution> marginals() const {
    return get_marginals(predictions());
  }

  // No valid way of computing marginals.
  template <
      typename DummyType = ModelType,
      typename std::enable_if<
          !has_marginal<Prediction<
              DummyType, FeatureType,
              typename fit_type<DummyType, FeatureType>::type>>::value &&
              !has_valid_cv_marginal<DummyType, FeatureType, GroupKey>::value,
          int>::type = 0>
  MarginalDistribution marginals() const = delete;

  // No valid way of computing marginals.
  template <
      typename DummyType = ModelType,
      typename std::enable_if<
          has_marginal<Prediction<
              DummyType, FeatureType,
              typename fit_type<DummyType, FeatureType>::type>>::value ||
              has_valid_cv_marginal<DummyType, FeatureType, GroupKey>::value,
          int>::type = 0>
  MarginalDistribution marginal() const {
    return concatenate_marginal_predictions(indexer_, marginals());
  }

  // No valid way of computing marginals.
  template <
      typename DummyType = ModelType,
      typename std::enable_if<
          !has_marginal<Prediction<
              DummyType, FeatureType,
              typename fit_type<DummyType, FeatureType>::type>>::value &&
              !has_valid_cv_marginal<DummyType, FeatureType, GroupKey>::value,
          int>::type = 0>
  MarginalDistribution marginal() const = delete;

  // JOINT

  // Cross validation specific joints().
  template <typename DummyType = ModelType,
            typename std::enable_if<
                has_valid_cv_joint<DummyType, FeatureType, GroupKey>::value,
                int>::type = 0>
  Grouped<GroupKey, JointDistribution> joints() const {
    return model_.cross_validated_predictions(
        dataset_, indexer_, PredictTypeIdentity<JointDistribution>());
  }

  // Generic joints().
  template <
      typename DummyType = ModelType,
      typename std::enable_if<
          has_joint<Prediction<
              DummyType, FeatureType,
              typename fit_type<DummyType, FeatureType>::type>>::value &&
              !has_valid_cv_joint<DummyType, FeatureType, GroupKey>::value,
          int>::type = 0>
  Grouped<GroupKey, JointDistribution> joints() const {
    return get_joints(predictions());
  }

  // No valid way of computing joints().
  template <
      typename DummyType = ModelType,
      typename std::enable_if<
          !has_joint<Prediction<
              DummyType, FeatureType,
              typename fit_type<DummyType, FeatureType>::type>>::value &&
              !has_valid_cv_joint<DummyType, FeatureType, GroupKey>::value,
          int>::type = 0>
  Grouped<GroupKey, JointDistribution> joints() const = delete;

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

  auto get(get_type<std::map<GroupKey, Eigen::VectorXd>>) const {
    return this->means().get_map();
  }

  auto get(get_type<Grouped<GroupKey, Eigen::VectorXd>>) const {
    return this->means();
  }

  auto get(get_type<MarginalDistribution>) const { return this->marginal(); }

  auto get(get_type<std::map<GroupKey, MarginalDistribution>>) const {
    return this->marginals().get_map();
  }

  auto get(get_type<Grouped<GroupKey, MarginalDistribution>>) const {
    return this->marginals();
  }

  auto get(get_type<JointDistribution>) const { return this->joint(); }

  auto get(get_type<std::map<GroupKey, JointDistribution>>) const {
    return this->joints();
  }

  auto get(get_type<Grouped<GroupKey, JointDistribution>>) const {
    return this->joints().get_map();
  }

  const ModelType model_;
  const RegressionDataset<FeatureType> dataset_;
  const GroupIndexer<GroupKey> indexer_;
};

/*
 * Cross Validation
 */

template <typename ModelType, typename FeatureType, typename GroupKey>
using CVPrediction =
    Prediction<CrossValidation<ModelType>, FeatureType, GroupIndexer<GroupKey>>;

template <typename ModelType> class CrossValidation {

  ModelType model_;

public:
  CrossValidation(const ModelType &model) : model_(model) {}

  // Predict

  template <typename FeatureType, typename GroupKey>
  CVPrediction<ModelType, FeatureType, GroupKey>
  predict(const RegressionDataset<FeatureType> &dataset,
          const GroupIndexer<GroupKey> &indexer) const {
    return CVPrediction<ModelType, FeatureType, GroupKey>(model_, dataset,
                                                          indexer);
  }

  template <typename FeatureType, typename GrouperFunction>
  auto predict(const RegressionDataset<FeatureType> &dataset,
               const GrouperFunction &grouper_function) const {
    return predict(dataset, dataset.group_by(grouper_function).indexers());
  }

  // Predictions

  template <typename FeatureType, typename GrouperFunction>
  auto predictions(const RegressionDataset<FeatureType> &dataset,
                   const GrouperFunction &grouper) const {
    return predict(dataset, grouper).predictions();
  }

  template <typename GroupKey, typename FeatureType>
  auto predictions(const RegressionFolds<GroupKey, FeatureType> &folds) const {
    const auto predict_one_fold = [&](const auto &, const auto &fold) {
      return predict_fold(model_, fold);
    };
    return folds.apply(predict_one_fold);
  }

  // Scores

  template <typename RequiredPredictType, typename GroupKey,
            typename FeatureType>
  Eigen::VectorXd
  scores(const PredictionMetric<RequiredPredictType> &metric,
         const RegressionFolds<GroupKey, FeatureType> &folds) const {
    const auto preds = predictions(folds);
    return cross_validated_scores(metric, folds, preds);
  }

  template <typename RequiredPredictType, typename FeatureType,
            typename GroupKey>
  Eigen::VectorXd scores(const PredictionMetric<RequiredPredictType> &metric,
                         const RegressionDataset<FeatureType> &dataset,
                         const GroupIndexer<GroupKey> &indexer) const {
    const auto folds = folds_from_group_indexer(dataset, indexer);
    const auto prediction = predict(dataset, indexer);
    const auto predictions_ =
        prediction.template get<Grouped<GroupKey, RequiredPredictType>>();
    return cross_validated_scores(metric, folds, predictions_);
  }

  template <typename RequiredPredictType, typename FeatureType,
            typename IndexFunc>
  Eigen::VectorXd scores(const PredictionMetric<RequiredPredictType> &metric,
                         const RegressionDataset<FeatureType> &dataset,
                         const IndexFunc &index_function) const {
    const auto indexer = group_by(dataset, index_function).indexers();
    return scores(metric, dataset, indexer);
  }
};

template <typename ModelType>
CrossValidation<ModelType> ModelBase<ModelType>::cross_validate() const {
  return CrossValidation<ModelType>(derived());
}

} // namespace albatross

#endif
