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

#ifndef ALBATROSS_CROSSVALIDATION_H
#define ALBATROSS_CROSSVALIDATION_H

namespace albatross {

/*
 * A combination of training and testing datasets, typically used in cross
 * validation.
 */
template <typename FeatureType> struct RegressionFold {
  RegressionDataset<FeatureType> train_dataset;
  RegressionDataset<FeatureType> test_dataset;
  FoldName name;
  FoldIndices test_indices;

  RegressionFold(const RegressionDataset<FeatureType> &train_dataset_,
                 const RegressionDataset<FeatureType> &test_dataset_,
                 const FoldName &name_, const FoldIndices &test_indices_)
      : train_dataset(train_dataset_), test_dataset(test_dataset_), name(name_),
        test_indices(test_indices_){};
};

template <typename ModelType>
class CrossValidation : public ModelBase<CrossValidation<ModelType>> {



//  // Because cross validation can never properly produce a full
//  // joint distribution it is common to only use the marginal
//  // predictions, hence the different default from predict.
//  template <typename PredictType = MarginalDistribution>
//  std::vector<PredictType> cross_validated_predictions(
//      const std::vector<RegressionFold<FeatureType>> &folds) {
//    // Iteratively make predictions and assemble the output vector
//    std::vector<PredictType> predictions;
//    for (std::size_t i = 0; i < folds.size(); i++) {
//      fit(folds[i].train_dataset);
//      predictions.push_back(
//          predict<PredictType>(folds[i].test_dataset.features));
//    }
//    return predictions;
//  }
//
//  std::vector<JointDistribution> cross_validated_predictions_(
//      const RegressionDataset<FeatureType> &dataset,
//      const FoldIndexer &fold_indexer,
//      const detail::PredictTypeIdentity<JointDistribution> &identity) override {
//
//    this->fit(dataset);
//    const FitType model_fit = this->get_fit();
//    const std::vector<FoldIndices> indices = map_values(fold_indexer);
//    const auto inverse_blocks = model_fit.train_ldlt.inverse_blocks(indices);
//
//    std::vector<JointDistribution> output;
//    for (std::size_t i = 0; i < inverse_blocks.size(); i++) {
//      Eigen::VectorXd yi = subset(indices[i], dataset.targets.mean);
//      Eigen::VectorXd vi = subset(indices[i], model_fit.information);
//      const auto A_inv = inverse_blocks[i].inverse();
//      output.push_back(JointDistribution(yi - A_inv * vi, A_inv));
//    }
//    return output;
//  }
};

template <typename ModelType>
CrossValidation<ModelType> ModelBase<ModelType>::cross_validate() const {
  return CrossValidation<ModelType>();
}

template <typename ModelType, typename FeatureType>
class Prediction<CrossValidation<ModelType>, FeatureType> {

 public:
   Prediction(const CrossValidation<ModelType> &model, const std::vector<FeatureType> &features)
       : model_(model), features_(features) {}

  /*
   * MEAN
   */
  Eigen::VectorXd mean() const {
    return Eigen::VectorXd::Ones(1);
  }

 private:
   const CrossValidation<ModelType> &model_;
   const std::vector<FeatureType> &features_;

};

///*
// * Each flavor of cross validation can be described by a set of
// * FoldIndices, which store which indices should be used for the
// * test cases.  This function takes a map from FoldName to
// * FoldIndices and a dataset and creates the resulting folds.
// */
//template <typename FeatureType>
//static inline std::vector<RegressionFold<FeatureType>>
//folds_from_fold_indexer(const RegressionDataset<FeatureType> &dataset,
//                        const FoldIndexer &groups) {
//  // For a dataset with n features, we'll have n folds.
//  const std::size_t n = dataset.features.size();
//  std::vector<RegressionFold<FeatureType>> folds;
//  // For each fold, partition into train and test sets.
//  for (const auto &pair : groups) {
//    // These get exposed inside the returned RegressionFold and because
//    // we'd like to prevent modification of the output from this function
//    // from changing the input FoldIndexer we perform a copy here.
//    const FoldName group_name(pair.first);
//    const FoldIndices test_indices(pair.second);
//    const auto train_indices = indices_complement(test_indices, n);
//
//    std::vector<FeatureType> train_features =
//        subset(train_indices, dataset.features);
//    MarginalDistribution train_targets = subset(train_indices, dataset.targets);
//
//    std::vector<FeatureType> test_features =
//        subset(test_indices, dataset.features);
//    MarginalDistribution test_targets = subset(test_indices, dataset.targets);
//
//    assert(train_features.size() == train_targets.size());
//    assert(test_features.size() == test_targets.size());
//    assert(test_targets.size() + train_targets.size() == n);
//
//    const RegressionDataset<FeatureType> train_split(train_features,
//                                                     train_targets);
//    const RegressionDataset<FeatureType> test_split(test_features,
//                                                    test_targets);
//    folds.push_back(RegressionFold<FeatureType>(train_split, test_split,
//                                                group_name, test_indices));
//  }
//  return folds;
//}
//
//template <typename FeatureType>
//static inline FoldIndexer
//leave_one_out_indexer(const RegressionDataset<FeatureType> &dataset) {
//  FoldIndexer groups;
//  for (std::size_t i = 0; i < dataset.features.size(); i++) {
//    FoldName group_name = std::to_string(i);
//    groups[group_name] = {i};
//  }
//  return groups;
//}
//
///*
// * Splits a dataset into cross validation folds where each fold contains all but
// * one predictor/target pair.
// */
//template <typename FeatureType>
//static inline FoldIndexer leave_one_group_out_indexer(
//    const std::vector<FeatureType> &features,
//    const std::function<FoldName(const FeatureType &)> &get_group_name) {
//  FoldIndexer groups;
//  for (std::size_t i = 0; i < features.size(); i++) {
//    const std::string k = get_group_name(features[i]);
//    // Get the existing indices if we've already encountered this group_name
//    // otherwise initialize a new one.
//    FoldIndices indices;
//    if (groups.find(k) == groups.end()) {
//      indices = FoldIndices();
//    } else {
//      indices = groups[k];
//    }
//    // Add the current index.
//    indices.push_back(i);
//    groups[k] = indices;
//  }
//  return groups;
//}
//
///*
// * Splits a dataset into cross validation folds where each fold contains all but
// * one predictor/target pair.
// */
//template <typename FeatureType>
//static inline FoldIndexer leave_one_group_out_indexer(
//    const RegressionDataset<FeatureType> &dataset,
//    const std::function<FoldName(const FeatureType &)> &get_group_name) {
//  return leave_one_group_out_indexer(dataset.features, get_group_name);
//}
//
///*
// * Generates cross validation folds which represent leave one out
// * cross validation.
// */
//template <typename FeatureType>
//static inline std::vector<RegressionFold<FeatureType>>
//leave_one_out(const RegressionDataset<FeatureType> &dataset) {
//  return folds_from_fold_indexer<FeatureType>(
//      dataset, leave_one_out_indexer<FeatureType>(dataset));
//}
//
///*
// * Uses a `get_group_name` function to bucket each FeatureType into
// * a group, then holds out one group at a time.
// */
//template <typename FeatureType>
//static inline std::vector<RegressionFold<FeatureType>> leave_one_group_out(
//    const RegressionDataset<FeatureType> &dataset,
//    const std::function<FoldName(const FeatureType &)> &get_group_name) {
//  const FoldIndexer indexer =
//      leave_one_group_out_indexer<FeatureType>(dataset, get_group_name);
//  return folds_from_fold_indexer<FeatureType>(dataset, indexer);
//}
//
/////*
//// * An evaluation metric is a function that takes a prediction distribution and
//// * corresponding targets and returns a single real value that summarizes
//// * the quality of the prediction.
//// */
//// template <typename PredictType>
//// using EvaluationMetric = std::function<double(
////    const PredictType &prediction, const MarginalDistribution &targets)>;
////
/////*
//// * Iterates over previously computed predictions for each fold and
//// * returns a vector of scores for each fold.
//// */
//// template <typename FeatureType, typename PredictType = JointDistribution>
//// static inline Eigen::VectorXd
//// compute_scores(const EvaluationMetric<PredictType> &metric,
////               const std::vector<RegressionFold<FeatureType>> &folds,
////               const std::vector<PredictType> &predictions) {
////  // Create a vector of metrics, one for each fold.
////  Eigen::VectorXd metrics(static_cast<Eigen::Index>(folds.size()));
////  // Loop over each fold, making predictions then evaluating them
////  // to create the final output.
////  for (Eigen::Index i = 0; i < metrics.size(); i++) {
////    metrics[i] = metric(predictions[i], folds[i].test_dataset.targets);
////  }
////  return metrics;
////}
////
//// template <typename FeatureType, typename CovarianceType>
//// static inline Eigen::VectorXd
//// compute_scores(const EvaluationMetric<Eigen::VectorXd> &metric,
////               const std::vector<RegressionFold<FeatureType>> &folds,
////               const std::vector<Distribution<CovarianceType>> &predictions) {
////  std::vector<Eigen::VectorXd> converted;
////  for (const auto &pred : predictions) {
////    converted.push_back(pred.mean);
////  }
////  return compute_scores(metric, folds, converted);
////}
////
/////*
//// * Iterates over each fold in a cross validation set and fits/predicts and
//// * scores the fold, returning a vector of scores for each fold.
//// */
//// template <typename FeatureType, typename PredictType = JointDistribution>
//// static inline Eigen::VectorXd
//// cross_validated_scores(const EvaluationMetric<PredictType> &metric,
////                       const std::vector<RegressionFold<FeatureType>> &folds,
////                       RegressionModel<FeatureType> *model) {
////  // Create a vector of predictions.
////  std::vector<PredictType> predictions =
////      model->template cross_validated_predictions<PredictType>(folds);
////  return compute_scores<FeatureType, PredictType>(metric, folds, predictions);
////}
////
/////*
//// * Iterates over each fold in a cross validation set and fits/predicts and
//// * scores the fold, returning a vector of scores for each fold.
//// */
//// template <typename FeatureType, typename PredictType = JointDistribution>
//// static inline Eigen::VectorXd
//// cross_validated_scores(const EvaluationMetric<PredictType> &metric,
////                       const RegressionDataset<FeatureType> &dataset,
////                       const FoldIndexer &fold_indexer,
////                       RegressionModel<FeatureType> *model) {
////  // Create a vector of predictions.
////  std::vector<PredictType> predictions =
////      model->template cross_validated_predictions<PredictType>(dataset,
////                                                               fold_indexer);
////  const auto folds = folds_from_fold_indexer(dataset, fold_indexer);
////  return compute_scores<FeatureType, PredictType>(metric, folds, predictions);
////}
////
/////*
//// * Returns a single cross validated prediction distribution
//// * for some cross validation folds, taking into account the
//// * fact that each fold may contain reordered data.
//// */
//// template <typename PredictType>
//// static inline MarginalDistribution concatenate_fold_predictions(
////    const FoldIndexer &fold_indexer,
////    const std::map<FoldName, PredictType> &predictions) {
////  // Create a new prediction mean that will eventually contain
////  // the ordered concatenation of each fold's predictions.
////  Eigen::Index n = 0;
////  for (const auto &pair : predictions) {
////    n += static_cast<decltype(n)>(pair.second.size());
////  }
////
////  Eigen::VectorXd mean(n);
////  Eigen::VectorXd diagonal(n);
////
////  Eigen::Index number_filled = 0;
////  // Put all the predicted means back in order.
////  for (const auto &pair : predictions) {
////    const auto pred = pair.second;
////    const auto fold_indices = fold_indexer.at(pair.first);
////    assert(pred.size() == fold_indices.size());
////    for (Eigen::Index i = 0; i < pred.mean.size(); i++) {
////      // The test indices map each element in the current fold back
////      // to the original order of the parent dataset.
////      auto test_ind = static_cast<Eigen::Index>(fold_indices[i]);
////      assert(test_ind < n);
////      mean[test_ind] = pred.mean[i];
////      diagonal[test_ind] = pred.get_diagonal(i);
////      number_filled++;
////    }
////  }
////  assert(number_filled == n);
////  return MarginalDistribution(mean, diagonal.asDiagonal());
////}
////
/////*
//// * Returns a single cross validated prediction distribution
//// * for some cross validation folds, taking into account the
//// * fact that each fold may contain reordered data.
//// */
//// template <typename FeatureType, typename PredictType>
//// static inline MarginalDistribution concatenate_fold_predictions(
////    const std::vector<RegressionFold<FeatureType>> &folds,
////    const std::vector<PredictType> &predictions) {
////
////  // Convert to map variants of the inputs.
////  FoldIndexer fold_indexer;
////  std::map<FoldName, PredictType> prediction_map;
////  for (std::size_t j = 0; j < predictions.size(); j++) {
////    prediction_map[folds[j].name] = predictions[j];
////    fold_indexer[folds[j].name] = folds[j].test_indices;
////  }
////
////  return concatenate_fold_predictions(fold_indexer, prediction_map);
////}
////
//// template <typename FeatureType>
//// static inline MarginalDistribution
//// cross_validated_predict(const std::vector<RegressionFold<FeatureType>>
//// &folds,
////                        RegressionModel<FeatureType> *model) {
////  // Get the cross validated predictions, note however that
////  // depending on the type of folds, these predictions may
////  // be shuffled.
////  const std::vector<MarginalDistribution> predictions =
////      model->template
////      cross_validated_predictions<MarginalDistribution>(folds);
////  return concatenate_fold_predictions(folds, predictions);
////}

} // namespace albatross

#endif
