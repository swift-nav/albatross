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

#ifndef ALBATROSS_CORE_OUTLIER_H
#define ALBATROSS_CORE_OUTLIER_H

#include "core/dataset.h"
#include "core/indexing.h"
#include "core/model.h"
#include "crossvalidation.h"
#include "random_utils.h"
#include <functional>
#include <random>
#include <set>

namespace albatross {

using Indexer = std::vector<std::size_t>;
using GroupIndexer = std::vector<std::vector<std::size_t>>;

Indexer concatenate_subset_of_groups(const Indexer &subset_indices,
                                     const GroupIndexer &indexer) {

  Indexer output;
  for (const auto i : subset_indices) {
    assert(i < static_cast<std::size_t>(indexer.size()));
    output.insert(output.end(), indexer[i].begin(), indexer[i].end());
  }
  return output;
}

/*
 * This RANdom SAmple Consensus (RANSAC) algorithm works as follows.
 *
 *   1) Randomly sample a small number of data points and fit a
 *      reference model to that data.
 *   2) Assemble all the data points that agree with the
 *      reference model into a set of inliers.
 *   3) Evaluate the quality of the inliers.
 *   4) Repeat N times keeping track of the best set of inliers.
 *
 * One of the large drawbacks of this approach is the computational
 * load since it requires fitting and predicting repeatedly.
 * The goal of this implementation is to provide a way
 * for the user to optionally perform a lot of computation upfront,
 * then use call backs which take indices as input to selectively
 * update/downdate the model to produce the fits and evaluation
 * metrics.
 */
template <typename FitType>
Indexer
ransac(std::function<FitType(const Indexer &)> &fitter,
       std::function<double(const Indexer &, const FitType &)> &outlier_metric,
       std::function<double(const Indexer &)> &model_metric,
       const GroupIndexer &indexer, double threshold, std::size_t min_features,
       std::size_t min_inliers, std::size_t max_iterations) {

  std::default_random_engine gen;

  Indexer reference;
  double best_metric = HUGE_VAL;
  Indexer best_inds;

  for (std::size_t i = 0; i < max_iterations; i++) {
    // Sample a random subset of the data and fit a model.
    reference =
        randint_without_replacement(min_features, 0, indexer.size() - 1, gen);
    auto ref_inds = concatenate_subset_of_groups(reference, indexer);
    const auto fit = fitter(ref_inds);

    // Find which of the other groups agree with the reference model
    auto test_groups = indices_complement(reference, indexer.size());
    Indexer inliers;
    for (const auto &test_ind : test_groups) {
      double metric_value = outlier_metric(indexer[test_ind], fit);
      if (metric_value < threshold) {
        inliers.push_back(test_ind);
      }
    }

    // If there is enough agreement, consider this random set of inliers
    // as a candidate model.
    if (inliers.size() > min_inliers) {
      const auto inlier_inds = concatenate_subset_of_groups(inliers, indexer);
      ref_inds.insert(ref_inds.end(), inlier_inds.begin(), inlier_inds.end());
      std::sort(ref_inds.begin(), ref_inds.end());
      double model_metric_value = model_metric(ref_inds);
      if (model_metric_value < best_metric) {
        best_inds = ref_inds;
        best_metric = model_metric_value;
      }
    }
  }
  assert(best_metric < HUGE_VAL);
  return best_inds;
}

template <typename FeatureType, typename PredictType>
RegressionDataset<FeatureType>
ransac(const RegressionDataset<FeatureType> &dataset,
       const FoldIndexer &fold_indexer, RegressionModel<FeatureType> *model,
       EvaluationMetric<PredictType> &metric, double threshold,
       std::size_t min_features, std::size_t min_inliers, int max_iterations) {

  using FitType = RegressionModel<FeatureType> *;

  using FitFunc = std::function<FitType(const std::vector<std::size_t> &)>;
  using OutlierFunc =
      std::function<double(const std::vector<std::size_t> &, const FitType &)>;
  using ModelMetricFunc =
      std::function<double(const std::vector<std::size_t> &)>;

  FitFunc fitter = [&](const std::vector<std::size_t> &inds) {
    RegressionDataset<FeatureType> dataset_subset(
        subset(inds, dataset.features), subset(inds, dataset.targets));
    model->fit(dataset_subset);
    return model;
  };

  OutlierFunc outlier_metric = [&](const std::vector<std::size_t> &inds,
                                   const FitType &fit) {
    const auto pred = fit->predict(subset(inds, dataset.features));
    const auto target = subset(inds, dataset.targets);
    double metric_value = metric(pred, target);
    return metric_value;
  };

  ModelMetricFunc model_metric = [&](const std::vector<std::size_t> &inds) {
    RegressionDataset<FeatureType> inlier_dataset(
        subset(inds, dataset.features), subset(inds, dataset.targets));
    const auto inlier_loo = leave_one_out_indexer(inlier_dataset);
    return cross_validated_scores(metric, inlier_dataset, inlier_loo, model)
        .mean();

  };

  const auto best_inds = ransac<FitType>(
      fitter, outlier_metric, model_metric, map_values(fold_indexer), threshold,
      min_features, min_inliers, max_iterations);
  RegressionDataset<FeatureType> best_dataset(
      subset(best_inds, dataset.features), subset(best_inds, dataset.targets));
  return best_dataset;
}

} // namespace albatross

#endif
