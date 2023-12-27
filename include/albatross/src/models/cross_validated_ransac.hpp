/*
 * Copyright (C) 2022 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef ALBATROSS_MODELS_CROSS_VALIDATED_RANSAC_H
#define ALBATROSS_MODELS_CROSS_VALIDATED_RANSAC_H

#include <albatross/Ransac>

namespace albatross {

template <typename StrategyType, typename ModelType, typename FeatureType>
auto
ransac(const StrategyType &strategy,
       const ModelType &model,
       const RegressionFold<FeatureType> &fold,
       const RansacConfig &config) {

  const auto all_data = albatross::concatenate_datasets(
      fold.train_dataset, fold.test_dataset);

  auto indexer = strategy.get_indexer(all_data);
  auto ransac_functions = strategy(model, all_data);

  const auto ransac_output = ransac(ransac_functions, indexer, config);
  using GroupKey = typename decltype(ransac_output)::key_type;

  for (const auto &iteration : ransac_output.iterations) {
    const auto consensus_inds = indices_from_groups(indexer, iteration.consensus());
    std::vector<std::size_t> train_inds;
    std::vector<std::size_t> test_inds;
    for (const auto &ind : consensus_inds) {
      if (ind < fold.train_dataset.size()) {
        train_inds.push_back(ind);
      } else {
        test_inds.push_back(ind);
      }
    }
    const auto train_consensus_data = albatross::subset(all_data, train_inds);
    const auto test_consensus_data = albatross::subset(all_data, test_inds);

    model.fit()


  }

  const auto consensus = iteration.consensus();
  if (consensus.size() >= min_consensus_size) {
    iteration.consensus_metric_value =
        ransac_functions.consensus_metric(consensus);
    const bool best_so_far =
        std::isnan(output.best.consensus_metric_value) ||
        iteration.consensus_metric_value < output.best.consensus_metric_value;
    if (best_so_far) {
      output.best = RansacIteration<GroupKey>(iteration);
    }
  }
  std::vector<RansacIteration<GroupKey>> iterations;


}

//template <typename ModelType>
//inline IonosphereDataset clean_dataset_using_neighborhood_ransac(
//    const ModelType &model,
//    const RegressionFold<IonosphereFeature> &neighborhood_fold,
//    const IonosphereModelConfig &config,
//    std::vector<metrics::Metric> *metrics) {
//
//  assert(config.ransac_config.has_value());
//  const RansacConfig ransac_config = config.ransac_config.value();
//
//  RegressionFold<IonosphereFeature> clean_fold = neighborhood_fold;
//
//  const auto original_data = albatross::concatenate_datasets(
//      neighborhood_fold.train_dataset, neighborhood_fold.test_dataset);
//  const auto all_data = remove_poorly_observed(original_data, config);
//
//  const std::string query_station =
//      neighborhood_fold.test_dataset.features[0].station_id;
//
//  const albatross::RansacConfig albatross_ransac_config =
//      convert_ransac_config(ransac_config, all_data.size());
//
//  albatross::ChiSquaredCdf chi2;
//  FeatureCountIfReasonableConsensusMetric feature_count;
//  WorstCaseChiSquaredIsValidCandidateMetric is_valid;
//
//  auto per_arc_grouper = [](const SignalArc &arc) {
//    return arc.station_id + "_" + arc.satellite_id.to_string();
//  };
//
//  const auto ransac_strategy = multi_shell_gp_ransac_strategy(
//      chi2, feature_count, is_valid, per_arc_grouper);
//
//  auto indexer = ransac_strategy.get_indexer(all_data);
//  auto ransac_functions = ransac_strategy(model, all_data);
//  const auto full_ransac_output =
//      ransac(ransac_functions, indexer, albatross_ransac_config);
//
//  auto only_ransac_output_for_query = [&](const albatross::RansacOutput<
//                                          std::string> &output) {
//    const auto query_station_groups = albatross::apply(
//        neighborhood_fold.test_dataset.features, per_arc_grouper);
//    std::set<std::string> unique_query_station_groups(
//        query_station_groups.begin(), query_station_groups.end());
//
//    auto is_query_key = [&](const std::string &key) {
//      return set_contains(unique_query_station_groups, key);
//    };
//
//    auto is_query_key_value = [&](const std::string &key, const auto &) {
//      return is_query_key(key);
//    };
//
//    albatross::RansacOutput<std::string> reduced(output);
//
//    reduced.best.candidates =
//        albatross::filter(output.best.candidates, is_query_key);
//    reduced.best.inliers =
//        albatross::filter(output.best.inliers, is_query_key_value).get_map();
//    reduced.best.outliers =
//        albatross::filter(output.best.outliers, is_query_key_value).get_map();
//    return reduced;
//  };
//
//  const auto ransac_output = only_ransac_output_for_query(full_ransac_output);
//
//  if (!albatross::ransac_success(ransac_output.return_code)) {
//    report_ransac_failure(ransac_output.return_code,
//                          keys::modeler::IONOSPHERE_MODEL, metrics);
//    return IonosphereDataset();
//  } else {
//    report_ransac_success(ransac_output, keys::modeler::IONOSPHERE_MODEL,
//                          metrics);
//    const auto good_inds =
//        indices_from_groups(indexer, ransac_output.best.consensus());
//
//    const auto clean_data = subset(all_data, good_inds);
//
//    const auto clean_station_data =
//        clean_data.group_by(feature_to_station<IonosphereFeature>)
//            .get_group(query_station);
//
//    return clean_station_data;
//  }
//
//  return IonosphereDataset();
//}

}

#endif /* ALBATROSS_MODELS_CROSS_VALIDATED_RANSAC_H */
