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

namespace albatross {

using Insights = std::map<std::string, std::string>;

constexpr bool DEFAULT_USE_ASYNC = false;

template <typename ModelType>
class ModelBase : public ParameterHandlingMixin {
  friend class JointPredictor;
  friend class MarginalPredictor;
  friend class MeanPredictor;

  template <typename T, typename FeatureType>
  friend class fit_model_type;

 protected:
  ModelBase()
      : insights(),
        threads_(DEFAULT_USE_ASYNC
                     ? std::make_shared<ThreadPool>(std::max(
                           std::size_t{1},
                           std::size_t{std::thread::hardware_concurrency()}))
                     : nullptr) {}

  /*
   * Fit
   */
  template <typename FeatureType,
            typename std::enable_if<
                has_valid_fit<ModelType, FeatureType>::value, int>::type = 0>
  auto _fit(const std::vector<FeatureType> &features,
            const MarginalDistribution &targets) const {
    auto fit_output = derived()._fit_impl(features, targets);
    return FitModel<ModelType, decltype(fit_output)>(derived(),
                                                     std::move(fit_output));
  }

  template <
      typename FeatureType,
      typename std::enable_if<has_possible_fit<ModelType, FeatureType>::value &&
                                  !has_valid_fit<ModelType, FeatureType>::value,
                              int>::type = 0>
  void _fit(const std::vector<FeatureType> &features ALBATROSS_UNUSED,
            const MarginalDistribution &targets ALBATROSS_UNUSED) const
      ALBATROSS_FAIL(FeatureType,
                     "The ModelType *almost* has a _fit_impl method for "
                     "FeatureType, but it appears to be invalid")

          template <typename FeatureType,
                    typename std::enable_if<
                        !has_possible_fit<ModelType, FeatureType>::value &&
                            !has_valid_fit<ModelType, FeatureType>::value,
                        int>::type = 0>
          void _fit(const std::vector<FeatureType> &features ALBATROSS_UNUSED,
                    const MarginalDistribution &targets ALBATROSS_UNUSED) const
      ALBATROSS_FAIL(
          FeatureType,
          "The ModelType is missing a _fit_impl method for FeatureType.")

          template <typename PredictFeatureType, typename FitType,
                    typename PredictType,
                    typename std::enable_if<
                        has_valid_predict<ModelType, PredictFeatureType,
                                          FitType, PredictType>::value,
                        int>::type = 0>
          PredictType
      predict_(const std::vector<PredictFeatureType> &features,
               const FitType &fit_, PredictTypeIdentity<PredictType> &&) const {
    return derived()._predict_impl(features, fit_,
                                   PredictTypeIdentity<PredictType>());
  }

// The `ALBATROSS_FAIL` macro leads to a mysterious error with GCC 6
// that does not appear in later versions.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
  template <
      typename PredictFeatureType, typename FitType, typename PredictType,
      typename std::enable_if<!has_valid_predict<ModelType, PredictFeatureType,
                                                 FitType, PredictType>::value,
                              int>::type = 0>
  PredictType predict_(const std::vector<PredictFeatureType> &features
                           ALBATROSS_UNUSED,
                       const FitType &fit ALBATROSS_UNUSED,
                       PredictTypeIdentity<PredictType> &&) const
      ALBATROSS_FAIL(PredictFeatureType,
                     "The ModelType is missing a _predict_impl method for "
                     "PredictFeatureType, FitType, PredictType.")
#pragma GCC diagnostic pop

          public :
      /*
       * CRTP Helpers
       */
      ModelType &derived() {
    return *static_cast<ModelType *>(this);
  }
  const ModelType &derived() const {
    return *static_cast<const ModelType *>(this);
  }

  bool operator==(const ModelType &other) const {
    return (derived().get_params() == other.get_params() &&
            derived().get_name() == other.get_name() &&
            derived().insights == other.insights);
  }

  template <typename DummyType = ModelType,
            typename std::enable_if<!has_name<DummyType>::value, int>::type = 0>
  std::string get_name() {
    return typeid(ModelType).name();
  }

  template <typename DummyType = ModelType,
            typename std::enable_if<has_name<DummyType>::value, int>::type = 0>
  std::string get_name() {
    return derived().name();
  }

  void set_thread_pool(std::shared_ptr<ThreadPool> new_pool) {
    threads_ = new_pool;
  }

  template <typename FeatureType>
  auto fit(const std::vector<FeatureType> &features,
           const MarginalDistribution &targets) const {
    return _fit(features, targets);
  }

  template <typename FeatureType>
  auto fit(const RegressionDataset<FeatureType> &dataset) const {
    return _fit(dataset.features, dataset.targets);
  }

  template <typename FeatureX, typename FeatureY>
  auto fit(const RegressionDataset<FeatureX> &x,
           const RegressionDataset<FeatureY> &y) const {
    return fit(concatenate_datasets(x, y));
  }

  CrossValidation<ModelType> cross_validate() const;

  template <typename Strategy>
  Ransac<ModelType, Strategy> ransac(const Strategy &strategy,
                                     double inlier_threshold,
                                     std::size_t random_sample_size,
                                     std::size_t min_consensus_size,
                                     std::size_t max_iteration) const;

  template <typename Strategy>
  Ransac<ModelType, Strategy> ransac(const Strategy &strategy,
                                     const RansacConfig &) const;

  Insights insights;
  std::shared_ptr<ThreadPool> threads_;
};
}  // namespace albatross
#endif
