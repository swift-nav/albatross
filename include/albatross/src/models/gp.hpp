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

#ifndef ALBATROSS_MODELS_GP_H
#define ALBATROSS_MODELS_GP_H

namespace albatross {

template <typename CovarianceRepresentation, typename FeatureType>
struct GPFit {};

inline Eigen::MatrixXd selector_matrix(Eigen::Index n_train,
                                       std::vector<std::size_t> indices) {
  Eigen::Index n_cols = albatross::cast::to_index(indices.size());
  Eigen::MatrixXd E{Eigen::MatrixXd::Zero(n_train, n_cols)};
  for (Eigen::Index col = 0; col < n_cols; ++col) {
    const std::size_t idx = albatross::cast::to_size(col);
    E(indices[idx], col) = 1.0;
  }
  return E;
}

// Precomputed data for making corrections to predictive
// distributions, so that we can remove observations after we have
// done the big expensive fit.
struct GPRemoveCache {
  // {z}
  std::vector<std::size_t> indices;
  // K^{-1}[x, z]
  Eigen::MatrixXd precision_train_removed;
  // K_{zz}
  //
  // TODO(@peddie): perhaps better to store this as a Cholesky factor,
  // because after precomputation, we may get more stable products if
  // we do a quadratic form rather than solving to one side.
  Eigen::MatrixXd conditional_cov_removed;
  // conditional_cov_removed * information_removed
  Eigen::VectorXd delta_removed;

  [[nodiscard]] inline operator bool() const { return !indices.empty(); }

  [[nodiscard]] inline bool operator==(const GPRemoveCache &other) const {
    // Deep-compare every field so a serialization round-trip that
    // silently drops a cached matrix fails equality; the matrix
    // fields are deterministic functions of indices + the fit's
    // covariance/information, so this stays consistent with the
    // fast-path "same indices ⟹ same cache" invariant.
    return indices == other.indices &&
           precision_train_removed == other.precision_train_removed &&
           conditional_cov_removed == other.conditional_cov_removed &&
           delta_removed == other.delta_removed;
  }

  GPRemoveCache() = default;

  template <typename CovarianceRepresentation>
  GPRemoveCache(const std::vector<std::size_t> &z,
                const CovarianceRepresentation &train_covariance,
                const Eigen::VectorXd &train_information)
      : indices{z} {
    if (!(*this)) {
      return;
    }
    const Eigen::Index n_train{train_covariance.rows()};
    const Eigen::Index n_removed{albatross::cast::to_index(indices.size())};

    const Eigen::MatrixXd E_removed{selector_matrix(n_train, indices)};

    precision_train_removed = train_covariance.solve(E_removed);

    Eigen::MatrixXd precision_removed =
        albatross::subset_rows(precision_train_removed, indices);
    Eigen::SerializableLDLT P_zz_ldlt(precision_removed);
    conditional_cov_removed =
        P_zz_ldlt.solve(Eigen::MatrixXd::Identity(n_removed, n_removed));

    Eigen::VectorXd v_removed = albatross::subset(train_information, indices);

    delta_removed = conditional_cov_removed * v_removed;
  }

  [[nodiscard]] inline Eigen::MatrixXd
  compute_W(const Eigen::MatrixXd &cross) const {
    ALBATROSS_ASSERT(*this &&
                     "Cannot compute W with no removed training data.");
    return cross.transpose() * precision_train_removed;
  }

  // Corrections are to be added to the full term.
  //
  // The only thing we need is W, which depends on the
  // cross-covariance matrix.  You can either compute it with
  // `compute_W()` directly, or use a subset of the `explained()`
  // matrix if you are computing explained uncertainty already.

  [[nodiscard]] inline Eigen::VectorXd
  mean_correction(const Eigen::MatrixXd &W) const {
    ALBATROSS_ASSERT(
        *this &&
        "Cannot compute mean correction with no removed training data.");
    return -W * delta_removed;
  }

  [[nodiscard]] inline Eigen::VectorXd
  marginal_correction(const Eigen::MatrixXd &W) const {
    ALBATROSS_ASSERT(
        *this &&
        "Cannot compute mean correction with no removed training data.");
    return (W * conditional_cov_removed).cwiseProduct(W).rowwise().sum();
  }

  [[nodiscard]] inline Eigen::MatrixXd
  joint_correction(const Eigen::MatrixXd &W) const {
    ALBATROSS_ASSERT(
        *this &&
        "Cannot compute mean correction with no removed training data.");
    return W * conditional_cov_removed * W.transpose();
  }
};

/*
 * The Gaussian Process needs to store the training data and the
 * prior covariance for those data points.  One way to do that would
 * be to simply store the entire RegressionDataset (features and targets)
 * as well as the covariance matrix between all features.  The way this
 * ends up getting used however makes it more efficient to store the
 * information, defined as:
 *
 *   information = prior_covariance^{-1} * y
 *
 * where
 *
 *   y = targets.mean
 *
 * and
 *
 *   prior_covariance = cov(features, features) + targets.covariance.
 *
 * Furthermore, rather than store the entire prior covariance we just
 * store the cholesky (LDLT).
 */
template <typename CovarianceRepresentation, typename FeatureType>
struct Fit<GPFit<CovarianceRepresentation, FeatureType>> {
  static_assert(has_solve<CovarianceRepresentation, Eigen::MatrixXd>::value,
                "CovarianceRepresentation must have a solve method");

  using Feature = FeatureType;

  std::vector<FeatureType> train_features;
  CovarianceRepresentation train_covariance;
  Eigen::VectorXd information;
  GPRemoveCache removed;

  Fit() {}

  Fit(const std::vector<FeatureType> &features_,
      const CovarianceRepresentation &train_covariance_,
      const Eigen::VectorXd &information_,
      const std::vector<std::size_t> remove_indices_ =
          std::vector<std::size_t>{})
      : train_features(features_), train_covariance(train_covariance_),
        information(information_),
        removed(remove_indices_, train_covariance, information) {}

  Fit(const std::vector<FeatureType> &features,
      const Eigen::MatrixXd &train_cov, const MarginalDistribution &targets,
      const std::vector<std::size_t> remove_indices_ =
          std::vector<std::size_t>{}) {
    train_features = features;
    Eigen::MatrixXd cov(train_cov);
    cov += targets.covariance;
    ALBATROSS_ASSERT(!cov.hasNaN());
    train_covariance = CovarianceRepresentation(cov);
    information = train_covariance.solve(targets.mean);
    removed = GPRemoveCache(remove_indices_, train_covariance, information);
  }

  bool operator==(
      const Fit<GPFit<CovarianceRepresentation, FeatureType>> &other) const {
    return (train_features == other.train_features &&
            train_covariance == other.train_covariance &&
            information == other.information && removed == other.removed);
  }
};

template <typename X>
std::vector<std::size_t> match_features(const std::vector<X> &train_data,
                                        const std::vector<X> &remove) {
  std::vector<std::size_t> remove_indices;
  for (std::size_t i = 0; i < train_data.size(); ++i) {
    if (std::find(remove.begin(), remove.end(), train_data[i]) !=
        remove.end()) {
      remove_indices.push_back(i);
    }
  }
  return remove_indices;
}

struct UpdateOverlapPartition {
  // The remove-set indices that survive the partition.  Entries of
  // `remove_indices` that matched an incoming update feature are
  // "resurrected" and dropped here.
  std::vector<std::size_t> reduced_remove_indices;
  // Positions in the incoming update vector for features that do NOT
  // match a currently-removed training row.  These are the features
  // that still need to be folded into the fit.
  std::vector<std::size_t> truly_new_update_indices;
};

// Partition an incoming update feature list against an existing
// remove-cache.  When the update feature type matches the training
// feature type, features that coincide with a currently-removed
// training row are treated as resurrects (removed from the cache,
// not appended); everything else is truly new.  When the types
// differ there can be no overlap by construction, so every incoming
// feature is truly new.
template <typename TrainFeatureType, typename UpdateFeatureType>
UpdateOverlapPartition partition_update_against_removed(
    const std::vector<TrainFeatureType> &train_features,
    const std::vector<std::size_t> &remove_indices,
    const std::vector<UpdateFeatureType> &update_features) {
  UpdateOverlapPartition out;
  out.reduced_remove_indices = remove_indices;
  out.truly_new_update_indices.reserve(update_features.size());
  if constexpr (std::is_same<TrainFeatureType, UpdateFeatureType>::value) {
    for (std::size_t ui = 0; ui < update_features.size(); ++ui) {
      const auto it =
          std::find_if(out.reduced_remove_indices.begin(),
                       out.reduced_remove_indices.end(), [&](std::size_t ri) {
                         return train_features[ri] == update_features[ui];
                       });
      if (it != out.reduced_remove_indices.end()) {
        out.reduced_remove_indices.erase(it);
      } else {
        out.truly_new_update_indices.push_back(ui);
      }
    }
  } else {
    out.truly_new_update_indices.resize(update_features.size());
    std::iota(out.truly_new_update_indices.begin(),
              out.truly_new_update_indices.end(), std::size_t{0});
  }
  return out;
}

// Build a no-op `Fit<GPFit<BlockSymmetric<Solver>, ...>>` that wraps
// an existing fit's covariance, information and features and carries
// the given (possibly-reduced) remove indices.  Used when an update
// consists entirely of resurrected features: the block-symmetric
// return type must still match the general update path, but there is
// no new data to fold in so the bottom block is empty.
template <typename Solver, typename FeatureType>
Fit<GPFit<BlockSymmetric<Solver>, FeatureType>> trivial_update_shortcut_fit(
    const Fit<GPFit<Solver, FeatureType>> &fit,
    const std::vector<std::size_t> &reduced_remove_indices) {
  const Eigen::Index n_train = fit.train_covariance.rows();
  BlockSymmetric<Solver> trivial_covariance(fit.train_covariance,
                                            Eigen::MatrixXd::Zero(n_train, 0),
                                            Eigen::SerializableLDLT());
  return Fit<GPFit<BlockSymmetric<Solver>, FeatureType>>(
      fit.train_features, trivial_covariance, fit.information,
      reduced_remove_indices);
}

/*
 * Gaussian Process Helper Functions.
 */
inline Eigen::VectorXd gp_mean_prediction(const Eigen::MatrixXd &cross_cov,
                                          const Eigen::VectorXd &information) {
  return cross_cov.transpose() * information;
}

inline Eigen::VectorXd gp_mean_prediction(const Eigen::MatrixXd &cross_cov,
                                          const Eigen::VectorXd &information,
                                          const GPRemoveCache &removed) {
  Eigen::VectorXd full_mean = gp_mean_prediction(cross_cov, information);
  if (removed) {
    full_mean += removed.mean_correction(removed.compute_W(cross_cov));
  }
  return full_mean;
}

template <typename CovarianceRepresentation>
inline MarginalDistribution
gp_marginal_prediction(const Eigen::MatrixXd &cross_cov,
                       const Eigen::VectorXd &prior_variance,
                       const Eigen::VectorXd &information,
                       const CovarianceRepresentation &train_covariance,
                       const GPRemoveCache &removed = {}) {
  Eigen::MatrixXd explained = train_covariance.solve(cross_cov);

  // Here we efficiently only compute the diagonal of the posterior
  // covariance matrix.
  Eigen::VectorXd explained_variance =
      explained.cwiseProduct(cross_cov).array().colwise().sum();
  Eigen::VectorXd marginal_variance = prior_variance - explained_variance;
  Eigen::VectorXd pred = gp_mean_prediction(cross_cov, information);

  if (removed) {
    Eigen::MatrixXd W = subset_rows(explained, removed.indices).transpose();
    pred += removed.mean_correction(W);
    marginal_variance += removed.marginal_correction(W);
  }

  return MarginalDistribution(pred, marginal_variance);
}

template <typename CovarianceRepresentation>
inline JointDistribution
gp_joint_prediction(const Eigen::MatrixXd &cross_cov,
                    const Eigen::MatrixXd &prior_cov,
                    const Eigen::VectorXd &information,
                    const CovarianceRepresentation &train_covariance,
                    const GPRemoveCache &removed = {}) {
  Eigen::VectorXd pred = gp_mean_prediction(cross_cov, information);
  Eigen::MatrixXd explained = train_covariance.solve(cross_cov);
  Eigen::MatrixXd explained_cov = cross_cov.transpose() * explained;

  if (removed) {
    Eigen::MatrixXd W = subset_rows(explained, removed.indices).transpose();
    pred += removed.mean_correction(W);
    // Subtract here because we are going to then subtract the
    // explained covariance from the prior.
    explained_cov -= removed.joint_correction(W);
  }

  return JointDistribution(pred, prior_cov - explained_cov);
}

// Joint prediction at `features` for a Gaussian Process fit, with an
// explicit remove cache decoupled from whatever cache the fit is
// carrying.  `_predict_impl` delegates here with `gp_fit.removed`;
// `_update_impl` borrows the same helper with an empty cache so it
// can use the full-fit prediction when computing delta without
// having to copy the fit just to swap out the cache.
template <typename CovFunc, typename MeanFunc, typename FeatureType,
          typename FitFeatureType, typename CovarianceRepresentation>
inline JointDistribution gp_joint_prediction_with_removed(
    const CovFunc &covariance_function, const MeanFunc &mean_function,
    const std::vector<FeatureType> &features,
    const Fit<GPFit<CovarianceRepresentation, FitFeatureType>> &gp_fit,
    const GPRemoveCache &removed) {
  const auto cross_cov = covariance_function(gp_fit.train_features, features);
  Eigen::MatrixXd prior_cov = covariance_function(features);
  auto pred = gp_joint_prediction(cross_cov, prior_cov, gp_fit.information,
                                  gp_fit.train_covariance, removed);
  mean_function.add_to(features, &pred.mean);
  return pred;
}

/*
 * These functions create a covariance matrix solver which
 * is used when building a new Gaussian process based off a
 * predicted covariance, P, from an external model.
 *
 * The idea is that a prediction for some test locations, *,
 * for a model fit with features, u, would have the form:
 *
 *   mean(*)       : K_*u v
 *   covariance(*) : K_** - K_*u C^-1 K_u*
 *
 * Our goal with this function is to find the matrix C for which
 * a query for the same features as the input would yield the
 * input prediction covariance.  Ie,
 *
 *   P = covariance(u)
 *     = K_uu - K_uu C^-1 K_uu
 *     = K_uu - K_uu K_uu^-1 (K_uu - P) K_uu^-1 K_uu
 *     = K_uu - (K_uu - P)
 *     = P
 *
 * Thus by setting C = K_uu (K_uu - P)^-1 K_uu we can build a
 * GP which recovers the prediction.
 */
template <typename FeatureType>
inline Fit<GPFit<ExplainedCovariance, FeatureType>>
gp_fit_from_prediction(const std::vector<FeatureType> &features,
                       const Eigen::MatrixXd &prior,
                       const JointDistribution &prediction) {
  Fit<GPFit<ExplainedCovariance, FeatureType>> fit;
  fit.train_features = features;
  fit.train_covariance =
      ExplainedCovariance(prior, prior - prediction.covariance);
  fit.information = prior.ldlt().solve(prediction.mean);
  return fit;
}

template <typename CovFunc, typename MeanFunc>
std::string default_model_name(const CovFunc &cov_func,
                               const MeanFunc &mean_func) {
  return "mean:" + mean_func.get_name() + "cov:" + cov_func.get_name();
}

/*
 * This GaussianProcessBase will provide a model which is capable of
 * producing fits and predicting for any FeatureType that is supported
 * by the covariance function.  Sometimes, however, you may want to
 * do some preprocessing of features since that operation could
 * happen in order N time, instead of repeatedly preprocessing every
 * time the covariance function is evaluated.  To do this you'll want
 * to define a custom ImplType.  See test_models.cc for an example.
 */
template <typename CovFunc, typename MeanFunc, typename ImplType>
class GaussianProcessBase : public ModelBase<ImplType> {
protected:
  using Base = ModelBase<ImplType>;

  template <typename CovarianceRepresentation, typename FitFeatureType>
  using GPFitType = Fit<GPFit<CovarianceRepresentation, FitFeatureType>>;

  template <typename FitFeatureType>
  using CholeskyFit = GPFitType<Eigen::SerializableLDLT, FitFeatureType>;

public:
  GaussianProcessBase()
      : covariance_function_(), mean_function_(),
        model_name_(default_model_name(covariance_function_, mean_function_)) {}

  GaussianProcessBase(const CovFunc &covariance_function)
      : covariance_function_(covariance_function), mean_function_(),
        model_name_(default_model_name(covariance_function_, mean_function_)) {}
  GaussianProcessBase(CovFunc &&covariance_function)
      : covariance_function_(std::move(covariance_function)), mean_function_(),
        model_name_(default_model_name(covariance_function_, mean_function_)) {}

  GaussianProcessBase(const CovFunc &covariance_function,
                      const MeanFunc &mean_function)
      : covariance_function_(covariance_function),
        mean_function_(mean_function),
        model_name_(default_model_name(covariance_function_, mean_function_)) {}
  GaussianProcessBase(CovFunc &&covariance_function, MeanFunc &&mean_function)
      : covariance_function_(std::move(covariance_function)),
        mean_function_(std::move(mean_function)),
        model_name_(default_model_name(covariance_function_, mean_function_)) {}

  /*
   * Sometimes it's nice to be able to provide a custom model name since
   * these models are generalizable.
   */
  GaussianProcessBase(const CovFunc &covariance_function,
                      const std::string &model_name)
      : covariance_function_(covariance_function), mean_function_(),
        model_name_(model_name) {}
  GaussianProcessBase(CovFunc &&covariance_function, std::string &model_name)
      : covariance_function_(std::move(covariance_function)), mean_function_(),
        model_name_(model_name) {}

  GaussianProcessBase(const std::string &model_name)
      : covariance_function_(), mean_function_(), model_name_(model_name) {}

  GaussianProcessBase(const CovFunc &covariance_function,
                      const MeanFunc &mean_function,
                      const std::string &model_name)
      : covariance_function_(covariance_function),
        mean_function_(mean_function), model_name_(model_name) {}
  GaussianProcessBase(CovFunc &&covariance_function, MeanFunc &&mean_function,
                      const std::string &model_name)
      : covariance_function_(std::move(covariance_function)),
        mean_function_(std::move(mean_function)), model_name_(model_name) {}

  ~GaussianProcessBase() {}

  static const std::uint32_t serialization_version = 1;

  // Create a fit based on a subset of predicted features (with the given joint
  // distribution) - the fit type also requires the prior covariance of the
  // features to determine the model that will give the same mean and covariance
  // for the initially predicted features
  template <typename FeatureType>
  auto fit_from_prediction(const std::vector<FeatureType> &features,
                           const JointDistribution &prediction) const {
    JointDistribution zero_mean_prediction(prediction);
    mean_function_.remove_from(features, &zero_mean_prediction.mean);
    using FitType = Fit<GPFit<ExplainedCovariance, FeatureType>>;
    return FitModel<ImplType, FitType>(
        impl(), gp_fit_from_prediction(features, covariance_function_(features),
                                       prediction));
  }

  std::string get_name() const { return model_name_; };

  void set_name(const std::string &model_name) { model_name_ = model_name; };

  /*
   * The Gaussian Process Regression model derives its parameters from
   * the covariance functions.
   */
  virtual ParameterStore get_params() const override {
    return map_join(impl().params_,
                    map_join(mean_function_.get_params(),
                             covariance_function_.get_params()));
  }

  virtual void set_param(const std::string &name,
                         const Parameter &param) override {
    const bool success =
        (set_param_if_exists(name, param, &impl().params_) ||
         set_param_if_exists_in_any(name, param, &covariance_function_,
                                    &mean_function_));
    ALBATROSS_ASSERT(success);
  }

  std::string pretty_string() const {
    std::ostringstream ss;
    ss << "model_name: " << get_name() << std::endl;
    ss << "covariance_name: " << covariance_function_.get_name() << std::endl;
    ss << "mean_name: " << mean_function_.get_name() << std::endl;
    ss << "params: " << pretty_params(impl().get_params());
    return ss.str();
  }

  // If the implementing class defines a _fit_impl method these will be
  // hidden, so they deal with the default case.
  template <
      typename FeatureType,
      std::enable_if_t<
          has_call_operator<CovFunc, FeatureType, FeatureType>::value, int> = 0>
  CholeskyFit<FeatureType>
  _fit_impl(const std::vector<FeatureType> &features,
            const MarginalDistribution &targets) const {
    const auto measurement_features = as_measurements(features);
    Eigen::MatrixXd cov =
        covariance_function_(measurement_features, Base::threads_.get());
    MarginalDistribution zero_mean_targets(targets);
    mean_function_.remove_from(measurement_features, &zero_mean_targets.mean);
    return CholeskyFit<FeatureType>(features, cov, zero_mean_targets);
  }

  // If the covariance is NOT defined.
  template <typename FeatureType,
            typename std::enable_if<
                !has_call_operator<CovFunc, FeatureType, FeatureType>::value,
                int>::type = 0>
  void _fit_impl(const std::vector<FeatureType> &features ALBATROSS_UNUSED,
                 const MarginalDistribution &targets ALBATROSS_UNUSED) const
      ALBATROSS_FAIL(FeatureType, "CovFunc is not defined for FeatureType")

          template <
              typename FeatureType, typename FitFeaturetype,
              typename CovarianceRepresentation,
              typename std::enable_if<
                  has_call_operator<CovFunc, FeatureType, FeatureType>::value &&
                      has_call_operator<CovFunc, FeatureType,
                                        FitFeaturetype>::value,
                  int>::type = 0>
          JointDistribution _predict_impl(
              const std::vector<FeatureType> &features,
              const GPFitType<CovarianceRepresentation, FitFeaturetype> &gp_fit,
              PredictTypeIdentity<JointDistribution> &&) const {
    return gp_joint_prediction_with_removed(
        covariance_function_, mean_function_, features, gp_fit, gp_fit.removed);
  }

  template <
      typename FeatureType, typename FitFeaturetype,
      typename CovarianceRepresentation,
      typename std::enable_if<
          has_call_operator<CovFunc, FeatureType, FeatureType>::value &&
              has_call_operator<CovFunc, FeatureType, FitFeaturetype>::value,
          int>::type = 0>
  MarginalDistribution _predict_impl(
      const std::vector<FeatureType> &features,
      const GPFitType<CovarianceRepresentation, FitFeaturetype> &gp_fit,
      PredictTypeIdentity<MarginalDistribution> &&) const {
    const auto cross_cov =
        covariance_function_(gp_fit.train_features, features);
    Eigen::VectorXd prior_variance(cast::to_index(features.size()));
    for (Eigen::Index i = 0; i < prior_variance.size(); ++i) {
      prior_variance[i] = covariance_function_(features[cast::to_size(i)],
                                               features[cast::to_size(i)]);
    }
    auto pred =
        gp_marginal_prediction(cross_cov, prior_variance, gp_fit.information,
                               gp_fit.train_covariance, gp_fit.removed);
    mean_function_.add_to(features, &pred.mean);
    return pred;
  }

  template <
      typename FeatureType, typename FitFeaturetype,
      typename CovarianceRepresentation,
      typename std::enable_if<
          has_call_operator<CovFunc, FeatureType, FeatureType>::value &&
              has_call_operator<CovFunc, FeatureType, FitFeaturetype>::value,
          int>::type = 0>
  Eigen::VectorXd _predict_impl(
      const std::vector<FeatureType> &features,
      const GPFitType<CovarianceRepresentation, FitFeaturetype> &gp_fit,
      PredictTypeIdentity<Eigen::VectorXd> &&) const {
    const auto cross_cov =
        covariance_function_(gp_fit.train_features, features);
    auto pred =
        gp_mean_prediction(cross_cov, gp_fit.information, gp_fit.removed);
    mean_function_.add_to(features, &pred);
    return pred;
  }

  template <
      typename FeatureType, typename FitFeatureType, typename PredictType,
      typename CovarianceRepresentation,
      typename std::enable_if<
          !has_call_operator<CovFunc, FeatureType, FeatureType>::value ||
              !has_call_operator<CovFunc, FeatureType, FitFeatureType>::value,
          int>::type = 0>
  PredictType _predict_impl(
      const std::vector<FeatureType> &features ALBATROSS_UNUSED,
      const GPFitType<CovarianceRepresentation, FitFeatureType> &gp_fit
          ALBATROSS_UNUSED,
      PredictTypeIdentity<PredictType> &&) const
      ALBATROSS_FAIL(
          FeatureType,
          "CovFunc is not defined for FeatureType and FitFeatureType")

          template <typename Solver, typename FeatureType,
                    typename UpdateFeatureType>
          auto _update_impl(const Fit<GPFit<Solver, FeatureType>> &fit_,
                            const std::vector<UpdateFeatureType> &features,
                            const MarginalDistribution &targets) const {
    // Partition the update set against the remove-cache.  Matching
    // features are resurrected (dropped from the remove-cache, not
    // appended); everything else is folded into the fit via the
    // usual block-symmetric update.
    const auto partition = partition_update_against_removed(
        fit_.train_features, fit_.removed.indices, features);

    std::vector<UpdateFeatureType> new_features_to_add =
        albatross::subset(features, partition.truly_new_update_indices);

    // Shortcut: every incoming feature resurrected a removed row, so
    // there is no new data to fold in.  Return a no-op
    // block-symmetric wrap carrying the reduced remove-cache.  This
    // branch compiles only for matching feature types, since the
    // partition can only resurrect when types match.
    if constexpr (std::is_same<FeatureType, UpdateFeatureType>::value) {
      if (new_features_to_add.empty()) {
        return trivial_update_shortcut_fit(fit_,
                                           partition.reduced_remove_indices);
      }
    }

    const auto new_features =
        concatenate(fit_.train_features, new_features_to_add);

    const MarginalDistribution filtered_targets =
        targets.subset(partition.truly_new_update_indices);

    // The block-symmetric update derives the new information vector
    // assuming delta = targets - full-fit-prediction (no removal
    // correction).  Routing through `_predict_impl` would apply the
    // existing remove cache to the prediction, producing a biased
    // delta whenever the fit carries a non-empty cache.  Use the
    // explicit-cache free helper with an empty cache so the math
    // stays correct regardless of the cache state going in; the
    // reduced cache is attached to the output fit below.
    auto pred = gp_joint_prediction_with_removed(
        covariance_function_, mean_function_, new_features_to_add, fit_,
        GPRemoveCache());

    Eigen::VectorXd delta = filtered_targets.mean - pred.mean;
    pred.covariance += filtered_targets.covariance;
    const auto S_ldlt = pred.covariance.ldlt();

    const Eigen::MatrixXd cross =
        covariance_function_(fit_.train_features, new_features_to_add);

    const auto new_covariance =
        build_block_symmetric(fit_.train_covariance, cross, S_ldlt);

    const Eigen::VectorXd Si_delta = S_ldlt.solve(delta);

    Eigen::VectorXd new_information(new_covariance.rows());
    new_information.topRows(fit_.train_covariance.rows()) =
        fit_.information - new_covariance.Ai_B * Si_delta;
    new_information.bottomRows(S_ldlt.rows()) = Si_delta;

    using NewFeatureType = typename decltype(new_features)::value_type;
    using NewFitType = Fit<GPFit<BlockSymmetric<Solver>, NewFeatureType>>;
    return NewFitType(new_features, new_covariance, new_information,
                      partition.reduced_remove_indices);
  }

  // Observation removal for dense GPs is implemented via a small
  // correction applied after the full prediction is made.  This
  // allows us to avoid modifying the (potentially large) sequential
  // covariance decomposition, but it also assumes that the number of
  // removed features is small relative to the number of training
  // features.  If you try to remove a lot of features, performance
  // may suffer.
  template <typename Solver, typename FeatureType>
  auto _prune_impl(const Fit<GPFit<Solver, FeatureType>> &fit_,
                   const std::vector<std::size_t> &remove_indices) const {
    // Just record the indices; the remove-cache build happens inside
    // the Fit<GPFit<...>> constructor.
    return Fit<GPFit<Solver, FeatureType>>(fit_.train_features,
                                           fit_.train_covariance,
                                           fit_.information, remove_indices);
  }

  CovFunc get_covariance() const { return covariance_function_; }

  MeanFunc get_mean() const { return mean_function_; }

  template <typename FeatureType>
  JointDistribution prior(const std::vector<FeatureType> &features) const {
    const auto measurement_features = as_measurements(features);
    return JointDistribution(mean_function_(measurement_features),
                             covariance_function_(measurement_features));
  }

  template <typename FeatureType>
  Eigen::MatrixXd
  compute_covariance(const std::vector<FeatureType> &features) const {
    return covariance_function_(features);
  }

  template <typename FeatureType,
            std::enable_if_t<has_valid_state_space_representation<
                                 CovFunc, FeatureType>::value,
                             int> = 0>
  auto
  state_space_representation(const std::vector<FeatureType> &features) const {
    return covariance_function_.state_space_representation(features);
  }

  template <typename FeatureType>
  double log_likelihood(const RegressionDataset<FeatureType> &dataset) const {
    Eigen::VectorXd zero_mean(dataset.targets.mean);
    const auto measurement_features = as_measurements(dataset.features);
    mean_function_.remove_from(measurement_features, &zero_mean);
    const Eigen::MatrixXd cov = covariance_function_(measurement_features);
    double ll = -negative_log_likelihood(zero_mean, cov);
    ll += this->prior_log_likelihood();
    return ll;
  }

protected:
  /*
   * CRTP Helpers
   */
  ImplType &impl() { return *static_cast<ImplType *>(this); }
  const ImplType &impl() const { return *static_cast<const ImplType *>(this); }

  CovFunc covariance_function_;
  MeanFunc mean_function_;
  std::string model_name_;
};

template <typename GroupKey, typename FeatureType, typename GPType,
          typename PredictType>
inline std::map<GroupKey, PredictType>
gp_cross_validated_predictions(const RegressionDataset<FeatureType> &dataset,
                               const GroupIndexer<GroupKey> &group_indexer,
                               const GPType &model,
                               PredictTypeIdentity<PredictType> predict_type) {
  const auto fit_model = model.fit(dataset);
  const auto gp_fit = fit_model.get_fit();
  // Note: it might look like we forgot to apply the mean function here,
  // but we don't actually need to use it, the information vector will
  // have been formed by taking the mean function into account and the
  // held out predictions will use that to derive deltas from the truth
  // so removing the mean, then adding it back later is unneccesary
  return details::held_out_predictions(
      gp_fit.train_covariance, dataset.targets.mean, gp_fit.information,
      group_indexer, predict_type, model.threads_.get());
}

// Subset a held-out-prediction result down to a list of positions
// within the augmented group.  One overload per PredictType so the
// fit-driven CV below can stay generic.
inline Eigen::VectorXd
subset_prediction(const Eigen::VectorXd &v,
                  const std::vector<std::size_t> &positions) {
  return subset(v, positions);
}

inline MarginalDistribution
subset_prediction(const MarginalDistribution &d,
                  const std::vector<std::size_t> &positions) {
  return d.subset(positions);
}

inline JointDistribution
subset_prediction(const JointDistribution &d,
                  const std::vector<std::size_t> &positions) {
  return JointDistribution(subset(d.mean, positions),
                           symmetric_subset(d.covariance, positions));
}

// Fit-driven dense-GP cross-validation.  Takes an already-fitted
// model so the caller can pass in a fit carrying a non-empty
// GPRemoveCache; the sibling above refits from the dataset every
// call, so that path never sees a cache.
//
// For each CV group, the group's indices are unioned with the fit's
// remove-cache indices before invoking the standard block-inverse
// machinery, and the resulting prediction is subset back down to the
// original group.  The LOO closed form generalises to any leave-out
// set S: mean_S = y_S - (K^{-1}[S,S])^{-1} v[S], cov = (K^{-1}[S,S])^{-1}.
// Setting S = R u z_g and restricting the result to z_g's positions
// gives "fit on X \ R, leave z_g out".  When the cache is empty the
// augmented indexer equals the original and this reduces exactly to
// the standard path.
//
// Targets come in separately: Fit<GPFit<...>> drops raw y after
// forming information = K^{-1} y, and the mean-prediction formula
// genuinely needs y, so it has to be supplied by the caller.
template <typename GPType, typename CovarianceRepresentation,
          typename FitFeatureType, typename GroupKey, typename PredictType>
inline std::map<GroupKey, PredictType> gp_cross_validated_predictions(
    const FitModel<GPType, Fit<GPFit<CovarianceRepresentation, FitFeatureType>>>
        &fit_model,
    const MarginalDistribution &targets,
    const GroupIndexer<GroupKey> &group_indexer,
    PredictTypeIdentity<PredictType> predict_type) {
  const auto &gp_fit = fit_model.get_fit();
  auto *pool = fit_model.get_model().threads_.get();

  if (!gp_fit.removed) {
    return details::held_out_predictions(gp_fit.train_covariance, targets.mean,
                                         gp_fit.information, group_indexer,
                                         predict_type, pool);
  }

  // Build an augmented indexer: each group's indices union-ed with
  // the fit's remove-cache indices.  std::set_union requires sorted
  // inputs; the cache indices are stored in construction order (sorted)
  // but the group indices may not be, so defensively sort a copy.
  GroupIndexer<GroupKey> augmented_indexer;
  for (const auto &kv : group_indexer) {
    std::vector<std::size_t> sorted_group = kv.second;
    std::sort(sorted_group.begin(), sorted_group.end());
    std::vector<std::size_t> merged;
    merged.reserve(sorted_group.size() + gp_fit.removed.indices.size());
    std::set_union(sorted_group.begin(), sorted_group.end(),
                   gp_fit.removed.indices.begin(), gp_fit.removed.indices.end(),
                   std::back_inserter(merged));
    augmented_indexer.emplace(kv.first, std::move(merged));
  }

  const auto augmented_predictions = details::held_out_predictions(
      gp_fit.train_covariance, targets.mean, gp_fit.information,
      augmented_indexer, predict_type, pool);

  std::map<GroupKey, PredictType> out;
  for (const auto &kv : augmented_predictions) {
    const auto &orig = group_indexer.at(kv.first);
    const auto &aug = augmented_indexer.at(kv.first);
    std::vector<std::size_t> positions;
    positions.reserve(orig.size());
    for (const auto i : orig) {
      const auto it = std::find(aug.begin(), aug.end(), i);
      ALBATROSS_ASSERT(it != aug.end());
      positions.push_back(
          static_cast<std::size_t>(std::distance(aug.begin(), it)));
    }
    out.emplace(kv.first, subset_prediction(kv.second, positions));
  }
  return out;
}

/*
 * Generic Gaussian Process Implementation.
 */
template <typename CovFunc, typename MeanFunc>
class GaussianProcessRegression
    : public GaussianProcessBase<CovFunc, MeanFunc,
                                 GaussianProcessRegression<CovFunc, MeanFunc>> {
public:
  using Base =
      GaussianProcessBase<CovFunc, MeanFunc,
                          GaussianProcessRegression<CovFunc, MeanFunc>>;
  using Base::Base;

  template <typename FeatureType, typename PredictType, typename GroupKey>
  std::map<GroupKey, PredictType>
  cross_validated_predictions(const RegressionDataset<FeatureType> &dataset,
                              const GroupIndexer<GroupKey> &group_indexer,
                              PredictTypeIdentity<PredictType> identity) const {
    return gp_cross_validated_predictions(dataset, group_indexer, *this,
                                          identity);
  }
};

template <typename CovFunc>
auto gp_from_covariance(CovFunc &&covariance_function,
                        const std::string &model_name) {
  return GaussianProcessRegression<typename std::decay<CovFunc>::type>(
      std::forward<CovFunc>(covariance_function), model_name);
};

template <typename CovFunc>
auto gp_from_covariance(CovFunc &&covariance_function) {
  return GaussianProcessRegression<typename std::decay<CovFunc>::type>(
      std::forward<CovFunc>(covariance_function));
};

template <typename CovFunc, typename MeanFunc>
auto gp_from_covariance_and_mean(CovFunc &&covariance_function,
                                 MeanFunc &&mean_func) {
  return GaussianProcessRegression<typename std::decay<CovFunc>::type,
                                   typename std::decay<MeanFunc>::type>(
      std::forward<CovFunc>(covariance_function),
      std::forward<MeanFunc>(mean_func));
};

template <typename CovFunc, typename MeanFunc>
auto gp_from_covariance_and_mean(CovFunc &&covariance_function,
                                 MeanFunc &&mean_func,
                                 const std::string &model_name) {
  return GaussianProcessRegression<typename std::decay<CovFunc>::type,
                                   typename std::decay<MeanFunc>::type>(
      std::forward<CovFunc>(covariance_function),
      std::forward<MeanFunc>(mean_func), model_name);
};

/*
 * Model Metric
 */
struct GaussianProcessNegativeLogLikelihood {
  template <typename FeatureType, typename CovFunc, typename MeanFunc,
            typename GPImplType>
  double operator()(
      const RegressionDataset<FeatureType> &dataset,
      const GaussianProcessBase<CovFunc, MeanFunc, GPImplType> &model) const {
    return -model.log_likelihood(dataset);
  }
};

} // namespace albatross

#endif
