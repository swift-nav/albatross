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

  Fit(){};

  Fit(const std::vector<FeatureType> &features_,
      const CovarianceRepresentation &train_covariance_,
      const Eigen::VectorXd &information_)
      : train_features(features_), train_covariance(train_covariance_),
        information(information_) {}

  Fit(const std::vector<FeatureType> &features,
      const Eigen::MatrixXd &train_cov, const MarginalDistribution &targets) {

    train_features = features;
    Eigen::MatrixXd cov(train_cov);
    cov += targets.covariance;
    ALBATROSS_ASSERT(!cov.hasNaN());
    train_covariance = CovarianceRepresentation(cov);
    information = train_covariance.solve(targets.mean);
  }

  bool operator==(
      const Fit<GPFit<CovarianceRepresentation, FeatureType>> &other) const {
    return (train_features == other.train_features &&
            train_covariance == other.train_covariance &&
            information == other.information);
  }
};

/*
 * Gaussian Process Helper Functions.
 */
inline Eigen::VectorXd gp_mean_prediction(const Eigen::MatrixXd &cross_cov,
                                          const Eigen::VectorXd &information) {
  return cross_cov.transpose() * information;
}

template <typename CovarianceRepresentation>
inline MarginalDistribution
gp_marginal_prediction(const Eigen::MatrixXd &cross_cov,
                       const Eigen::VectorXd &prior_variance,
                       const Eigen::VectorXd &information,
                       const CovarianceRepresentation &train_covariance) {
  const Eigen::VectorXd pred = gp_mean_prediction(cross_cov, information);
  // Here we efficiently only compute the diagonal of the posterior
  // covariance matrix.
  Eigen::MatrixXd explained = train_covariance.solve(cross_cov);
  Eigen::VectorXd explained_variance =
      explained.cwiseProduct(cross_cov).array().colwise().sum();
  Eigen::VectorXd marginal_variance = prior_variance - explained_variance;
  return MarginalDistribution(pred, marginal_variance);
}

template <typename CovarianceRepresentation>
inline JointDistribution
gp_joint_prediction(const Eigen::MatrixXd &cross_cov,
                    const Eigen::MatrixXd &prior_cov,
                    const Eigen::VectorXd &information,
                    const CovarianceRepresentation &train_covariance) {
  const Eigen::VectorXd pred = gp_mean_prediction(cross_cov, information);
  Eigen::MatrixXd explained_cov =
      cross_cov.transpose() * train_covariance.solve(cross_cov);
  return JointDistribution(pred, prior_cov - explained_cov);
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
        model_name_(default_model_name(covariance_function_, mean_function_)){};

  GaussianProcessBase(const CovFunc &covariance_function)
      : covariance_function_(covariance_function), mean_function_(),
        model_name_(default_model_name(covariance_function_, mean_function_)){};
  GaussianProcessBase(CovFunc &&covariance_function)
      : covariance_function_(std::move(covariance_function)), mean_function_(),
        model_name_(default_model_name(covariance_function_, mean_function_)){};

  GaussianProcessBase(const CovFunc &covariance_function,
                      const MeanFunc &mean_function)
      : covariance_function_(covariance_function),
        mean_function_(mean_function),
        model_name_(default_model_name(covariance_function_, mean_function_)){};
  GaussianProcessBase(CovFunc &&covariance_function, MeanFunc &&mean_function)
      : covariance_function_(std::move(covariance_function)),
        mean_function_(std::move(mean_function)),
        model_name_(default_model_name(covariance_function_, mean_function_)){};

  /*
   * Sometimes it's nice to be able to provide a custom model name since
   * these models are generalizable.
   */
  GaussianProcessBase(const CovFunc &covariance_function,
                      const std::string &model_name)
      : covariance_function_(covariance_function), mean_function_(),
        model_name_(model_name){};
  GaussianProcessBase(CovFunc &&covariance_function, std::string &model_name)
      : covariance_function_(std::move(covariance_function)), mean_function_(),
        model_name_(model_name){};

  GaussianProcessBase(const std::string &model_name)
      : covariance_function_(), mean_function_(), model_name_(model_name){};

  GaussianProcessBase(const CovFunc &covariance_function,
                      const MeanFunc &mean_function,
                      const std::string &model_name)
      : covariance_function_(covariance_function),
        mean_function_(mean_function), model_name_(model_name){};
  GaussianProcessBase(CovFunc &&covariance_function, MeanFunc &&mean_function,
                      const std::string &model_name)
      : covariance_function_(std::move(covariance_function)),
        mean_function_(std::move(mean_function)), model_name_(model_name){};

  ~GaussianProcessBase(){};

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
      ALBATROSS_FAIL(FeatureType, "CovFunc is not defined for FeatureType");

  template <
      typename FeatureType, typename FitFeaturetype,
      typename CovarianceRepresentation,
      typename std::enable_if<
          has_call_operator<CovFunc, FeatureType, FeatureType>::value &&
              has_call_operator<CovFunc, FeatureType, FitFeaturetype>::value,
          int>::type = 0>
  JointDistribution _predict_impl(
      const std::vector<FeatureType> &features,
      const GPFitType<CovarianceRepresentation, FitFeaturetype> &gp_fit,
      PredictTypeIdentity<JointDistribution> &&) const {
    const auto cross_cov =
        covariance_function_(gp_fit.train_features, features);
    Eigen::MatrixXd prior_cov = covariance_function_(features);
    auto pred = gp_joint_prediction(cross_cov, prior_cov, gp_fit.information,
                                    gp_fit.train_covariance);
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
    auto pred = gp_marginal_prediction(
        cross_cov, prior_variance, gp_fit.information, gp_fit.train_covariance);
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
    auto pred = gp_mean_prediction(cross_cov, gp_fit.information);
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
          "CovFunc is not defined for FeatureType and FitFeatureType");

  template <typename Solver, typename FeatureType, typename UpdateFeatureType>
  auto _update_impl(const Fit<GPFit<Solver, FeatureType>> &fit_,
                    const std::vector<UpdateFeatureType> &features,
                    const MarginalDistribution &targets) const {

    const auto new_features = concatenate(fit_.train_features, features);

    auto pred = this->_predict_impl(features, fit_,
                                    PredictTypeIdentity<JointDistribution>());

    Eigen::VectorXd delta = targets.mean - pred.mean;
    pred.covariance += targets.covariance;
    const auto S_ldlt = pred.covariance.ldlt();

    const Eigen::MatrixXd cross =
        covariance_function_(fit_.train_features, features);

    const auto new_covariance =
        build_block_symmetric(fit_.train_covariance, cross, S_ldlt);

    const Eigen::VectorXd Si_delta = S_ldlt.solve(delta);

    Eigen::VectorXd new_information(new_covariance.rows());
    new_information.topRows(fit_.train_covariance.rows()) =
        fit_.information - new_covariance.Ai_B * Si_delta;
    new_information.bottomRows(S_ldlt.rows()) = Si_delta;

    using NewFeatureType = typename decltype(new_features)::value_type;
    using NewFitType = Fit<GPFit<BlockSymmetric<Solver>, NewFeatureType>>;
    return NewFitType(new_features, new_covariance, new_information);
  }

  CovFunc get_covariance() const { return covariance_function_; }

  MeanFunc get_mean() const { return mean_function_; };

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
  return details::held_out_predictions(gp_fit.train_covariance,
                                       dataset.targets.mean, gp_fit.information,
                                       group_indexer, predict_type);
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
