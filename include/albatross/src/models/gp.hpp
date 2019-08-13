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
    if (targets.has_covariance()) {
      cov += targets.covariance;
    }
    assert(!cov.hasNaN());
    train_covariance = CovarianceRepresentation(cov);
    // Precompute the information vector
    information = train_covariance.solve(targets.mean);
  }

  template <typename Archive> void serialize(Archive &archive) {
    archive(cereal::make_nvp("information", information));
    archive(cereal::make_nvp("train_ldlt", train_covariance));
    archive(cereal::make_nvp("train_features", train_features));
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
  return MarginalDistribution(pred, marginal_variance.asDiagonal());
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

/*
 * This GaussianProcessBase will provide a model which is capable of
 * producing fits and predicting for any FeatureType that is supported
 * by the covariance function.  Sometimes, however, you may want to
 * do some preprocessing of features since that operation could
 * happen in order N time, instead of repeatedly preprocessing every
 * time the covariance function is evaluated.  To do this you'll want
 * to define a custom ImplType.  See test_models.cc for an example.
 */
template <typename CovFunc, typename ImplType>
class GaussianProcessBase : public ModelBase<ImplType> {

protected:
  template <typename CovarianceRepresentation, typename FitFeatureType>
  using GPFitType = Fit<GPFit<CovarianceRepresentation, FitFeatureType>>;

  template <typename FitFeatureType>
  using CholeskyFit = GPFitType<Eigen::SerializableLDLT, FitFeatureType>;

public:
  GaussianProcessBase()
      : covariance_function_(), model_name_(covariance_function_.get_name()){};
  GaussianProcessBase(const CovFunc &covariance_function)
      : covariance_function_(covariance_function),
        model_name_(covariance_function_.get_name()){};
  /*
   * Sometimes it's nice to be able to provide a custom model name since
   * these models are generalizable.
   */
  GaussianProcessBase(const CovFunc &covariance_function,
                      const std::string &model_name)
      : covariance_function_(covariance_function), model_name_(model_name){};
  GaussianProcessBase(const std::string &model_name)
      : covariance_function_(), model_name_(model_name){};

  ~GaussianProcessBase(){};

  // Create a fit based on a subset of predicted features (with the given joint
  // distribution) - the fit type also requires the prior covariance of the
  // features to determine the model that will give the same mean and covariance
  // for the initially predicted features
  template <typename FeatureType>
  auto fit_from_prediction(const std::vector<FeatureType> &features,
                           const JointDistribution &prediction) const {

    using FitType = Fit<GPFit<ExplainedCovariance, FeatureType>>;
    return FitModel<ImplType, FitType>(
        impl(), gp_fit_from_prediction(features, covariance_function_(features),
                                       prediction));
  }

  std::string get_name() const { return model_name_; };

  /*
   * The Gaussian Process Regression model derives its parameters from
   * the covariance functions.
   */
  ParameterStore get_params() const override {
    return map_join(impl().params_, covariance_function_.get_params());
  }

  void unchecked_set_param(const std::string &name,
                           const Parameter &param) override {

    if (map_contains(covariance_function_.get_params(), name)) {
      covariance_function_.set_param(name, param);
    } else {
      impl().params_[name] = param;
    }
  }

  std::string pretty_string() const {
    std::ostringstream ss;
    ss << "model_name: " << get_name() << std::endl;
    ss << "covariance_name: " << covariance_function_.get_name() << std::endl;
    ss << "params: " << pretty_params(impl().get_params());
    return ss.str();
  }

  // If the implementing class doesn't have a fit method for this
  // FeatureType but the CovarianceFunction does.
  template <typename FeatureType,
            typename std::enable_if<
                has_call_operator<CovFunc, FeatureType, FeatureType>::value,
                int>::type = 0>
  CholeskyFit<FeatureType>
  _fit_impl(const std::vector<FeatureType> &features,
            const MarginalDistribution &targets) const {
    const auto measurement_features = as_measurements(features);
    Eigen::MatrixXd cov = covariance_function_(measurement_features);
    return CholeskyFit<FeatureType>(features, cov, targets);
  }

  // If the CovarianceFunction is NOT defined.
  template <typename FeatureType,
            typename std::enable_if<
                !has_call_operator<CovFunc, FeatureType, FeatureType>::value,
                int>::type = 0>
  void _fit_impl(const std::vector<FeatureType> &features,
                 const MarginalDistribution &targets) const
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
    return gp_joint_prediction(cross_cov, prior_cov, gp_fit.information,
                               gp_fit.train_covariance);
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
    Eigen::VectorXd prior_variance(static_cast<Eigen::Index>(features.size()));
    for (Eigen::Index i = 0; i < prior_variance.size(); ++i) {
      prior_variance[i] = covariance_function_(features[i], features[i]);
    }
    return gp_marginal_prediction(cross_cov, prior_variance, gp_fit.information,
                                  gp_fit.train_covariance);
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
    return gp_mean_prediction(cross_cov, gp_fit.information);
  }

  template <
      typename FeatureType, typename FitFeatureType, typename PredictType,
      typename CovarianceRepresentation,
      typename std::enable_if<
          !has_call_operator<CovFunc, FeatureType, FeatureType>::value ||
              !has_call_operator<CovFunc, FeatureType, FitFeatureType>::value,
          int>::type = 0>
  PredictType _predict_impl(
      const std::vector<FeatureType> &features,
      const GPFitType<CovarianceRepresentation, FitFeatureType> &gp_fit,
      PredictTypeIdentity<PredictType> &&) const
      ALBATROSS_FAIL(
          FeatureType,
          "CovFunc is not defined for FeatureType and FitFeatureType");

  CovFunc get_covariance() const { return covariance_function_; }

  template <typename FeatureType>
  Eigen::MatrixXd
  compute_covariance(const std::vector<FeatureType> &features) const {
    return covariance_function_(features);
  }

protected:
  /*
   * CRTP Helpers
   */
  ImplType &impl() { return *static_cast<ImplType *>(this); }
  const ImplType &impl() const { return *static_cast<const ImplType *>(this); }

  CovFunc covariance_function_;
  std::string model_name_;
};

template <typename FeatureType, typename GPType>
std::map<std::string, JointDistribution>
gp_cross_validated_predictions(const RegressionDataset<FeatureType> &dataset,
                               const FoldIndexer &fold_indexer,
                               const GPType &model,
                               PredictTypeIdentity<JointDistribution>) {

  const auto fit_model = model.fit(dataset);
  const auto gp_fit = fit_model.get_fit();

  const std::vector<FoldIndices> indices = map_values(fold_indexer);
  const std::vector<std::string> fold_names = map_keys(fold_indexer);
  const auto inverse_blocks = gp_fit.train_covariance.inverse_blocks(indices);

  std::map<std::string, JointDistribution> output;
  for (std::size_t i = 0; i < inverse_blocks.size(); i++) {
    Eigen::VectorXd yi = subset(dataset.targets.mean, indices[i]);
    Eigen::VectorXd vi = subset(gp_fit.information, indices[i]);
    const auto A_inv = inverse_blocks[i].inverse();
    output[fold_names[i]] = JointDistribution(yi - A_inv * vi, A_inv);
  }
  return output;
}

template <typename FeatureType, typename GPType>
std::map<std::string, MarginalDistribution>
gp_cross_validated_predictions(const RegressionDataset<FeatureType> &dataset,
                               const FoldIndexer &fold_indexer,
                               const GPType &model,
                               PredictTypeIdentity<MarginalDistribution>) {

  const auto fit_model = model.fit(dataset);
  const auto gp_fit = fit_model.get_fit();

  const std::vector<FoldIndices> indices = map_values(fold_indexer);
  const std::vector<std::string> fold_names = map_keys(fold_indexer);
  const auto inverse_blocks = gp_fit.train_covariance.inverse_blocks(indices);

  std::map<std::string, MarginalDistribution> output;
  for (std::size_t i = 0; i < inverse_blocks.size(); i++) {
    Eigen::VectorXd yi = subset(dataset.targets.mean, indices[i]);
    Eigen::VectorXd vi = subset(gp_fit.information, indices[i]);
    const auto A_ldlt = Eigen::SerializableLDLT(inverse_blocks[i].ldlt());
    output[fold_names[i]] = MarginalDistribution(
        yi - A_ldlt.solve(vi), A_ldlt.inverse_diagonal().asDiagonal());
  }
  return output;
}

template <typename FeatureType, typename GPType>
std::map<std::string, Eigen::VectorXd>
gp_cross_validated_predictions(const RegressionDataset<FeatureType> &dataset,
                               const FoldIndexer &fold_indexer,
                               const GPType &model,
                               PredictTypeIdentity<Eigen::VectorXd>) {
  const auto fit_model = model.fit(dataset);
  const auto gp_fit = fit_model.get_fit();
  const std::vector<FoldIndices> indices = map_values(fold_indexer);
  const std::vector<std::string> fold_names = map_keys(fold_indexer);
  const auto inverse_blocks = gp_fit.train_covariance.inverse_blocks(indices);

  std::map<std::string, Eigen::VectorXd> output;
  for (std::size_t i = 0; i < inverse_blocks.size(); i++) {
    Eigen::VectorXd yi = subset(dataset.targets.mean, indices[i]);
    Eigen::VectorXd vi = subset(gp_fit.information, indices[i]);
    const auto A_ldlt = Eigen::SerializableLDLT(inverse_blocks[i].ldlt());
    output[fold_names[i]] = yi - A_ldlt.solve(vi);
  }
  return output;
}

/*
 * Generic Gaussian Process Implementation.
 */
template <typename CovFunc>
class GaussianProcessRegression
    : public GaussianProcessBase<CovFunc, GaussianProcessRegression<CovFunc>> {
public:
  using Base = GaussianProcessBase<CovFunc, GaussianProcessRegression<CovFunc>>;
  using Base::Base;

  template <typename FeatureType, typename PredictType>
  std::map<std::string, PredictType>
  cross_validated_predictions(const RegressionDataset<FeatureType> &dataset,
                              const FoldIndexer &fold_indexer,
                              PredictTypeIdentity<PredictType> identity) const {
    return gp_cross_validated_predictions(dataset, fold_indexer, *this,
                                          identity);
  }
};

template <typename CovFunc>
auto gp_from_covariance(CovFunc covariance_function,
                        const std::string &model_name) {
  return GaussianProcessRegression<CovFunc>(covariance_function, model_name);
};

template <typename CovFunc>
auto gp_from_covariance(CovFunc covariance_function) {
  return GaussianProcessRegression<CovFunc>(covariance_function,
                                            covariance_function.get_name());
};

/*
 * Model Metric
 */
struct GaussianProcessLikelihood {

  template <typename FeatureType, typename CovFunc, typename GPImplType>
  double
  operator()(const RegressionDataset<FeatureType> &dataset,
             const GaussianProcessBase<CovFunc, GPImplType> &model) const {
    const auto gp_fit = model.fit(dataset).get_fit();
    double nll =
        negative_log_likelihood(dataset.targets.mean, gp_fit.train_covariance);
    nll -= model.prior_log_likelihood();
    return nll;
  }
};

template <typename ModelType, typename Solver, typename FeatureType,
          typename UpdateFeatureType>
auto update(
    const FitModel<ModelType, Fit<GPFit<Solver, FeatureType>>> &fit_model,
    const RegressionDataset<UpdateFeatureType> &dataset) {

  const auto fit = fit_model.get_fit();
  const auto new_features = concatenate(fit.train_features, dataset.features);

  auto pred = fit_model.predict(dataset.features).joint();

  Eigen::VectorXd delta = dataset.targets.mean - pred.mean;
  if (dataset.targets.has_covariance()) {
    pred.covariance += dataset.targets.covariance;
  }
  const auto S_ldlt = pred.covariance.ldlt();

  const auto model = fit_model.get_model();
  const Eigen::MatrixXd cross =
      model.get_covariance()(fit.train_features, dataset.features);

  const auto new_covariance =
      build_block_symmetric(fit.train_covariance, cross, S_ldlt);

  const Eigen::VectorXd Si_delta = S_ldlt.solve(delta);

  Eigen::VectorXd new_information(new_covariance.rows());
  new_information.topRows(fit.train_covariance.rows()) =
      fit.information - new_covariance.Ai_B * Si_delta;
  new_information.bottomRows(S_ldlt.rows()) = Si_delta;

  using NewFeatureType = typename decltype(new_features)::value_type;
  using NewFitType = Fit<GPFit<BlockSymmetric<Solver>, NewFeatureType>>;

  return FitModel<ModelType, NewFitType>(
      model, NewFitType(new_features, new_covariance, new_information));
}

} // namespace albatross

#endif
