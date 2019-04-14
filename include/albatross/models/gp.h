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
template <typename CovFunc, typename ImplType, typename FeatureType>
struct Fit<GaussianProcessBase<CovFunc, ImplType>, FeatureType> {

  std::vector<FeatureType> train_features;
  Eigen::SerializableLDLT train_ldlt;
  Eigen::VectorXd information;

  Fit(){};

  Fit(const std::vector<FeatureType> &features,
      const Eigen::MatrixXd &train_cov, const MarginalDistribution &targets) {
    train_features = features;
    Eigen::MatrixXd cov(train_cov);
    if (targets.has_covariance()) {
      cov += targets.covariance;
    }
    assert(!cov.hasNaN());
    train_ldlt = Eigen::SerializableLDLT(cov.ldlt());
    // Precompute the information vector
    information = train_ldlt.solve(targets.mean);
  }

  // Create a fit that will replicate the given prediction states and
  // covariances (targets) for the given features on the basis that k_uu is the
  // prior covariance matrix relating them. This is useful because it allows us
  // to create a model based on all the data we wish to include (and therefore
  // be most accurate), then predict a few selected points based on this full
  // model (getting the most accuracy possible) but after that only require a
  // small amount of CPU load and memory to compute predictions close to those
  // initial selected points while still, presumably, getting excellent
  // accuracy. At least, predictions arbitrarily close to the initially
  // requested predictions in the feature domain will also be arbitrarily close
  // in the prediction distribution domain.
  Fit(const std::vector<FeatureType> &features,
      const JointDistribution &targets, const Eigen::MatrixXd &k_uu) {
    // This construction is based on the formulae that ensure the new sub-model
    // will obtain the same results as the full model for the initially
    // predicted data points:
    // information = K_uu^{-1}*targets.mean
    // train_ldlt = LDLt(K_uu*(K_uu-cov)^{-1}*K_uu)
    train_features = features;
    Eigen::MatrixXd k_uu_minus_cov(k_uu - targets.covariance);
    Eigen::SerializableLDLT k_uu_minus_cov_ldlt(k_uu_minus_cov.ldlt());
    Eigen::MatrixXd cov(k_uu * k_uu_minus_cov_ldlt.solve(k_uu));
    assert(!cov.hasNaN());
    train_ldlt = Eigen::SerializableLDLT(cov.ldlt());
    // Precompute the information vector
    Eigen::SerializableLDLT k_uu_ldlt(k_uu.ldlt());
    information = k_uu_ldlt.solve(targets.mean);
  }

  template <typename Archive> void serialize(Archive &archive) {
    archive(cereal::make_nvp("information", information));
    archive(cereal::make_nvp("train_ldlt", train_ldlt));
    archive(cereal::make_nvp("train_features", train_features));
  }

  template <typename OtherImplType>
  bool operator==(const Fit<GaussianProcessBase<CovFunc, OtherImplType>,
                            FeatureType> &other) const {
    return (train_features == other.train_features &&
            train_ldlt == other.train_ldlt && information == other.information);
  }
};

/*
 * Gaussian Process Helper Functions.
 */
inline Eigen::VectorXd gp_mean_prediction(const Eigen::MatrixXd &cross_cov,
                                          const Eigen::VectorXd &information) {
  return cross_cov.transpose() * information;
}

inline MarginalDistribution
gp_marginal_prediction(const Eigen::MatrixXd &cross_cov,
                       const Eigen::VectorXd &prior_variance,
                       const Eigen::VectorXd &information,
                       const Eigen::SerializableLDLT &train_ldlt) {
  const Eigen::VectorXd pred = gp_mean_prediction(cross_cov, information);
  // Here we efficiently only compute the diagonal of the posterior
  // covariance matrix.
  Eigen::MatrixXd explained = train_ldlt.solve(cross_cov);
  Eigen::VectorXd explained_variance =
      explained.cwiseProduct(cross_cov).array().colwise().sum();
  Eigen::VectorXd marginal_variance = prior_variance - explained_variance;
  return MarginalDistribution(pred, marginal_variance.asDiagonal());
}

inline JointDistribution
gp_joint_prediction(const Eigen::MatrixXd &cross_cov,
                    const Eigen::MatrixXd &prior_cov,
                    const Eigen::VectorXd &information,
                    const Eigen::SerializableLDLT &train_ldlt) {
  const Eigen::VectorXd pred = gp_mean_prediction(cross_cov, information);
  Eigen::MatrixXd explained_cov =
      cross_cov.transpose() * train_ldlt.solve(cross_cov);
  return JointDistribution(pred, prior_cov - explained_cov);
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
  template <typename FitFeatureType>
  using GPFitType = Fit<GaussianProcessBase<CovFunc, ImplType>, FitFeatureType>;

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
                           const JointDistribution &joint) {
    using FitType = typename fit_type<ImplType, FeatureType>::type;
    using FitModelType = typename fit_model_type<ImplType, FeatureType>::type;
    return FitModelType(
        impl(), FitType(features, joint, covariance_function_(features)));
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
    ss << "covariance_name: " << covariance_function_.pretty_string();
    return ss.str();
  }

  // If the implementing class doesn't have a fit method for this
  // FeatureType but the CovarianceFunction does.
  template <typename FeatureType,
            typename std::enable_if<
                has_call_operator<CovFunc, FeatureType, FeatureType>::value,
                int>::type = 0>
  GPFitType<FeatureType> _fit_impl(const std::vector<FeatureType> &features,
                                   const MarginalDistribution &targets) const {
    Eigen::MatrixXd cov = covariance_function_(features);
    return GPFitType<FeatureType>(features, cov, targets);
  }

  template <
      typename FeatureType, typename FitFeaturetype,
      typename std::enable_if<
          has_call_operator<CovFunc, FeatureType, FeatureType>::value &&
              has_call_operator<CovFunc, FeatureType, FitFeaturetype>::value,
          int>::type = 0>
  JointDistribution
  _predict_impl(const std::vector<FeatureType> &features,
                const GPFitType<FitFeaturetype> &gp_fit,
                PredictTypeIdentity<JointDistribution> &&) const {
    const auto cross_cov =
        covariance_function_(gp_fit.train_features, features);
    Eigen::MatrixXd prior_cov = covariance_function_(features);
    return gp_joint_prediction(cross_cov, prior_cov, gp_fit.information,
                               gp_fit.train_ldlt);
  }

  template <
      typename FeatureType, typename FitFeaturetype,
      typename std::enable_if<
          has_call_operator<CovFunc, FeatureType, FeatureType>::value &&
              has_call_operator<CovFunc, FeatureType, FitFeaturetype>::value,
          int>::type = 0>
  MarginalDistribution
  _predict_impl(const std::vector<FeatureType> &features,
                const GPFitType<FitFeaturetype> &gp_fit,
                PredictTypeIdentity<MarginalDistribution> &&) const {
    const auto cross_cov =
        covariance_function_(gp_fit.train_features, features);
    Eigen::VectorXd prior_variance(static_cast<Eigen::Index>(features.size()));
    for (Eigen::Index i = 0; i < prior_variance.size(); ++i) {
      prior_variance[i] = covariance_function_(features[i], features[i]);
    }
    return gp_marginal_prediction(cross_cov, prior_variance, gp_fit.information,
                                  gp_fit.train_ldlt);
  }

  template <
      typename FeatureType, typename FitFeaturetype,
      typename std::enable_if<
          has_call_operator<CovFunc, FeatureType, FeatureType>::value &&
              has_call_operator<CovFunc, FeatureType, FitFeaturetype>::value,
          int>::type = 0>
  Eigen::VectorXd _predict_impl(const std::vector<FeatureType> &features,
                                const GPFitType<FitFeaturetype> &gp_fit,
                                PredictTypeIdentity<Eigen::VectorXd> &&) const {
    const auto cross_cov =
        covariance_function_(gp_fit.train_features, features);
    return gp_mean_prediction(cross_cov, gp_fit.information);
  }

  template <
      typename FeatureType, typename FitFeatureType, typename PredictType,
      typename std::enable_if<
          !has_call_operator<CovFunc, FeatureType, FeatureType>::value ||
              !has_call_operator<CovFunc, FeatureType, FitFeatureType>::value,
          int>::type = 0>
  PredictType _predict_impl(const std::vector<FeatureType> &features,
                            const GPFitType<FitFeatureType> &gp_fit,
                            PredictTypeIdentity<PredictType> &&) const =
      delete; // Covariance Function isn't defined for FeatureType.

  template <typename FeatureType>
  std::map<std::string, JointDistribution>
  cross_validated_predictions(const RegressionDataset<FeatureType> &dataset,
                              const FoldIndexer &fold_indexer,
                              PredictTypeIdentity<JointDistribution>) const {

    const auto fit_model = impl().fit(dataset);
    const auto gp_fit = fit_model.get_fit();

    const std::vector<FoldIndices> indices = map_values(fold_indexer);
    const std::vector<std::string> fold_names = map_keys(fold_indexer);
    const auto inverse_blocks = gp_fit.train_ldlt.inverse_blocks(indices);

    std::map<std::string, JointDistribution> output;
    for (std::size_t i = 0; i < inverse_blocks.size(); i++) {
      Eigen::VectorXd yi = subset(dataset.targets.mean, indices[i]);
      Eigen::VectorXd vi = subset(gp_fit.information, indices[i]);
      const auto A_inv = inverse_blocks[i].inverse();
      output[fold_names[i]] = JointDistribution(yi - A_inv * vi, A_inv);
    }
    return output;
  }

  template <typename FeatureType>
  std::map<std::string, MarginalDistribution>
  cross_validated_predictions(const RegressionDataset<FeatureType> &dataset,
                              const FoldIndexer &fold_indexer,
                              PredictTypeIdentity<MarginalDistribution>) const {

    const auto fit_model = impl().fit(dataset);
    const auto gp_fit = fit_model.get_fit();

    const std::vector<FoldIndices> indices = map_values(fold_indexer);
    const std::vector<std::string> fold_names = map_keys(fold_indexer);
    const auto inverse_blocks = gp_fit.train_ldlt.inverse_blocks(indices);

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

  template <typename FeatureType>
  std::map<std::string, Eigen::VectorXd>
  cross_validated_predictions(const RegressionDataset<FeatureType> &dataset,
                              const FoldIndexer &fold_indexer,
                              PredictTypeIdentity<Eigen::VectorXd>) const {
    const auto fit_model = impl().fit(dataset);
    const auto gp_fit = fit_model.get_fit();
    const std::vector<FoldIndices> indices = map_values(fold_indexer);
    const std::vector<std::string> fold_names = map_keys(fold_indexer);
    const auto inverse_blocks = gp_fit.train_ldlt.inverse_blocks(indices);

    std::map<std::string, Eigen::VectorXd> output;
    for (std::size_t i = 0; i < inverse_blocks.size(); i++) {
      Eigen::VectorXd yi = subset(dataset.targets.mean, indices[i]);
      Eigen::VectorXd vi = subset(gp_fit.information, indices[i]);
      const auto A_ldlt = Eigen::SerializableLDLT(inverse_blocks[i].ldlt());
      output[fold_names[i]] = yi - A_ldlt.solve(vi);
    }
    return output;
  }

  CovFunc get_covariance() const { return covariance_function_; }

protected:
  /*
   * CRTP Helpers
   */
  ImplType &impl() { return *static_cast<ImplType *>(this); }
  const ImplType &impl() const { return *static_cast<const ImplType *>(this); }

  CovFunc covariance_function_;
  std::string model_name_;
};

/*
 * Generic Gaussian Process Implementation.
 */
template <typename CovFunc>
class GaussianProcessRegression
    : public GaussianProcessBase<CovFunc, GaussianProcessRegression<CovFunc>> {
public:
  using Base = GaussianProcessBase<CovFunc, GaussianProcessRegression<CovFunc>>;
  using Base::Base;
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
        negative_log_likelihood(dataset.targets.mean, gp_fit.train_ldlt);
    nll -= model.prior_log_likelihood();
    return nll;
  }
};

} // namespace albatross

#endif
