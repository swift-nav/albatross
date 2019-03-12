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
    train_ldlt = Eigen::SerializableLDLT(cov.ldlt());
    // Precompute the information vector
    information = train_ldlt.solve(targets.mean);
  }

  template <typename Archive>
  // todo: enable if FeatureType is serializable
  void serialize(Archive &archive) {
    archive(cereal::make_nvp("information", information));
    archive(cereal::make_nvp("train_ldlt", train_ldlt));
    archive(cereal::make_nvp("train_features", train_features));
  }

  bool operator==(const Fit &other) const {
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
class GaussianProcessBase
    : public ModelBase<GaussianProcessBase<CovFunc, ImplType>> {

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
                has_call_operator<CovFunc, FeatureType, FeatureType>::value &&
                    !has_valid_fit<ImplType, FeatureType>::value,
                int>::type = 0>
  GPFitType<FeatureType> fit(const std::vector<FeatureType> &features,
                             const MarginalDistribution &targets) const {
    Eigen::MatrixXd cov = covariance_function_(features);
    return GPFitType<FeatureType>(features, cov, targets);
  }

  template <typename FeatureType,
            typename std::enable_if<has_valid_fit<ImplType, FeatureType>::value,
                                    int>::type = 0>
  auto fit(const std::vector<FeatureType> &features,
           const MarginalDistribution &targets) const {
    return impl().fit(features, targets);
  }

  template <
      typename FeatureType, typename FitFeaturetype,
      typename std::enable_if<
          has_call_operator<CovFunc, FeatureType, FeatureType>::value &&
              has_call_operator<CovFunc, FeatureType, FitFeaturetype>::value &&
              !has_valid_predict<ImplType, FeatureType,
                                 GPFitType<FitFeaturetype>,
                                 JointDistribution>::value,
          int>::type = 0>
  JointDistribution predict(const std::vector<FeatureType> &features,
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
              has_call_operator<CovFunc, FeatureType, FitFeaturetype>::value &&
              !has_valid_predict<ImplType, FeatureType,
                                 GPFitType<FitFeaturetype>,
                                 MarginalDistribution>::value,
          int>::type = 0>
  MarginalDistribution
  predict(const std::vector<FeatureType> &features,
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
              has_call_operator<CovFunc, FeatureType, FitFeaturetype>::value &&
              !has_valid_predict<ImplType, FeatureType,
                                 GPFitType<FitFeaturetype>,
                                 Eigen::VectorXd>::value,
          int>::type = 0>
  Eigen::VectorXd predict(const std::vector<FeatureType> &features,
                          const GPFitType<FitFeaturetype> &gp_fit,
                          PredictTypeIdentity<Eigen::VectorXd> &&) const {
    const auto cross_cov =
        covariance_function_(gp_fit.train_features, features);
    return gp_mean_prediction(cross_cov, gp_fit.information);
  }

  template <typename FeatureType, typename FitFeatureType, typename PredictType,
            typename std::enable_if<has_valid_predict<ImplType, FeatureType,
                                                      GPFitType<FitFeatureType>,
                                                      PredictType>::value,
                                    int>::type = 0>
  PredictType predict(const std::vector<FeatureType> &features,
                      const GPFitType<FitFeatureType> &gp_fit,
                      PredictTypeIdentity<PredictType> &&) const {
    return impl().predict(features, gp_fit, PredictTypeIdentity<PredictType>());
  }

  template <
      typename FeatureType, typename FitFeatureType, typename PredictType,
      typename std::enable_if<
          (!has_call_operator<CovFunc, FeatureType, FeatureType>::value ||
           !has_call_operator<CovFunc, FeatureType, FitFeatureType>::value) &&
              !has_valid_predict<ImplType, FeatureType,
                                 GPFitType<FitFeatureType>, PredictType>::value,
          int>::type = 0>
  PredictType predict(const std::vector<FeatureType> &features,
                      const GPFitType<FitFeatureType> &gp_fit,
                      PredictTypeIdentity<PredictType> &&) const =
      delete; // Covariance Function isn't defined for FeatureType.

protected:
  /*
   * CRTP Helpers
   */
  ImplType &impl() { return *static_cast<ImplType *>(this); }
  const ImplType &impl() const { return *static_cast<const ImplType *>(this); }

  CovFunc covariance_function_;
  std::string model_name_;
};

template <typename CovFunc>
class GaussianProcessRegression
    : public GaussianProcessBase<CovFunc, GaussianProcessRegression<CovFunc>> {
public:
  using Base = GaussianProcessBase<CovFunc, GaussianProcessRegression<CovFunc>>;

  GaussianProcessRegression() : Base(){};
  GaussianProcessRegression(CovFunc &covariance_function)
      : Base(covariance_function){};
  GaussianProcessRegression(CovFunc &covariance_function,
                            const std::string &model_name)
      : Base(covariance_function, model_name){};

  // The only reason these are here is to hide the base class implementations.
  void fit() const = delete;
  void predict() const = delete;
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

} // namespace albatross

#endif
