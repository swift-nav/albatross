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

#include "evaluate.h"
#include "stdio.h"
#include <functional>
#include <memory>

#include "cereal/eigen.h"
#include "core/model.h"
#include "core/serialize.h"
#include "eigen/serializable_ldlt.h"

namespace albatross {

using InspectionDistribution = JointDistribution;

template <typename FeatureType> struct GaussianProcessFit {
  std::vector<FeatureType> train_features;
  Eigen::SerializableLDLT train_ldlt;
  Eigen::VectorXd information;

  GaussianProcessFit(){};

  GaussianProcessFit(const std::vector<FeatureType> &features,
                     const Eigen::MatrixXd &train_cov,
                     const MarginalDistribution &targets);

  template <typename Archive>
  // todo: enable if FeatureType is serializable
  void serialize(Archive &archive) {
    archive(cereal::make_nvp("information", information));
    archive(cereal::make_nvp("train_ldlt", train_ldlt));
    archive(cereal::make_nvp("train_features", train_features));
  }

  bool operator==(const GaussianProcessFit &other) const {
    return (train_features == other.train_features &&
            train_ldlt == other.train_ldlt && information == other.information);
  }
};

template <typename FeatureType>
GaussianProcessFit<FeatureType>::GaussianProcessFit(
    const std::vector<FeatureType> &features, const Eigen::MatrixXd &train_cov,
    const MarginalDistribution &targets) {
  train_features = features;
  Eigen::MatrixXd cov(train_cov);
  if (targets.has_covariance()) {
    cov += targets.covariance;
  }
  train_ldlt = Eigen::SerializableLDLT(cov.ldlt());
  // Precompute the information vector
  information = train_ldlt.solve(targets.mean);
}

template <typename FeatureType>
inline JointDistribution predict_from_covariance_and_fit(
    const Eigen::MatrixXd &cross_cov, const Eigen::MatrixXd &pred_cov,
    const GaussianProcessFit<FeatureType> &model_fit) {
  const Eigen::VectorXd pred = cross_cov.transpose() * model_fit.information;
  auto ldlt = model_fit.train_ldlt;
  Eigen::MatrixXd posterior_cov = ldlt.solve(cross_cov);
  posterior_cov = cross_cov.transpose() * posterior_cov;
  posterior_cov = pred_cov - posterior_cov;
  return JointDistribution(pred, posterior_cov);
}

template <typename FeatureType, typename SubFeatureType = FeatureType>
using SerializableGaussianProcess =
    SerializableRegressionModel<FeatureType,
                                GaussianProcessFit<SubFeatureType>>;

template <typename FeatureType, typename CovarianceFunction>
class GaussianProcessRegression
    : public SerializableGaussianProcess<FeatureType> {
public:
  typedef GaussianProcessFit<FeatureType> FitType;
  typedef CovarianceFunction CovarianceType;

  GaussianProcessRegression()
      : covariance_function_(), model_name_(covariance_function_.get_name()){};
  GaussianProcessRegression(CovarianceFunction &covariance_function)
      : covariance_function_(covariance_function),
        model_name_(covariance_function_.get_name()){};
  /*
   * Sometimes it's nice to be able to provide a custom model name since
   * these models are generalizable.
   */
  GaussianProcessRegression(CovarianceFunction &covariance_function,
                            const std::string &model_name)
      : covariance_function_(covariance_function), model_name_(model_name){};
  GaussianProcessRegression(const std::string &model_name)
      : covariance_function_(), model_name_(model_name){};

  ~GaussianProcessRegression(){};

  std::string get_name() const override { return model_name_; };

  template <typename Archive> void save(Archive &archive) const {
    archive(cereal::base_class<SerializableRegressionModel<
                FeatureType, GaussianProcessFit<FeatureType>>>(this));
    archive(model_name_);
  }

  template <typename Archive> void load(Archive &archive) {
    archive(cereal::base_class<SerializableRegressionModel<
                FeatureType, GaussianProcessFit<FeatureType>>>(this));
    archive(model_name_);
  }

  template <typename OtherFeatureType>
  InspectionDistribution
  inspect(const std::vector<OtherFeatureType> &features) const {
    assert(this->has_been_fit());
    const auto cross_cov = asymmetric_covariance(
        covariance_function_, features, this->model_fit_.train_features);
    // Then we can use the information vector to determine the posterior
    const Eigen::VectorXd pred = cross_cov * this->model_fit_.information;
    Eigen::MatrixXd pred_cov =
        symmetric_covariance(covariance_function_, features);
    auto ldlt = this->model_fit_.train_ldlt;
    pred_cov -= cross_cov * ldlt.solve(cross_cov.transpose());
    assert(static_cast<std::size_t>(pred.size()) == features.size());
    return InspectionDistribution(pred, pred_cov);
  }

  /*
   * The Gaussian Process Regression model derives its parameters from
   * the covariance functions.
   */
  ParameterStore get_params() const override {
    return covariance_function_.get_params();
  }

  void unchecked_set_param(const std::string &name,
                           const Parameter &param) override {
    covariance_function_.set_param(name, param);
  }

  std::string pretty_string() const {
    std::ostringstream ss;
    ss << get_name() << std::endl;
    ss << covariance_function_.pretty_string();
    ss << "has_been_fit: " << this->has_been_fit() << std::endl;
    return ss.str();
  }

protected:
  /*
   * Cross validation specializations
   *
   * The leave one out cross validated predictions for a Gaussian Process
   * can be efficiently computed by dropping rows and columns from the
   * covariance and obtaining the prediction for the dropped index.  This
   * results in something like,
   *
   *     mean[group] = y[group] - A^{-1} (C^{-1} y)[group]
   *     variance[group] = A^{-1}
   *
   * with group the set of indices for the held out group and
   *     A = C^{-1}[group, group]
   * is the block of the inverse of the covariance that corresponds
   * to the group in question.
   *
   * See section 5.4.2 Rasmussen Gaussian Processes
   */
  virtual std::vector<JointDistribution> cross_validated_predictions_(
      const RegressionDataset<FeatureType> &dataset,
      const FoldIndexer &fold_indexer,
      const detail::PredictTypeIdentity<JointDistribution> &identity) override {

    this->fit(dataset);
    const FitType model_fit = this->get_fit();
    const std::vector<FoldIndices> indices = map_values(fold_indexer);
    const auto inverse_blocks = model_fit.train_ldlt.inverse_blocks(indices);

    std::vector<JointDistribution> output;
    for (std::size_t i = 0; i < inverse_blocks.size(); i++) {
      Eigen::VectorXd yi = subset(indices[i], dataset.targets.mean);
      Eigen::VectorXd vi = subset(indices[i], model_fit.information);
      const auto A_inv = inverse_blocks[i].inverse();
      output.push_back(JointDistribution(yi - A_inv * vi, A_inv));
    }
    return output;
  }

  virtual std::vector<MarginalDistribution> cross_validated_predictions_(
      const RegressionDataset<FeatureType> &dataset,
      const FoldIndexer &fold_indexer,
      const detail::PredictTypeIdentity<MarginalDistribution> &identity)
      override {
    this->fit(dataset);
    const FitType model_fit = this->get_fit();

    const std::vector<FoldIndices> indices = map_values(fold_indexer);
    const auto inverse_blocks = model_fit.train_ldlt.inverse_blocks(indices);

    std::vector<MarginalDistribution> output;
    for (std::size_t i = 0; i < inverse_blocks.size(); i++) {
      Eigen::VectorXd yi = subset(indices[i], dataset.targets.mean);
      Eigen::VectorXd vi = subset(indices[i], model_fit.information);
      const auto A_ldlt = Eigen::SerializableLDLT(inverse_blocks[i].ldlt());

      output.push_back(MarginalDistribution(
          yi - A_ldlt.solve(vi), A_ldlt.inverse_diagonal().asDiagonal()));
    }
    return output;
  }

  virtual std::vector<Eigen::VectorXd> cross_validated_predictions_(
      const RegressionDataset<FeatureType> &dataset,
      const FoldIndexer &fold_indexer,
      const detail::PredictTypeIdentity<PredictMeanOnly> &identity) override {
    this->fit(dataset);
    const FitType model_fit = this->get_fit();

    const std::vector<FoldIndices> indices = map_values(fold_indexer);
    const auto inverse_blocks = model_fit.train_ldlt.inverse_blocks(indices);

    std::vector<Eigen::VectorXd> output;
    for (std::size_t i = 0; i < inverse_blocks.size(); i++) {
      Eigen::VectorXd yi = subset(indices[i], dataset.targets.mean);
      Eigen::VectorXd vi = subset(indices[i], model_fit.information);
      const auto A_ldlt = Eigen::SerializableLDLT(inverse_blocks[i].ldlt());
      output.push_back(yi - A_ldlt.solve(vi));
    }
    return output;
  }

  FitType
  serializable_fit_(const std::vector<FeatureType> &features,
                    const MarginalDistribution &targets) const override {
    Eigen::MatrixXd cov = symmetric_covariance(covariance_function_, features);
    return GaussianProcessFit<FeatureType>(features, cov, targets);
  }

  JointDistribution
  predict_(const std::vector<FeatureType> &features) const override {
    const auto cross_cov = asymmetric_covariance(
        covariance_function_, this->model_fit_.train_features, features);
    Eigen::MatrixXd pred_cov =
        symmetric_covariance(covariance_function_, features);
    return predict_from_covariance_and_fit(cross_cov, pred_cov,
                                           this->model_fit_);
  }

  virtual MarginalDistribution
  predict_marginal_(const std::vector<FeatureType> &features) const override {
    const auto cross_cov = asymmetric_covariance(
        covariance_function_, features, this->model_fit_.train_features);
    const Eigen::VectorXd pred = cross_cov * this->model_fit_.information;
    // Here we efficiently only compute the diagonal of the posterior
    // covariance matrix.
    auto ldlt = this->model_fit_.train_ldlt;
    Eigen::MatrixXd explained = ldlt.solve(cross_cov.transpose());
    Eigen::VectorXd marginal_variance =
        -explained.cwiseProduct(cross_cov.transpose()).array().colwise().sum();
    for (Eigen::Index i = 0; i < pred.size(); i++) {
      marginal_variance[i] += covariance_function_(features[i], features[i]);
    }

    return MarginalDistribution(pred, marginal_variance.asDiagonal());
  }

  virtual Eigen::VectorXd
  predict_mean_(const std::vector<FeatureType> &features) const override {
    const auto cross_cov = asymmetric_covariance(
        covariance_function_, features, this->model_fit_.train_features);
    const Eigen::VectorXd pred = cross_cov * this->model_fit_.information;
    return pred;
  }

  CovarianceFunction covariance_function_;
  std::string model_name_;
};

template <typename FeatureType, typename CovFunc>
GaussianProcessRegression<FeatureType, CovFunc>
gp_from_covariance(CovFunc covariance_function) {
  return GaussianProcessRegression<FeatureType, CovFunc>(covariance_function);
};

template <typename FeatureType, typename CovFunc>
GaussianProcessRegression<FeatureType, CovFunc>
gp_from_covariance(CovFunc covariance_function, const std::string &model_name) {
  return GaussianProcessRegression<FeatureType, CovFunc>(covariance_function,
                                                         model_name);
};

template <typename FeatureType, typename CovFunc>
std::unique_ptr<GaussianProcessRegression<FeatureType, CovFunc>>
gp_pointer_from_covariance(CovFunc covariance_function) {
  return std::make_unique<GaussianProcessRegression<FeatureType, CovFunc>>(
      covariance_function);
}

template <typename FeatureType, typename CovFunc>
std::unique_ptr<GaussianProcessRegression<FeatureType, CovFunc>>
gp_pointer_from_covariance(CovFunc covariance_function,
                           const std::string &model_name) {
  return std::make_unique<GaussianProcessRegression<FeatureType, CovFunc>>(
      covariance_function, model_name);
}
} // namespace albatross

#endif
