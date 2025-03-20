/*
 * Copyright (C) 2025 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

namespace albatross {

template <typename FeatureType, typename Preconditioner>
struct ConjugateGradientGPFit {};

template <typename FeatureType, typename Preconditioner>
struct Fit<ConjugateGradientGPFit<FeatureType, Preconditioner>> {
  std::vector<FeatureType> train_features;
  // We have to store this; the solver only takes a reference to it
  // and doesn't own it.
  Eigen::MatrixXd K_ff;
  using SolverType =
      Eigen::ConjugateGradient<Eigen::MatrixXd, Eigen::Lower, Preconditioner>;
  // Pretty annoying that you can't copy an Eigen solver object.
  // Maybe this is a bad idea for now, but seemingly Albatross
  // requires you to be able to copy these.
  std::shared_ptr<SolverType> solver;
  Eigen::VectorXd information;

  Fit(const std::vector<FeatureType> &train_features_, Eigen::MatrixXd &&K_ff_,
      const MarginalDistribution &targets_)
      : train_features{train_features_}, K_ff{std::move(K_ff_)},
        // N.B. we give the CG solver a reference to our local member
        // matrix.
        solver{std::make_shared<SolverType>(K_ff)},
        information{solver->solve(targets_.mean)} {}

  bool operator==(const Fit &other) {
    return std::tie(train_features, K_ff, information) ==
           std::tie(other.train_features, other.K_ff, other.information);
  }
};

template <typename CGSolver>
inline JointDistribution cg_gp_joint_prediction(
    const Eigen::MatrixXd &cross_cov, const Eigen::MatrixXd &prior_cov,
    const Eigen::VectorXd &information, const CGSolver &solver) {
  return JointDistribution(gp_mean_prediction(cross_cov, information),
                           prior_cov -
                               cross_cov.transpose() * solver.solve(cross_cov));
}

template <typename CGSolver>
inline MarginalDistribution cg_gp_marginal_prediction(
    const Eigen::MatrixXd &cross_cov, const Eigen::VectorXd &prior_variance,
    const Eigen::VectorXd &information, const CGSolver &solver) {
  return MarginalDistribution(
      gp_mean_prediction(cross_cov, information),
      prior_variance -
          (cross_cov.transpose() * solver.solve(cross_cov)).diagonal());
}

template <typename CovFunc, typename MeanFunc, typename Preconditioner>
class ConjugateGradientGaussianProcessRegression
    : public GaussianProcessBase<CovFunc, MeanFunc,
                                 ConjugateGradientGaussianProcessRegression<
                                     CovFunc, MeanFunc, Preconditioner>> {
public:
  using Base = GaussianProcessBase<CovFunc, MeanFunc,
                                   ConjugateGradientGaussianProcessRegression<
                                       CovFunc, MeanFunc, Preconditioner>>;

  using Base::covariance_function_;
  using Base::mean_function_;

  ConjugateGradientGaussianProcessRegression() : Base() {};

  template <typename Cov>
  ConjugateGradientGaussianProcessRegression(Cov &&covariance_function)
      : Base(std::forward<CovFunc>(covariance_function)) {
    static_assert(
        std::is_same<std::decay_t<Cov>, CovFunc>::value,
        "Please construct this with the same covariance it's declared for.");
  }

  template <typename Cov>
  ConjugateGradientGaussianProcessRegression(Cov &&covariance_function,
                                             const std::string &model_name)
      : Base(std::forward<CovFunc>(covariance_function), model_name) {
    static_assert(
        std::is_same<std::decay_t<Cov>, CovFunc>::value,
        "Please construct this with the same covariance it's declared for.");
  }

  template <typename Cov, typename Mean>
  ConjugateGradientGaussianProcessRegression(Cov &&covariance_function,
                                             Mean &&mean_function,
                                             const std::string &model_name)
      : Base(std::forward<CovFunc>(covariance_function),
             std::forward<MeanFunc>(mean_function), model_name) {
    static_assert(std::is_same<std::decay_t<Cov>, CovFunc>::value &&
                      std::is_same<std::decay_t<Mean>, MeanFunc>::value,
                  "Please construct this with the same covariance and mean "
                  "it's declared for.");
  }

  template <
      typename FeatureType,
      std::enable_if_t<
          has_call_operator<CovFunc, FeatureType, FeatureType>::value, int> = 0>
  Fit<ConjugateGradientGPFit<FeatureType, Preconditioner>>
  _fit_impl(const std::vector<FeatureType> &features,
            const MarginalDistribution &targets) const {
    Eigen::MatrixXd K_ff =
        covariance_function_(features, Base::threads_.get());
    MarginalDistribution zero_mean_targets(targets);
    K_ff += targets.covariance;
    mean_function_.remove_from(features, &zero_mean_targets.mean);
    return Fit<ConjugateGradientGPFit<FeatureType, Preconditioner>>{
        features, std::move(K_ff), std::move(zero_mean_targets)};
  }

  template <
      typename FeatureType, typename FitFeatureType,
      std::enable_if_t<
          has_call_operator<CovFunc, FeatureType, FeatureType>::value &&
              has_call_operator<CovFunc, FeatureType, FitFeatureType>::value,
          int> = 0>
  JointDistribution _predict_impl(
      const std::vector<FeatureType> &features,
      const Fit<ConjugateGradientGPFit<FitFeatureType, Preconditioner>> &cg_gp_fit,
      PredictTypeIdentity<JointDistribution> &&) const {
    const auto cross_cov =
        covariance_function_(cg_gp_fit.train_features, features);
    Eigen::MatrixXd prior_cov{covariance_function_(features)};
    auto pred = cg_gp_joint_prediction(
        cross_cov, prior_cov, cg_gp_fit.information, *cg_gp_fit.solver);
    mean_function_.add_to(features, &pred.mean);
    return pred;
  }

  template <
      typename FeatureType, typename FitFeatureType,
      std::enable_if_t<
          has_call_operator<CovFunc, FeatureType, FeatureType>::value &&
              has_call_operator<CovFunc, FeatureType, FitFeatureType>::value,
          int> = 0>
  MarginalDistribution _predict_impl(
      const std::vector<FeatureType> &features,
      const Fit<ConjugateGradientGPFit<FitFeatureType, Preconditioner>>
          &cg_gp_fit,
      PredictTypeIdentity<MarginalDistribution> &&) const {
    const auto cross_cov =
        covariance_function_(cg_gp_fit.train_features, features);
    Eigen::VectorXd prior_variance(Eigen::VectorXd::NullaryExpr(
        cast::to_index(features.size()), [this, &features](Eigen::Index i) {
          const auto &f = features[cast::to_size(i)];
          return covariance_function_(f, f);
        }));
    auto pred = cg_gp_marginal_prediction(
        cross_cov, prior_variance, cg_gp_fit.information, *cg_gp_fit.solver);
    mean_function_.add_to(features, &pred.mean);
    return pred;
  }

  template <
      typename FeatureType, typename FitFeatureType,
      std::enable_if_t<
          has_call_operator<CovFunc, FeatureType, FeatureType>::value &&
              has_call_operator<CovFunc, FeatureType, FitFeatureType>::value,
          int> = 0>
  Eigen::VectorXd _predict_impl(
      const std::vector<FeatureType> &features,
      const Fit<ConjugateGradientGPFit<FitFeatureType, Preconditioner>>
          &cg_gp_fit,
      PredictTypeIdentity<Eigen::VectorXd> &&) const {
    const auto cross_cov =
        covariance_function_(cg_gp_fit.train_features, features);
    auto pred = gp_mean_prediction(cross_cov, cg_gp_fit.information);
    mean_function_.add_to(features, &pred);
    return pred;
  }
};

// TODO(@peddie): this is known not to have any effect for stationary
// kernels, but IncompleteCholesky assumes a `twistedBy` (sparse
// self-adjoint view) member, so this is all we can do for the moment.
using AlbatrossCGDefaultPreconditioner = Eigen::DiagonalPreconditioner<double>;

template <typename CovFunc,
          typename Preconditioner = AlbatrossCGDefaultPreconditioner>
auto cg_gp_from_covariance(CovFunc &&covariance_function,
                           const std::string &model_name,
                           Preconditioner && = Preconditioner{}) {
  return ConjugateGradientGaussianProcessRegression<
      typename std::decay<CovFunc>::type, decltype(ZeroMean()), Preconditioner>(
      std::forward<CovFunc>(covariance_function), model_name);
}

template <typename CovFunc, typename MeanFunc,
          typename Preconditioner = AlbatrossCGDefaultPreconditioner>
auto cg_gp_from_mean_and_covariance(CovFunc &&covariance_function,
                                    MeanFunc &&mean_function,
                                    const std::string &model_name,
                                    Preconditioner && = Preconditioner{}) {
  return ConjugateGradientGaussianProcessRegression<
      typename std::decay<CovFunc>::type, typename std::decay<MeanFunc>::type,
      Preconditioner>(std::forward<CovFunc>(covariance_function),
                      std::forward<MeanFunc>(mean_function), model_name);
}

} // namespace albatross