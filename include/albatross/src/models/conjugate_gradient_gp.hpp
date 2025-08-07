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

struct IterativeSolverOptions {
  // Negative values for these will be ignored.
  static constexpr const Eigen::Index cDefaultMaxIterations{-1};
  Eigen::Index max_iterations{cDefaultMaxIterations};
  static constexpr const double cDefaultTolerance{-1};
  double tolerance{cDefaultTolerance};

  IterativeSolverOptions() {}

  IterativeSolverOptions(Eigen::Index iterations, double tol)
      : max_iterations{iterations}, tolerance{tol} {}

  template <typename SolverType>
  explicit IterativeSolverOptions(SolverType &&solver)
      : max_iterations{solver.maxIterations()}, tolerance{solver.tolerance()} {}

  template <typename SolverType>
  void configure_solver(SolverType &solver) const {
    if (tolerance > 0) {
      solver.setTolerance(tolerance);
    }

    if (max_iterations > 0) {
      solver.setMaxIterations(max_iterations);
    }
  }
};

struct IterativeSolverState {
  Eigen::ComputationInfo info;
  Eigen::Index iterations;
  double error;

  IterativeSolverState() = delete;

  template <typename SolverType>
  explicit IterativeSolverState(SolverType &&solver)
      : info{solver.info()},
        iterations{solver.iterations()}, error{solver.error()} {}
};

namespace detail {

template <typename SolverType>
std::shared_ptr<std::decay_t<SolverType>>
make_shared_solver_with_options(const Eigen::MatrixXd &m,
                                const IterativeSolverOptions &options) {
  auto solver = std::make_shared<std::decay_t<SolverType>>(m);
  options.configure_solver(*solver);
  return solver;
}
} // namespace detail

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

  IterativeSolverOptions get_options() const {
    return IterativeSolverOptions{*solver};
  }

  IterativeSolverState get_solver_state() const {
    return IterativeSolverState{*solver};
  }

  void set_options(const IterativeSolverOptions &options) {
    options.configure_solver(*solver);
  }

  Fit(const std::vector<FeatureType> &train_features_, Eigen::MatrixXd &&K_ff_,
      const MarginalDistribution &targets_)
      : train_features{train_features_}, K_ff{std::move(K_ff_)},
        // N.B. we give the CG solver a reference to our local member
        // matrix.
        solver{std::make_shared<SolverType>(K_ff)}, information{solver->solve(
                                                        targets_.mean)} {
    if (solver->info() != Eigen::Success) {
      information.setConstant(std::numeric_limits<double>::quiet_NaN());
    }
  }

  Fit(const std::vector<FeatureType> &train_features_, Eigen::MatrixXd &&K_ff_,
      const MarginalDistribution &targets_,
      const IterativeSolverOptions &options)
      : train_features{train_features_}, K_ff{std::move(K_ff_)},
        // N.B. we give the CG solver a reference to our local member
        // matrix.
        solver{
            detail::make_shared_solver_with_options<SolverType>(K_ff, options)},
        information{solver->solve(targets_.mean)} {
    if (solver->info() != Eigen::Success) {
      information.setConstant(std::numeric_limits<double>::quiet_NaN());
    }
  }

  bool operator==(const Fit &other) {
    return std::tie(train_features, K_ff, information) ==
           std::tie(other.train_features, other.K_ff, other.information);
  }
};

template <typename CGSolver>
inline JointDistribution cg_gp_joint_prediction(
    const Eigen::MatrixXd &cross_cov, const Eigen::MatrixXd &prior_cov,
    const Eigen::VectorXd &information, const CGSolver &solver) {
  JointDistribution ret(gp_mean_prediction(cross_cov, information),
                        prior_cov -
                            cross_cov.transpose() * solver.solve(cross_cov));
  if (solver.info() != Eigen::Success) {
    ret.covariance.setConstant(std::numeric_limits<double>::quiet_NaN());
  }

  return ret;
}

template <typename CGSolver>
inline MarginalDistribution cg_gp_marginal_prediction(
    const Eigen::MatrixXd &cross_cov, const Eigen::VectorXd &prior_variance,
    const Eigen::VectorXd &information, const CGSolver &solver) {
  MarginalDistribution ret(
      gp_mean_prediction(cross_cov, information),
      prior_variance -
          (cross_cov.transpose() * solver.solve(cross_cov)).diagonal());
  if (solver.info() != Eigen::Success) {
    ret.covariance.diagonal().setConstant(
        std::numeric_limits<double>::quiet_NaN());
  }

  return ret;
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

  ConjugateGradientGaussianProcessRegression() : Base(){};

  template <typename Cov,
            std::enable_if_t<std::is_same<std::decay_t<Cov>, CovFunc>::value,
                             int> = 0>
  ConjugateGradientGaussianProcessRegression(Cov &&covariance_function)
      : Base(std::forward<CovFunc>(covariance_function)) {}

  template <typename Cov,
            std::enable_if_t<std::is_same<std::decay_t<Cov>, CovFunc>::value,
                             int> = 0>
  ConjugateGradientGaussianProcessRegression(Cov &&covariance_function,
                                             const std::string &model_name)
      : Base(std::forward<CovFunc>(covariance_function), model_name) {}

  template <typename Cov,
            std::enable_if_t<std::is_same<std::decay_t<Cov>, CovFunc>::value,
                             int> = 0>
  ConjugateGradientGaussianProcessRegression(
      Cov &&covariance_function, const std::string &model_name,
      const IterativeSolverOptions &options)
      : Base(std::forward<CovFunc>(covariance_function), model_name),
        options_{options} {}

  template <
      typename Cov, typename Mean,
      std::enable_if_t<std::is_same<std::decay_t<Cov>, CovFunc>::value &&
                           std::is_same<std::decay_t<Mean>, MeanFunc>::value,
                       int> = 0>
  ConjugateGradientGaussianProcessRegression(Cov &&covariance_function,
                                             Mean &&mean_function,
                                             const std::string &model_name)
      : Base(std::forward<CovFunc>(covariance_function),
             std::forward<MeanFunc>(mean_function), model_name) {}

  template <
      typename Cov, typename Mean,
      std::enable_if_t<std::is_same<std::decay_t<Cov>, CovFunc>::value &&
                           std::is_same<std::decay_t<Mean>, MeanFunc>::value,
                       int> = 0>
  ConjugateGradientGaussianProcessRegression(
      Cov &&covariance_function, Mean &&mean_function,
      const std::string &model_name, const IterativeSolverOptions &options)
      : Base(std::forward<CovFunc>(covariance_function),
             std::forward<MeanFunc>(mean_function), model_name),
        options_{options} {}

  template <
      typename FeatureType,
      std::enable_if_t<
          has_call_operator<CovFunc, FeatureType, FeatureType>::value, int> = 0>
  Fit<ConjugateGradientGPFit<FeatureType, Preconditioner>>
  _fit_impl(const std::vector<FeatureType> &features,
            const MarginalDistribution &targets) const {
    Eigen::MatrixXd K_ff = covariance_function_(features, Base::threads_.get());
    MarginalDistribution zero_mean_targets(targets);
    K_ff += targets.covariance;
    mean_function_.remove_from(features, &zero_mean_targets.mean);
    // Clamp this so that we never try to iterate more than the
    // theoretical max of CG
    auto options = options_;
    options.max_iterations = std::min(options_.max_iterations, K_ff.rows());
    return Fit<ConjugateGradientGPFit<FeatureType, Preconditioner>>{
        features, std::move(K_ff), std::move(zero_mean_targets), options};
  }

  template <
      typename FeatureType, typename FitFeatureType,
      std::enable_if_t<
          has_call_operator<CovFunc, FeatureType, FeatureType>::value &&
              has_call_operator<CovFunc, FeatureType, FitFeatureType>::value,
          int> = 0>
  JointDistribution _predict_impl(
      const std::vector<FeatureType> &features,
      const Fit<ConjugateGradientGPFit<FitFeatureType, Preconditioner>>
          &cg_gp_fit,
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

  IterativeSolverOptions get_options() const { return options_; }

  void set_options(const IterativeSolverOptions &options) {
    options_ = options;
  }

private:
  IterativeSolverOptions options_{};
};

// TODO(@peddie): this is known not to have any effect for stationary
// kernels, but IncompleteCholesky assumes a `twistedBy` (sparse
// self-adjoint view) member, so this is all we can do for the moment.
using AlbatrossCGDefaultPreconditioner = Eigen::DiagonalPreconditioner<double>;

template <typename CovFunc,
          typename Preconditioner = AlbatrossCGDefaultPreconditioner>
auto cg_gp_from_covariance(
    CovFunc &&covariance_function, const std::string &model_name,
    const IterativeSolverOptions &options = IterativeSolverOptions{},
    Preconditioner && = Preconditioner{}) {
  return ConjugateGradientGaussianProcessRegression<
      typename std::decay<CovFunc>::type, decltype(ZeroMean()), Preconditioner>(
      std::forward<CovFunc>(covariance_function), model_name, options);
}

template <typename CovFunc, typename MeanFunc,
          typename Preconditioner = AlbatrossCGDefaultPreconditioner>
auto cg_gp_from_mean_and_covariance(
    CovFunc &&covariance_function, MeanFunc &&mean_function,
    const std::string &model_name,
    const IterativeSolverOptions &options = IterativeSolverOptions{},
    Preconditioner && = Preconditioner{}) {
  return ConjugateGradientGaussianProcessRegression<
      typename std::decay<CovFunc>::type, typename std::decay<MeanFunc>::type,
      Preconditioner>(std::forward<CovFunc>(covariance_function),
                      std::forward<MeanFunc>(mean_function), model_name,
                      options);
}

} // namespace albatross