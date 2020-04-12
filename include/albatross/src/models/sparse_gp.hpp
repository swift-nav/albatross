/*
 * Copyright (C) 2019 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef INCLUDE_ALBATROSS_MODELS_SPARSE_GP_H_
#define INCLUDE_ALBATROSS_MODELS_SPARSE_GP_H_

namespace albatross {

namespace details {

constexpr double DEFAULT_NUGGET = 1e-12;

inline std::string measurement_nugget_name() { return "measurement_nugget"; }

inline std::string inducing_nugget_name() { return "inducing_nugget"; }

} // namespace details

template <typename CovFunc, typename MeanFunc, typename GrouperFunction,
          typename InducingPointStrategy>
class SparseGaussianProcessRegression;

struct UniformlySpacedInducingPoints {

  UniformlySpacedInducingPoints(std::size_t num_points_ = 10)
      : num_points(num_points_) {}

  template <typename CovarianceFunction>
  std::vector<double> operator()(const CovarianceFunction &cov,
                                 const std::vector<double> &features) const {
    double min = *std::min_element(features.begin(), features.end());
    double max = *std::max_element(features.begin(), features.end());
    return linspace(min, max, num_points);
  }

  std::size_t num_points;
};

struct StateSpaceInducingPointStrategy {

  template <typename CovarianceFunction, typename FeatureType,
            std::enable_if_t<has_valid_state_space_representation<
                                 CovarianceFunction, FeatureType>::value,
                             int> = 0>
  auto operator()(const CovarianceFunction &cov,
                  const std::vector<FeatureType> &features) const {
    return cov.state_space_representation(features);
  }

  template <typename CovarianceFunction, typename FeatureType,
            std::enable_if_t<!has_valid_state_space_representation<
                                 CovarianceFunction, FeatureType>::value,
                             int> = 0>
  auto operator()(const CovarianceFunction &cov,
                  const std::vector<FeatureType> &features) const
      ALBATROSS_FAIL(
          CovarianceFunction,
          "Covariance function is missing state_space_representation method, "
          "be sure _ssr_impl has been defined for the types concerned");
};

/*
 *  This class implements an approximation technique for Gaussian processes
 * which relies on an assumption that all observations are independent (or
 * groups of observations are independent) conditional on a set of inducing
 * points.  The method is based off:
 *
 *     [1] Sparse Gaussian Processes using Pseudo-inputs
 *     Edward Snelson, Zoubin Ghahramani
 *     http://www.gatsby.ucl.ac.uk/~snelson/SPGP_up.pdf
 *
 *  Though the code uses notation closer to that used in this (excellent)
 * overview of these methods:
 *
 *     [2] A Unifying View of Sparse Approximate Gaussian Process Regression
 *     Joaquin Quinonero-Candela, Carl Edward Rasmussen
 *     http://www.jmlr.org/papers/volume6/quinonero-candela05a/quinonero-candela05a.pdf
 *
 *  Very broadly speaking this method starts with a prior over the observations,
 *
 *     [f] ~ N(0, K_ff)
 *
 *  where K_ff(i, j) = covariance_function(features[i], features[j]) and f
 * represents the function value.
 *
 *  It then uses a set of inducing points, u, and makes some assumptions about
 * the conditional distribution:
 *
 *     [f|u] ~ N(K_fu K_uu^-1 u, K_ff - Q_ff)
 *
 *  Where Q_ff = K_fu K_uu^-1 K_uf represents the variance in f that is
 * explained by u.
 *
 *  For FITC (Fully Independent Training Contitional) the assumption is that
 * K_ff - Qff is diagonal, for PITC (Partially Independent Training Conditional)
 * that it is block diagonal.  These assumptions lead to an efficient way of
 * inferring the posterior distribution for some new location f*,
 *
 *     [f*|f=y] ~ N(K_*u S K_uf A^-1 y, K_** - Q_** + K_*u S K_u*)
 *
 *  Where S = (K_uu + K_uf A^-1 K_fu)^-1 and A = diag(K_ff - Q_ff) and "diag"
 * may mean diagonal or block diagonal.  Regardless we end up with O(m^2n)
 * complexity instead of O(n^3) of direct Gaussian processes.  (Note that in [2]
 * S is called sigma and A is lambda.)
 *
 *  Of course, the implementation details end up somewhat more complex in order
 * to improve numerical stability.  A few great resources were heavily used to
 * get those deails straight:
 *
 *     - https://bwengals.github.io/pymc3-fitcvfe-implementation-notes.html
 *        (beware, the final formulae are correct but there are typos in
 *         their derivations).
 *     - https://github.com/SheffieldML/GPy see fitc.py
 */
template <typename CovFunc, typename MeanFunc, typename GrouperFunction,
          typename InducingPointStrategy>
class SparseGaussianProcessRegression
    : public GaussianProcessBase<
          CovFunc, MeanFunc,
          SparseGaussianProcessRegression<CovFunc, MeanFunc, GrouperFunction,
                                          InducingPointStrategy>> {

public:
  using Base = GaussianProcessBase<
      CovFunc, MeanFunc,
      SparseGaussianProcessRegression<CovFunc, MeanFunc, GrouperFunction,
                                      InducingPointStrategy>>;

  SparseGaussianProcessRegression() : Base() { initialize_params(); };

  SparseGaussianProcessRegression(const CovFunc &covariance_function,
                                  const MeanFunc &mean_function)
      : Base(covariance_function, mean_function) {
    initialize_params();
  };
  SparseGaussianProcessRegression(CovFunc &&covariance_function,
                                  MeanFunc &&mean_function)
      : Base(std::move(covariance_function), std::move(mean_function)) {
    initialize_params();
  };

  SparseGaussianProcessRegression(
      const CovFunc &covariance_function, const MeanFunc &mean_function,
      const GrouperFunction &independent_group_function,
      const InducingPointStrategy &inducing_point_strategy,
      const std::string &model_name)
      : Base(covariance_function, mean_function, model_name),
        inducing_point_strategy_(inducing_point_strategy),
        independent_group_function_(independent_group_function) {
    initialize_params();
  };
  SparseGaussianProcessRegression(
      CovFunc &&covariance_function, MeanFunc &&mean_function,
      GrouperFunction &&independent_group_function,
      InducingPointStrategy &&inducing_point_strategy,
      const std::string &model_name)
      : Base(std::move(covariance_function), std::move(mean_function),
             model_name),
        inducing_point_strategy_(std::move(inducing_point_strategy)),
        independent_group_function_(std::move(independent_group_function)) {
    initialize_params();
  };

  void initialize_params() {
    measurement_nugget_ = {
        details::DEFAULT_NUGGET,
        LogScaleUniformPrior(PARAMETER_EPSILON, PARAMETER_MAX)};
    inducing_nugget_ = {details::DEFAULT_NUGGET,
                        LogScaleUniformPrior(PARAMETER_EPSILON, PARAMETER_MAX)};
  }

  ParameterStore get_params() const override {
    auto params = map_join(this->mean_function_.get_params(),
                           this->covariance_function_.get_params());
    params[details::measurement_nugget_name()] = measurement_nugget_;
    params[details::inducing_nugget_name()] = inducing_nugget_;
    return params;
  }

  void unchecked_set_param(const std::string &name,
                           const Parameter &param) override {
    if (map_contains(this->covariance_function_.get_params(), name)) {
      this->covariance_function_.set_param(name, param);
    } else if (map_contains(this->mean_function_.get_params(), name)) {
      this->mean_function_.set_param(name, param);
    } else if (name == details::measurement_nugget_name()) {
      measurement_nugget_ = param;
    } else if (name == details::inducing_nugget_name()) {
      inducing_nugget_ = param;
    } else {
      std::cerr << "Unknown param: " << name << std::endl;
      assert(false);
    }
  }

  struct SparseGPComponents {
    Eigen::SerializableLDLT K_uu_ldlt;
    Eigen::MatrixXd P;
    BlockDiagonalLDLT A_ldlt;
    Eigen::MatrixXd RtR;
    Eigen::SerializableLDLT B_ldlt;
    Eigen::VectorXd y;
  };

  template <typename FeatureType, typename InducingFeatureType>
  SparseGPComponents
  sparse_components(const std::vector<FeatureType> &out_of_order_features,
                    const std::vector<InducingFeatureType> &inducing_features,
                    const MarginalDistribution &out_of_order_targets) const {

    const auto indexer =
        group_by(out_of_order_features, independent_group_function_).indexers();

    const auto out_of_order_measurement_features =
        as_measurements(out_of_order_features);

    std::vector<std::size_t> reordered_inds;
    BlockDiagonal K_ff;
    for (const auto &pair : indexer) {
      reordered_inds.insert(reordered_inds.end(), pair.second.begin(),
                            pair.second.end());
      auto subset_features =
          subset(out_of_order_measurement_features, pair.second);
      K_ff.blocks.emplace_back(this->covariance_function_(subset_features));
      K_ff.blocks.back().diagonal() +=
          subset(out_of_order_targets.covariance.diagonal(), pair.second);
    }

    const auto features =
        subset(out_of_order_measurement_features, reordered_inds);
    auto targets = subset(out_of_order_targets, reordered_inds);

    this->mean_function_.remove_from(
        subset(out_of_order_features, reordered_inds), &targets.mean);

    Eigen::Index m = static_cast<Eigen::Index>(inducing_features.size());

    const Eigen::MatrixXd K_fu =
        this->covariance_function_(features, inducing_features);
    Eigen::MatrixXd K_uu = this->covariance_function_(inducing_features);

    K_uu.diagonal() +=
        inducing_nugget_.value * Eigen::VectorXd::Ones(K_uu.rows());

    const Eigen::SerializableLDLT K_uu_ldlt = K_uu.ldlt();
    // P is such that:
    //     Q_ff = K_fu K_uu^-1 K_uf
    //          = K_fu L^-T L^-1 K_uf
    //          = P^T P
    const Eigen::MatrixXd P = K_uu_ldlt.sqrt_solve(K_fu.transpose());

    // We only need the diagonal blocks of Q_ff to get A
    BlockDiagonal Q_ff_diag;
    Eigen::Index i = 0;
    for (const auto &pair : indexer) {
      Eigen::Index cols = static_cast<Eigen::Index>(pair.second.size());
      auto P_cols = P.block(0, i, P.rows(), cols);
      Q_ff_diag.blocks.emplace_back(P_cols.transpose() * P_cols);
      i += cols;
    }
    auto A = K_ff - Q_ff_diag;

    // It's possible that the inducing points will perfectly describe
    // some of the data, in which case we need to add a bit of extra
    // noise to make sure lambda is invertible.
    for (auto &b : A.blocks) {
      b.diagonal() +=
          measurement_nugget_.value * Eigen::VectorXd::Ones(b.rows());
    }

    /*
     * The end goal here is to produce a vector, v, and matrix, C, such that
     * for a prediction, f*, we can do,
     *
     *     [f*|f=y] ~ N(K_*u * v , K_** - K_*u * C^-1 * K_u*)
     *
     *  and it would match the desired prediction described above,
     *
     *     [f*|f=y] ~ N(K_*u S K_uf^-1 A^-1 y, K_** − Q_** + K_*u S K_u*)
     *
     *  we can find v easily,
     *
     *     v = S K_uf A^-1 y
     *
     *  and to get C we need to do some algebra,
     *
     *     K_** - K_*u * C^-1 * K_u* = K_** - Q_** + K_*u S K_u*
     *                               = K_** - K_*u (K_uu^-1 - S) K_u*
     *
     *  which leads to:
     *
     *     C^-1 = K_uu^-1 - S
     *                                                  (Expansion of S)
     *          = K_uu^-1 - (K_uu + K_uf A^-1 K_fu)^-1
     *                                        (Woodbury Matrix Identity)
     *          = (K_uu^-1 K_uf (A + K_fu K_uu^-1 K_uf)^-1 K_fu K_uu^-1)
     *                                   (LL^T = K_uu and P = L^-1 K_uf)
     *          = L^-T P (A + P^T P)^-1 P^T L^-1
     *                                        (Searle Set of Identities)
     *          = L^-T P A^-1 P^T (I + P A^-1 P^T)^-1 L^-1
     *                         (B = (I + P A^-1 P^T) and R = A^-1/2 P^T)
     *          = L^-T R^T R B^-1 L^-1
     *
     *  reusing some of the precomputed values there leads to:
     *
     *     v = L^-T B^-1 P * A^-1 y
     */
    const auto A_ldlt = A.ldlt();
    Eigen::MatrixXd Pt = P.transpose();
    Eigen::MatrixXd RtR = A_ldlt.sqrt_solve(Pt);
    RtR = RtR.transpose() * RtR;
    const Eigen::MatrixXd B = Eigen::MatrixXd::Identity(m, m) + RtR;

    const Eigen::SerializableLDLT B_ldlt(B);

    SparseGPComponents components = {
        K_uu_ldlt, P, A_ldlt, RtR, B_ldlt, targets.mean,
    };

    return components;
  }

  template <typename FeatureType>
  auto _fit_impl(const std::vector<FeatureType> &features,
                 const MarginalDistribution &targets) const {

    // Determine the set of inducing points, u.
    const auto u =
        inducing_point_strategy_(this->covariance_function_, features);

    const auto sc = sparse_components(features, u, targets);

    const Eigen::Index m = sc.K_uu_ldlt.rows();
    const Eigen::MatrixXd L_uu_inv =
        sc.K_uu_ldlt.sqrt_solve(Eigen::MatrixXd::Identity(m, m));
    const Eigen::MatrixXd BiLi = sc.B_ldlt.solve(L_uu_inv);
    const Eigen::MatrixXd RtRBiLi = sc.RtR * BiLi;
    Eigen::VectorXd v = BiLi.transpose() * sc.P * sc.A_ldlt.solve(sc.y);

    const Eigen::MatrixXd C_inv = sc.K_uu_ldlt.sqrt_transpose_solve(RtRBiLi);
    DirectInverse solver(C_inv);

    using InducingPointFeatureType = typename std::decay<decltype(u[0])>::type;
    return Fit<GPFit<DirectInverse, InducingPointFeatureType>>(u, solver, v);
  }

  template <typename FeatureType>
  double log_likelihood(const RegressionDataset<FeatureType> &dataset) const {

    const auto u =
        inducing_point_strategy_(this->covariance_function_, dataset.features);

    const SparseGPComponents sc =
        sparse_components(dataset.features, u, dataset.targets);

    // The log likelihood for y ~ N(0, K) is:
    //
    //   L = 1/2 (n log(2 pi) + log(|K|) + y^T K^-1 y)
    //
    // where in our case we have
    //   K = A + Q_ff
    // and
    //   Q_ff = K_fu K_uu^-1 K_uf
    //
    // First we get the determinant, |K|:
    //
    //   |K| = |A + Q_ff|
    //       = |K_uu + K_uf A^-1 K_fu| |K_uu^-1| |A|
    //       = |LL^T + K_uf A^-1 K_fu| |L^-T L^-1| |A|
    //       = |L^-1| |LL^T + K_uf A^-1 K_fu| |L^-T| |A|
    //       = |I + L^-1 K_uf A^-1 K_fu L^-T| |A|
    //       = |I + P A^-1 P^T| |A|
    //       = |B| |A|
    //
    // which is derived here:
    //   https://bwengals.github.io/pymc3-fitcvfe-implementation-notes.html
    // though as of Jan 2020 there are typos in the derivation.

    double log_det_a = sc.A_ldlt.log_determinant();
    const double log_det_b = sc.B_ldlt.vectorD().array().log().sum();
    const double log_det = log_det_a + log_det_b;

    // Then we need the Mahalanobis distance. To do so we'll draw heavily
    // on some of the definitions used early in this module.
    //
    //   d = y^T K^-1 y
    //     = y^T (A + Q_ff)^-1 y
    //     = y^T (A^-1 + A^-1 K_fu S^-1 K_uf A^-1) y
    //         with S = K_uu + K_uf A^-1 K_fu  (woodbury identity)
    //     = y^T (A^-1 + A^-1 P^T B^-1 P A^-1) y
    //     = y^T A^-1 y + y^T A^-1 P^T B^-1 P A^-1 y
    //     = y^T (A^-1 y) + (B^{-1/2} P A^-1 y)^T (B^{-1/2} P A^-1 y)
    //     = y^T y_a + c^T c
    //
    // with
    //
    //   y_a = A^-1 y   and  c = B^{-1/2} P y_a

    Eigen::VectorXd y_a = sc.A_ldlt.solve(sc.y);
    Eigen::VectorXd c = sc.B_ldlt.sqrt_solve((sc.P * y_a).eval());

    double log_quadratic = sc.y.transpose() * y_a;
    log_quadratic -= c.transpose() * c;

    const double rank = static_cast<double>(sc.y.size());
    const double log_dimension = rank * log(2 * M_PI);

    return -0.5 * (log_det + log_quadratic + log_dimension) +
           this->prior_log_likelihood();
  }

private:
  Parameter measurement_nugget_;
  Parameter inducing_nugget_;
  InducingPointStrategy inducing_point_strategy_;
  GrouperFunction independent_group_function_;
};

template <typename CovFunc, typename MeanFunc, typename GrouperFunction,
          typename InducingPointStrategy>
auto sparse_gp_from_covariance_and_mean(CovFunc &&covariance_function,
                                        MeanFunc &&mean_function,
                                        GrouperFunction &&grouper_function,
                                        InducingPointStrategy &&strategy,
                                        const std::string &model_name) {
  return SparseGaussianProcessRegression<
      typename std::decay<CovFunc>::type, typename std::decay<MeanFunc>::type,
      typename std::decay<GrouperFunction>::type,
      typename std::decay<InducingPointStrategy>::type>(
      std::forward<CovFunc>(covariance_function),
      std::forward<MeanFunc>(mean_function),
      std::forward<GrouperFunction>(grouper_function),
      std::forward<InducingPointStrategy>(strategy), model_name);
};

template <typename CovFunc, typename MeanFunc, typename GrouperFunction>
auto sparse_gp_from_covariance_and_mean(CovFunc &&covariance_function,
                                        MeanFunc &&mean_function,
                                        GrouperFunction &&grouper_function,
                                        const std::string &model_name) {
  return sparse_gp_from_covariance_and_mean(
      std::forward<CovFunc>(covariance_function),
      std::forward<MeanFunc>(mean_function),
      std::forward<GrouperFunction>(grouper_function),
      StateSpaceInducingPointStrategy(), model_name);
};

template <typename CovFunc, typename GrouperFunction,
          typename InducingPointStrategy>
auto sparse_gp_from_covariance(CovFunc &&covariance_function,
                               GrouperFunction &&grouper_function,
                               InducingPointStrategy &&strategy,
                               const std::string &model_name) {
  return sparse_gp_from_covariance_and_mean(
      std::forward<CovFunc>(covariance_function), ZeroMean(),
      std::forward<GrouperFunction>(grouper_function),
      std::forward<InducingPointStrategy>(strategy), model_name);
};

template <typename CovFunc, typename GrouperFunction>
auto sparse_gp_from_covariance(CovFunc covariance_function,
                               GrouperFunction grouper_function,
                               const std::string &model_name) {
  return sparse_gp_from_covariance_and_mean(
      std::forward<CovFunc>(covariance_function), ZeroMean(),
      std::forward<GrouperFunction>(grouper_function),
      StateSpaceInducingPointStrategy(), model_name);
};

} // namespace albatross

#endif /* INCLUDE_ALBATROSS_MODELS_SPARSE_GP_H_ */
