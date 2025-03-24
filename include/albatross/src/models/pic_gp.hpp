/*
 * Copyright (C) 2024 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef INCLUDE_ALBATROSS_MODELS_PIC_GP_H_
#define INCLUDE_ALBATROSS_MODELS_PIC_GP_H_

namespace albatross {

/*
 * This class directly implements the PIC covariance structure:
 * conditional on the inducing points, each group of observations is
 * independent.  You can use it as a covariance function for a dense
 * GP model, and it should give you the same results as the PIC
 * approximation, but since it's implemented directly as a covariance
 * function, it's much less efficient than modelling using the
 * PICGaussianProcessRegression class defined below.  It is provided
 * for debugging purposes, since it can be tricky to follow what the
 * efficient implementation is doing.
 *
 * See the PIC GP model test suite for an example of how to use this
 * class.
 */
template <typename CovarianceType, typename InducingFeatureType,
          typename GrouperFunction>
class BruteForcePIC
    : public CovarianceFunction<
          BruteForcePIC<CovarianceType, InducingFeatureType, GrouperFunction>> {
public:
  CovarianceType cov_;
  std::vector<InducingFeatureType> inducing_points_;
  GrouperFunction grouper_;
  Eigen::LDLT<Eigen::MatrixXd> K_uu_ldlt_;

  BruteForcePIC(const std::vector<InducingFeatureType> &inducing_points,
                CovarianceType &&cov, GrouperFunction &&grouper)
      : cov_{cov}, inducing_points_{inducing_points}, grouper_{grouper},
        K_uu_ldlt_{cov_(inducing_points_, inducing_points_).ldlt()} {}

  template <typename X, typename Y>
  double _call_impl(const X &x, const Y &y) const {
    if (grouper_(x) == grouper_(y)) {
      return cov_(x, y);
    }

    Eigen::VectorXd K_xu(inducing_points_.size());
    Eigen::VectorXd K_uy(inducing_points_.size());
    for (Eigen::Index i = 0;
         i < static_cast<Eigen::Index>(inducing_points_.size()); ++i) {
      K_xu[i] = cov_(x, inducing_points_[i]);
      K_uy[i] = cov_(inducing_points_[i], y);
    }
    return K_xu.dot(K_uu_ldlt_.solve(K_uy));
  }
};

template <typename CovarianceType, typename InducingFeatureType,
          typename GrouperFunction>
auto make_brute_force_pic_covariance(
    const std::vector<InducingFeatureType> &inducing_points,
    CovarianceType &&cov, GrouperFunction &&grouper) {
  return BruteForcePIC<CovarianceType, InducingFeatureType, GrouperFunction>(
      inducing_points, std::forward<CovarianceType>(cov),
      std::forward<GrouperFunction>(grouper));
}

template <typename CovFunc, typename MeanFunc, typename GrouperFunction,
          typename InducingPointStrategy, typename QRImplementation>
class PICGaussianProcessRegression;

template <typename GrouperFunction, typename InducingFeatureType,
          typename FeatureType>
struct PICGPFit {};

template <typename GrouperFunction, typename FeatureType>
using PICGroupIndexType =
    typename std::result_of<GrouperFunction(const FeatureType &)>::type;

template <typename FeatureType> struct PICGroup {
  // Row indices are into the original set of features into the data
  // used to form S.
  Eigen::Index initial_row;
  Eigen::Index block_size;
  Eigen::Index block_index;
  albatross::RegressionDataset<FeatureType> dataset;
};

template <typename GrouperFunction, typename FeatureType>
using PICGroupMap = std::map<PICGroupIndexType<GrouperFunction, FeatureType>,
                             PICGroup<FeatureType>>;

template <typename GrouperFunction, typename InducingFeatureType,
          typename FeatureType>
struct Fit<PICGPFit<GrouperFunction, InducingFeatureType, FeatureType>> {
  using PermutationIndices = Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1>;
  using GroupIndexType = PICGroupIndexType<GrouperFunction, FeatureType>;
  using GroupMap = PICGroupMap<GrouperFunction, FeatureType>;

  std::vector<FeatureType> train_features;
  std::vector<InducingFeatureType> inducing_features;
  Eigen::SerializableLDLT train_covariance;
  Eigen::MatrixXd sigma_R;
  Eigen::PermutationMatrixX P;
  std::vector<Eigen::VectorXd> mean_w;
  Eigen::MatrixXd W;
  std::vector<Eigen::MatrixXd> covariance_Y;
  Eigen::MatrixXd Z;
  BlockDiagonalLDLT A_ldlt;
  GroupMap measurement_groups;
  Eigen::VectorXd information;
  Eigen::Index numerical_rank;
  std::vector<Eigen::SparseMatrix<double>> cols_Bs;
  GrouperFunction grouper;

  Fit(){};

  Fit(const std::vector<FeatureType> &features_,
      const std::vector<InducingFeatureType> &inducing_features_,
      const Eigen::SerializableLDLT &train_covariance_,
      const Eigen::MatrixXd &sigma_R_, const Eigen::PermutationMatrixX &P_,
      const std::vector<Eigen::VectorXd> &mean_w_, const Eigen::MatrixXd &W_,
      const std::vector<Eigen::MatrixXd> &covariance_Y_,
      const Eigen::MatrixXd &Z_, const BlockDiagonalLDLT &A_ldlt_,
      const GroupMap &measurement_groups_, const Eigen::VectorXd &information_,
      Eigen::Index numerical_rank_,
      const std::vector<Eigen::SparseMatrix<double>> cols_Bs_,
      GrouperFunction grouper_)
      : train_features(features_), inducing_features(inducing_features_),
        train_covariance(train_covariance_), sigma_R(sigma_R_), P(P_),
        mean_w(mean_w_), W(W_), covariance_Y(covariance_Y_), Z(Z_),
        A_ldlt(A_ldlt_), measurement_groups(measurement_groups_),
        information(information_), numerical_rank(numerical_rank_),
        cols_Bs(cols_Bs_), grouper(grouper_) {}

  void shift_mean(const Eigen::VectorXd &mean_shift) {
    ALBATROSS_ASSERT(mean_shift.size() == information.size());
    information += train_covariance.solve(mean_shift);
  }

  bool operator==(
      const Fit<PICGPFit<GrouperFunction, InducingFeatureType, FeatureType>>
          &other) const {
    return (train_features == other.train_features &&
            inducing_features == other.inducing_features &&
            train_covariance == other.train_covariance &&
            sigma_R == other.sigma_R && P.indices() == other.P.indices() &&
            mean_w == other.mean_w && W == W &&
            covariance_Y == other.covariance_Y && Z == other.Z &&
            A_ldlt == other.A_ldlt &&
            measurement_groups == other.measurement_groups &&
            information == other.information &&
            numerical_rank == other.numerical_rank && cols_Bs == other.cols_Bs);
  }
};

/*
 * This class implements an approximation technique for Gaussian
 * processes which relies on the following:
 *
 *  1. Observations occur in correlated groups
 *
 *  2. Conditional on the values of observations within the same group
 *     and on the values of the inducing points, an observation is
 *     independent of observations in other groups.
 *
 * This is known as the PIC (Partially Independent Conditional)
 * approximation and was introduced in
 *
 *     [1] Local and global sparse Gaussian process approximations
 *     Edward Snelson, Zoubin Ghahramani
 *     https://proceedings.mlr.press/v2/snelson07a/snelson07a.pdf
 *
 * The upshot of this method is that for enough small enough groups,
 * the fit performance scales as O(N M^2) for N observations and M
 * inducing points (just like FITC/PITC), but fine-grained local
 * information from correlation groups can provide better predictive
 * performance.
 *
 * Inducing point and group definitions are handled as in the
 * FITC/PITC approximation class (see sparse_gp.hpp).  If a prediction
 * point is not correlated with any of the training observations (as
 * defined by the grouper function), the resulting predicted
 * distribution will be computed using only the fitted inducing
 * points.
 *
 * Several operations supported by other Albatross GP model classes,
 * like "rebasing" (computing new inducing points from old ones) and
 * reduced-complexity updating of a fit with new information, are not
 * yet supported by the PIC approximation class.
 *
 */
template <typename CovFunc, typename MeanFunc, typename GrouperFunction,
          typename InducingPointStrategy,
          typename QRImplementation = DenseQRImplementation>
class PICGaussianProcessRegression
    : public GaussianProcessBase<CovFunc, MeanFunc,
                                 PICGaussianProcessRegression<
                                     CovFunc, MeanFunc, GrouperFunction,
                                     InducingPointStrategy, QRImplementation>> {
public:
  using Base = GaussianProcessBase<
      CovFunc, MeanFunc,
      PICGaussianProcessRegression<CovFunc, MeanFunc, GrouperFunction,
                                   InducingPointStrategy, QRImplementation>>;

  template <typename FeatureType>
  using GroupIndexType = PICGroupIndexType<GrouperFunction, FeatureType>;

  template <typename FeatureType>
  using GroupMap = PICGroupMap<GrouperFunction, FeatureType>;

  PICGaussianProcessRegression() : Base() { initialize_params(); };

  PICGaussianProcessRegression(const CovFunc &covariance_function,
                               const MeanFunc &mean_function)
      : Base(covariance_function, mean_function) {
    initialize_params();
  };
  PICGaussianProcessRegression(CovFunc &&covariance_function,
                               MeanFunc &&mean_function)
      : Base(std::move(covariance_function), std::move(mean_function)) {
    initialize_params();
  };

  PICGaussianProcessRegression(
      const CovFunc &covariance_function, const MeanFunc &mean_function,
      const GrouperFunction &independent_group_function,
      const InducingPointStrategy &inducing_point_strategy,
      const std::string &model_name)
      : Base(covariance_function, mean_function, model_name),
        inducing_point_strategy_(inducing_point_strategy),
        independent_group_function_(independent_group_function) {
    initialize_params();
  };
  PICGaussianProcessRegression(CovFunc &&covariance_function,
                               MeanFunc &&mean_function,
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

  void set_param(const std::string &name, const Parameter &param) override {
    if (name == details::measurement_nugget_name()) {
      measurement_nugget_ = param;
      return;
    } else if (name == details::inducing_nugget_name()) {
      inducing_nugget_ = param;
      return;
    }
    const bool success = set_param_if_exists_in_any(
        name, param, &this->covariance_function_, &this->mean_function_);
    ALBATROSS_ASSERT(success);
  }

  template <typename NewGrouper = GrouperFunction,
            typename PleaseDontChange = typename std::enable_if<
                std::is_same<typename std::remove_reference<NewGrouper>::type,
                             GrouperFunction>::value>::type>
  void update_grouper_function(NewGrouper &&f) {
    independent_group_function_ = std::forward<NewGrouper>(f);
  }

  template <typename NewStrategy = InducingPointStrategy,
            typename PleaseDontChange = typename std::enable_if<
                std::is_same<typename std::remove_reference<NewStrategy>::type,
                             InducingPointStrategy>::value>::type>
  void update_inducing_point_strategy(NewStrategy &&f) {
    inducing_point_strategy_ = std::forward<NewStrategy>(f);
  }

  // Here we create the QR decomposition of:
  //
  //   B = |A^-1/2 K_fu| = |Q_1| R P^T
  //       |K_uu^{T/2} |   |Q_2|
  //
  // which corresponds to the inverse square root of Sigma
  //
  //   Sigma = (B^T B)^-1
  std::unique_ptr<typename QRImplementation::QRType>
  compute_sigma_qr(const Eigen::SerializableLDLT &K_uu_ldlt,
                   const BlockDiagonalLDLT &A_ldlt,
                   const Eigen::MatrixXd &K_fu) const {
    Eigen::MatrixXd B(A_ldlt.rows() + K_uu_ldlt.rows(), K_uu_ldlt.rows());
    B.topRows(A_ldlt.rows()) = A_ldlt.sqrt_solve(K_fu);
    B.bottomRows(K_uu_ldlt.rows()) = K_uu_ldlt.sqrt_transpose();
    return QRImplementation::compute(B, Base::threads_.get());
  };

  template <
      typename FeatureType,
      std::enable_if_t<
          has_call_operator<CovFunc, FeatureType, FeatureType>::value, int> = 0>
  auto _fit_impl(const std::vector<FeatureType> &features,
                 const MarginalDistribution &targets) const {
    // Determine the set of inducing points, u.
    const auto u =
        inducing_point_strategy_(this->covariance_function_, features);
    ALBATROSS_ASSERT(u.size() > 0 && "Empty inducing points!");

    BlockDiagonalLDLT A_ldlt;
    Eigen::SerializableLDLT K_uu_ldlt;
    Eigen::MatrixXd K_fu;
    Eigen::VectorXd y;
    GroupMap<FeatureType> measurement_groups;
    compute_internal_components(u, features, targets, &A_ldlt, &K_uu_ldlt,
                                &K_fu, &y, &measurement_groups);
    auto B_qr = compute_sigma_qr(K_uu_ldlt, A_ldlt, K_fu);

    // To make a prediction, we will need to compute cross-terms with
    // sub-blocks of S^-1, where
    //
    // S^-1 = A^-1 - A^-1 K_fu (K_uu + K_uf A^-1 K_fu)^-1 K_uf A^-1
    //               \____________________________________________/
    //                                    Z^T Z
    //
    // The inverse part of the right-hand term of this is B_qr above,
    // and we already have A_ldlt, so we can form Z.
    //
    // Now to compute sub-blocks of S^-1, we just index into Z^T Z
    // appropriately, and if the term is diagonal, also solve using
    // the blocks of A^-1.
    const Eigen::MatrixXd Z = sqrt_solve(*B_qr, A_ldlt.solve(K_fu).transpose());

    Eigen::VectorXd y_augmented = Eigen::VectorXd::Zero(B_qr->rows());
    y_augmented.topRows(y.size()) = A_ldlt.sqrt_solve(y, Base::threads_.get());
    const Eigen::VectorXd v = B_qr->solve(y_augmented);

    using InducingPointFeatureType = typename std::decay<decltype(u[0])>::type;

    const Eigen::MatrixXd W = K_uu_ldlt.solve(
        (A_ldlt.solve(K_fu) - Z.transpose() * Z * K_fu).transpose());

    // TODO(@peddie): a lot of this can be batched
    std::vector<Eigen::MatrixXd> covariance_Y(measurement_groups.size());
    std::vector<Eigen::VectorXd> mean_w(measurement_groups.size());
    std::vector<Eigen::SparseMatrix<double>> cols_Bs(measurement_groups.size());
    const auto block_start_indices = A_ldlt.block_to_row_map();

    const auto precompute_block = [&, this](std::size_t block_number,
                                            Eigen::Index start_row) -> void {
      const Eigen::Index block_size = A_ldlt.blocks[block_number].rows();
      // K_fu is already computed, so we can form K_uB and K_uA by
      // appropriate use of sparse indexing matrices and avoid an O(N
      // M) operation.
      //
      // This nonsense would be a lot more straightforward with Eigen
      // 3.4's slicing and indexing API.
      Eigen::SparseMatrix<double> cols_B(features.size(), block_size);
      cols_B.reserve(block_size);

      for (Eigen::Index i = 0; i < block_size; ++i) {
        cols_B.insert(start_row + i, i) = 1.;
      }
      cols_B.makeCompressed();

      cols_Bs[block_number] = cols_B;
      // v_b \in R^b = A_BB^-1 (y_b - K_Bu v)
      Eigen::MatrixXd ydiff_b = cols_B.transpose() * (y - K_fu * v);
      const Eigen::MatrixXd mean_w_full =
          A_ldlt.blocks[block_number].solve(ydiff_b);
      mean_w[block_number] = mean_w_full;
      // Y \in R^(u x b) = K_uu^-1 K_uB
      covariance_Y[block_number] = K_uu_ldlt.solve(K_fu.transpose() * cols_B);
    };

    apply_map(block_start_indices, precompute_block, Base::threads_.get());

    using FitType =
        Fit<PICGPFit<GrouperFunction, InducingPointFeatureType, FeatureType>>;
    return FitType(features, u, K_uu_ldlt, get_R(*B_qr), get_P(*B_qr), mean_w,
                   W, covariance_Y, Z, A_ldlt, measurement_groups, v,
                   B_qr->rank(), cols_Bs, independent_group_function_);
  }

  // This is included to allow the SparseGP
  // to be compatible with fits generated
  // using a standard GP.
  using Base::_predict_impl;

  template <typename FeatureType, typename InducingFeatureType,
            typename FitFeatureType>
  Eigen::VectorXd _predict_impl(
      const std::vector<FeatureType> &features,
      const Fit<PICGPFit<GrouperFunction, InducingFeatureType, FitFeatureType>>
          &sparse_gp_fit,
      PredictTypeIdentity<Eigen::VectorXd> &&) const {
    const auto find_group = [this, &sparse_gp_fit](const auto &feature) {
      return sparse_gp_fit.measurement_groups.find(
          independent_group_function_(without_measurement(feature)));
    };
    const Eigen::MatrixXd K_up =
        this->covariance_function_(sparse_gp_fit.inducing_features, features);
    Eigen::VectorXd mean_correction = Eigen::VectorXd::Zero(features.size());
    for (Eigen::Index j = 0; j < features.size(); ++j) {
      const auto group = find_group(features[j]);
      if (group == sparse_gp_fit.measurement_groups.end()) {
        continue;
      }
      const std::vector<FeatureType> fvec = {features[j]};

      const Eigen::VectorXd features_cov =
          this->covariance_function_(group->second.dataset.features, fvec);
      const Eigen::VectorXd kpuy =
          K_up.transpose().row(j) *
          sparse_gp_fit.covariance_Y[group->second.block_index];
      const Eigen::VectorXd Vbp = features_cov - kpuy;
      mean_correction[j] =
          Vbp.dot(sparse_gp_fit.mean_w[group->second.block_index]);
    }

    Eigen::VectorXd mean =
        K_up.transpose() * sparse_gp_fit.information + mean_correction;

    this->mean_function_.add_to(features, &mean);
    return mean;
  }

  template <typename FeatureType, typename InducingFeatureType,
            typename FitFeatureType>
  MarginalDistribution _predict_impl(
      const std::vector<FeatureType> &features,
      const Fit<PICGPFit<GrouperFunction, InducingFeatureType, FitFeatureType>>
          &sparse_gp_fit,
      PredictTypeIdentity<MarginalDistribution> &&) const {
    const auto K_up =
        this->covariance_function_(sparse_gp_fit.inducing_features, features);
    Eigen::VectorXd mean = gp_mean_prediction(K_up, sparse_gp_fit.information);
    this->mean_function_.add_to(features, &mean);

    // First pass: compute an O(1) mapping from features to groups
    std::vector<std::size_t> feature_to_block(features.size());
    std::vector<decltype(sparse_gp_fit.measurement_groups.begin())> groups(
        features.size());
    Eigen::VectorXi col_alloc{Eigen::VectorXi::Zero(features.size())};
    bool all_same_group = true;
    for (Eigen::Index j = 0; j < features.size(); ++j) {
      groups[j] = sparse_gp_fit.measurement_groups.find(
          independent_group_function_(without_measurement(features[j])));
      if (groups[j] != sparse_gp_fit.measurement_groups.end()) {
        col_alloc(j) = groups[j]->second.block_size;
        all_same_group = all_same_group && groups[j] == groups[0];
      }
      feature_to_block[j] =
          std::distance(sparse_gp_fit.measurement_groups.begin(), groups[j]);
    }

    // Second pass: compute mean vector and fill sparse Vp matrix
    Eigen::VectorXd mean_correction = Eigen::VectorXd::Zero(features.size());
    Eigen::SparseMatrix<double> Vp(sparse_gp_fit.train_features.size(),
                                   features.size());
    Vp.reserve(col_alloc);
    for (Eigen::Index j = 0; j < features.size(); ++j) {
      if (groups[j] == sparse_gp_fit.measurement_groups.end()) {
        continue;
      }

      const auto &B = groups[j]->second;
      Eigen::VectorXd Vbp =
          this->covariance_function_(B.dataset.features,
                                     std::vector<FeatureType>{features[j]}) -
          K_up.col(j).transpose() * sparse_gp_fit.covariance_Y[B.block_index];
      for (Eigen::Index i = 0; i < Vbp.size(); ++i) {
        Vp.insert(B.initial_row + i, j) = Vbp(i);
      }
      mean_correction[j] = Vbp.dot(sparse_gp_fit.mean_w[B.block_index]);
    }
    Vp.makeCompressed();

    Eigen::MatrixXd xi_lambda = sparse_gp_fit.A_ldlt.sqrt_solve(Vp);
    Eigen::MatrixXd xi_u = sparse_gp_fit.Z * Vp;
    Eigen::VectorXd VSV_diag{
        (xi_lambda.transpose() * xi_lambda - xi_u.transpose() * xi_u)
            .diagonal()};

    const Eigen::VectorXd U_diag =
        (K_up.transpose() * sparse_gp_fit.W * Vp).diagonal();

    Eigen::VectorXd marginal_variance(cast::to_index(features.size()));
    for (Eigen::Index i = 0; i < marginal_variance.size(); ++i) {
      marginal_variance[i] = this->covariance_function_(
          features[cast::to_size(i)], features[cast::to_size(i)]);
    }

    const Eigen::MatrixXd Q_sqrt =
        sparse_gp_fit.train_covariance.sqrt_solve(K_up);
    const Eigen::VectorXd Q_diag =
        Q_sqrt.cwiseProduct(Q_sqrt).array().colwise().sum();
    marginal_variance -= Q_diag;

    const Eigen::MatrixXd S_sqrt =
        sqrt_solve(sparse_gp_fit.sigma_R, sparse_gp_fit.P, K_up);
    const Eigen::VectorXd S_diag =
        S_sqrt.cwiseProduct(S_sqrt).array().colwise().sum();
    marginal_variance += S_diag;

    mean += mean_correction;

    return MarginalDistribution(mean,
                                marginal_variance - (2 * U_diag + VSV_diag));
  }

  template <typename FeatureType, typename InducingFeatureType,
            typename FitFeatureType>
  JointDistribution _predict_impl(
      const std::vector<FeatureType> &features,
      const Fit<PICGPFit<GrouperFunction, InducingFeatureType, FitFeatureType>>
          &sparse_gp_fit,
      PredictTypeIdentity<JointDistribution> &&) const {
    // K_up
    const Eigen::MatrixXd K_up =
        this->covariance_function_(sparse_gp_fit.inducing_features, features);
    const Eigen::MatrixXd prior_cov = this->covariance_function_(features);

    // First pass: compute an O(1) mapping from features to groups
    std::vector<std::size_t> feature_to_block(features.size());
    std::vector<decltype(sparse_gp_fit.measurement_groups.begin())> groups(
        features.size());
    Eigen::VectorXi col_alloc{Eigen::VectorXi::Zero(features.size())};
    bool all_same_group = true;
    for (Eigen::Index j = 0; j < features.size(); ++j) {
      groups[j] = sparse_gp_fit.measurement_groups.find(
          independent_group_function_(without_measurement(features[j])));
      if (groups[j] != sparse_gp_fit.measurement_groups.end()) {
        col_alloc(j) = groups[j]->second.block_size;
        all_same_group = all_same_group && groups[j] == groups[0];
      }
      feature_to_block[j] =
          std::distance(sparse_gp_fit.measurement_groups.begin(), groups[j]);
    }

    // Second pass: compute mean vector and fill sparse Vp matrix
    Eigen::VectorXd mean_correction = Eigen::VectorXd::Zero(features.size());
    Eigen::SparseMatrix<double> Vp(sparse_gp_fit.train_features.size(),
                                   features.size());
    Vp.reserve(col_alloc);
    for (Eigen::Index j = 0; j < features.size(); ++j) {
      if (groups[j] == sparse_gp_fit.measurement_groups.end()) {
        // Fall back to PITC
        continue;
      }

      const auto &B = groups[j]->second;
      // TODO(@peddie): there must be some way to assign this
      // expression directly to a map of the column memory
      Eigen::VectorXd Vbp =
          this->covariance_function_(B.dataset.features,
                                     std::vector<FeatureType>{features[j]}) -
          K_up.col(j).transpose() * sparse_gp_fit.covariance_Y[B.block_index];
      for (Eigen::Index i = 0; i < Vbp.size(); ++i) {
        Vp.insert(B.initial_row + i, j) = Vbp(i);
      }
      mean_correction[j] = Vbp.dot(sparse_gp_fit.mean_w[B.block_index]);
    }
    Vp.makeCompressed();

    Eigen::MatrixXd xi_lambda = sparse_gp_fit.A_ldlt.sqrt_solve(Vp);
    Eigen::MatrixXd xi_u = sparse_gp_fit.Z * Vp;
    Eigen::MatrixXd VSV{xi_lambda.transpose() * xi_lambda -
                        xi_u.transpose() * xi_u};

    const Eigen::MatrixXd U = K_up.transpose() * sparse_gp_fit.W * Vp;

    const Eigen::MatrixXd S_sqrt =
        sqrt_solve(sparse_gp_fit.sigma_R, sparse_gp_fit.P, K_up);

    const Eigen::MatrixXd Q_sqrt =
        sparse_gp_fit.train_covariance.sqrt_solve(K_up);

    const Eigen::MatrixXd max_explained = Q_sqrt.transpose() * Q_sqrt;
    const Eigen::MatrixXd unexplained = S_sqrt.transpose() * S_sqrt;

    const Eigen::MatrixXd pitc_covariance =
        prior_cov - max_explained + unexplained;

    const Eigen::MatrixXd pic_correction = U + U.transpose() + VSV;
    JointDistribution pred(K_up.transpose() * sparse_gp_fit.information +
                               mean_correction,
                           pitc_covariance - pic_correction);
    this->mean_function_.add_to(features, &pred.mean);

    return pred;
  }

  InducingPointStrategy get_inducing_point_strategy() const {
    return inducing_point_strategy_;
  }

private:
  // This method takes care of a lot of the common book keeping required to
  // setup the Sparse Gaussian Process problem.  Namely, we want to get from
  // possibly unordered features to a structured representation
  // in the form K_ff = A_ff + Q_ff where Q_ff = K_fu K_uu^-1 K_uf and
  // A_ff is block diagonal and is formed by subtracting Q_ff from K_ff.
  //
  template <typename InducingFeatureType, typename FeatureType>
  void compute_internal_components(
      const std::vector<InducingFeatureType> &inducing_features,
      const std::vector<FeatureType> &out_of_order_features,
      const MarginalDistribution &out_of_order_targets,
      BlockDiagonalLDLT *A_ldlt, Eigen::SerializableLDLT *K_uu_ldlt,
      Eigen::MatrixXd *K_fu, Eigen::VectorXd *y,
      GroupMap<FeatureType> *measurement_groups) const {

    ALBATROSS_ASSERT(A_ldlt != nullptr);
    ALBATROSS_ASSERT(K_uu_ldlt != nullptr);
    ALBATROSS_ASSERT(K_fu != nullptr);
    ALBATROSS_ASSERT(y != nullptr);

    const auto indexer =
        group_by(out_of_order_features, independent_group_function_).indexers();

    const auto out_of_order_measurement_features =
        as_measurements(out_of_order_features);

    std::vector<std::size_t> reordered_inds;
    BlockDiagonal K_ff;
    Eigen::Index begin_row_index = 0;
    Eigen::Index block_index = 0;
    // TODO(@peddie): compute these blocks asynchronously?
    for (const auto &pair : indexer) {
      reordered_inds.insert(reordered_inds.end(), pair.second.begin(),
                            pair.second.end());
      measurement_groups->operator[](pair.first) = PICGroup<FeatureType>{
          begin_row_index, static_cast<Eigen::Index>(pair.second.size()),
          block_index,
          albatross::RegressionDataset<FeatureType>(
              subset(out_of_order_features, pair.second),
              out_of_order_targets.subset(pair.second))};
      auto subset_features =
          subset(out_of_order_measurement_features, pair.second);
      K_ff.blocks.emplace_back(this->covariance_function_(subset_features));
      K_ff.blocks.back().diagonal() +=
          subset(out_of_order_targets.covariance.diagonal(), pair.second);
      begin_row_index += pair.second.size();
      ++block_index;
    }

    const auto features =
        subset(out_of_order_measurement_features, reordered_inds);
    auto targets = subset(out_of_order_targets, reordered_inds);
    *y = targets.mean;

    this->mean_function_.remove_from(
        subset(out_of_order_features, reordered_inds), &targets.mean);

    *K_fu = this->covariance_function_(features, inducing_features,
                                       Base::threads_.get());

    auto K_uu =
        this->covariance_function_(inducing_features, Base::threads_.get());
    K_uu.diagonal() +=
        inducing_nugget_.value * Eigen::VectorXd::Ones(K_uu.rows());

    *K_uu_ldlt = K_uu.ldlt();
    // P is such that:
    //     Q_ff = K_fu K_uu^-1 K_uf
    //          = K_fu K_uu^-T/2 K_uu^-1/2 K_uf
    //          = P^T P
    const Eigen::MatrixXd P = K_uu_ldlt->sqrt_solve(K_fu->transpose());

    // We only need the diagonal blocks of Q_ff to get A
    BlockDiagonal Q_ff_diag;
    Eigen::Index i = 0;
    for (const auto &pair : indexer) {
      Eigen::Index cols = cast::to_index(pair.second.size());
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

    *A_ldlt = A.ldlt();
  }

  Parameter measurement_nugget_;
  Parameter inducing_nugget_;
  InducingPointStrategy inducing_point_strategy_;
  GrouperFunction independent_group_function_;
};

template <typename CovFunc, typename MeanFunc, typename GrouperFunction,
          typename InducingPointStrategy>
using SparseQRPicGaussianProcessRegression =
    PICGaussianProcessRegression<CovFunc, GrouperFunction,
                                 InducingPointStrategy, SPQRImplementation>;

template <typename CovFunc, typename MeanFunc, typename GrouperFunction,
          typename InducingPointStrategy,
          typename QRImplementation = DenseQRImplementation>
auto pic_gp_from_covariance_and_mean(
    CovFunc &&covariance_function, MeanFunc &&mean_function,
    GrouperFunction &&grouper_function, InducingPointStrategy &&strategy,
    const std::string &model_name,
    QRImplementation qr __attribute__((unused)) = DenseQRImplementation{}) {
  return PICGaussianProcessRegression<
      typename std::decay<CovFunc>::type, typename std::decay<MeanFunc>::type,
      typename std::decay<GrouperFunction>::type,
      typename std::decay<InducingPointStrategy>::type,
      typename std::decay<QRImplementation>::type>(
      std::forward<CovFunc>(covariance_function),
      std::forward<MeanFunc>(mean_function),
      std::forward<GrouperFunction>(grouper_function),
      std::forward<InducingPointStrategy>(strategy), model_name);
};

template <typename CovFunc, typename MeanFunc, typename GrouperFunction,
          typename QRImplementation = DenseQRImplementation>
auto pic_gp_from_covariance_and_mean(
    CovFunc &&covariance_function, MeanFunc &&mean_function,
    GrouperFunction &&grouper_function, const std::string &model_name,
    QRImplementation qr = DenseQRImplementation{}) {
  return pic_gp_from_covariance_and_mean(
      std::forward<CovFunc>(covariance_function),
      std::forward<MeanFunc>(mean_function),
      std::forward<GrouperFunction>(grouper_function),
      StateSpaceInducingPointStrategy(), model_name, qr);
};

template <typename CovFunc, typename GrouperFunction,
          typename InducingPointStrategy,
          typename QRImplementation = DenseQRImplementation>
auto pic_gp_from_covariance(CovFunc &&covariance_function,
                            GrouperFunction &&grouper_function,
                            InducingPointStrategy &&strategy,
                            const std::string &model_name,
                            QRImplementation qr = DenseQRImplementation{}) {
  return pic_gp_from_covariance_and_mean<CovFunc, decltype(ZeroMean()),
                                         GrouperFunction, InducingPointStrategy,
                                         QRImplementation>(
      std::forward<CovFunc>(covariance_function), ZeroMean(),
      std::forward<GrouperFunction>(grouper_function),
      std::forward<InducingPointStrategy>(strategy), model_name, qr);
};

template <typename CovFunc, typename GrouperFunction,
          typename QRImplementation = DenseQRImplementation>
auto pic_gp_from_covariance(CovFunc covariance_function,
                            GrouperFunction grouper_function,
                            const std::string &model_name,
                            QRImplementation qr = DenseQRImplementation{}) {
  return pic_gp_from_covariance_and_mean<
      CovFunc, decltype(ZeroMean()), GrouperFunction,
      decltype(StateSpaceInducingPointStrategy()), QRImplementation>(
      std::forward<CovFunc>(covariance_function), ZeroMean(),
      std::forward<GrouperFunction>(grouper_function),
      StateSpaceInducingPointStrategy(), model_name, qr);
};

} // namespace albatross

#endif /* INCLUDE_ALBATROSS_MODELS_PIC_GP_H_ */
