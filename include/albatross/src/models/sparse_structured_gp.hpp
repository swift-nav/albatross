/*
 * Copyright (C) 2023 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef INCLUDE_ALBATROSS_MODELS_SPARSE_STRUC
#define INCLUDE_ALBATROSS_MODELS_SPARSE_STRUC

namespace albatross {

template <typename CovFunc, typename MeanFunc, typename GrouperFunction,
          typename InducingPointStrategy>
class SparseStructuredGaussianProcessRegression;

template <typename Key, typename FeatureType>
struct LocalGlobalInducingPoints {
  std::vector<FeatureType> global;
  std::map<Key, std::vector<FeatureType>> local;
};

template <typename GroupKey, typename FeatureType> struct SparseStructuredGPFit {};

template <typename GroupKey, typename FeatureType>
struct Fit<SparseStructuredGPFit<GroupKey, FeatureType>> {
  GroupIndexer<GroupKey> indexer;
  Grouped<GroupKey, std::vector<FeatureType>> train_features;
  GroupKey common_key;

  BlockSymmetricArrowLDLT train_covariance;
  StructuredR sigma_R;

  Eigen::VectorXd information;
  Eigen::Index numerical_rank;

  Fit(){};

  // Fit(const std::vector<FeatureType> &features_,
  //     const Eigen::SerializableLDLT &train_covariance_,
  //     const Eigen::MatrixXd &sigma_R_,
  //     PermutationIndices &&permutation_indices_,
  //     const Eigen::VectorXd &information_, Eigen::Index numerical_rank_)
  //     : train_features(features_), train_covariance(train_covariance_),
  //       sigma_R(sigma_R_), permutation_indices(std::move(permutation_indices_)),
  //       information(information_), numerical_rank(numerical_rank_) {}

  // bool operator==(const Fit<SparseGPFit<FeatureType>> &other) const {
  //   return (train_features == other.train_features &&
  //           train_covariance == other.train_covariance &&
  //           sigma_R == other.sigma_R &&
  //           permutation_indices == other.permutation_indices &&
  //           information == other.information &&
  //           numerical_rank == other.numerical_rank);
  // }
};

template <typename CovFunc, typename MeanFunc, typename GrouperFunction,
          typename InducingPointStrategy>
class SparseStructuredGaussianProcessRegression
    : public ModelBase<SparseStructuredGaussianProcessRegression<
                                     CovFunc, MeanFunc, GrouperFunction,
  InducingPointStrategy>> {

protected:
  template <typename CovarianceRepresentation, typename FitFeatureType>
  using GPFitType = Fit<GPFit<CovarianceRepresentation, FitFeatureType>>;

  CovFunc covariance_function_;
  MeanFunc mean_function_;
  std::string model_name_;
  InducingPointStrategy inducing_point_strategy_;
  GrouperFunction independent_group_function_;
  Parameter measurement_nugget_;
  Parameter inducing_nugget_;

public:

  SparseStructuredGaussianProcessRegression() : covariance_function_(),
                                                mean_function_(),
        model_name_(default_model_name(covariance_function_, mean_function_)){};

  SparseStructuredGaussianProcessRegression(
      const CovFunc &covariance_function, const MeanFunc &mean_function,
      const GrouperFunction &independent_group_function,
      const InducingPointStrategy &inducing_point_strategy,
      const std::string &model_name)
      : covariance_function_(covariance_function),
        mean_function_(mean_function), model_name_(model_name),
        inducing_point_strategy_(inducing_point_strategy),
        independent_group_function_(independent_group_function),
        measurement_nugget_(
            {details::cDefaultNugget,
             LogScaleUniformPrior(PARAMETER_EPSILON, PARAMETER_MAX)}),
        inducing_nugget_(
            {details::cDefaultNugget,
             LogScaleUniformPrior(PARAMETER_EPSILON, PARAMETER_MAX)}){};

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

  // template <typename FeatureType, typename InducingPointFeatureType>
  // auto _update_impl(const Fit<SparseGPFit<InducingPointFeatureType>>
  // &old_fit,
  //                   const std::vector<FeatureType> &features,
  //                   const MarginalDistribution &targets) const {

  //   BlockDiagonalLDLT A_ldlt;
  //   Eigen::SerializableLDLT K_uu_ldlt;
  //   Eigen::MatrixXd K_fu;
  //   Eigen::VectorXd y;
  //   compute_internal_components(old_fit.train_features, features, targets,
  //                               &A_ldlt, &K_uu_ldlt, &K_fu, &y);

  //   const Eigen::Index n_old = old_fit.sigma_R.rows();
  //   const Eigen::Index n_new = A_ldlt.rows();
  //   const Eigen::Index k = old_fit.sigma_R.cols();
  //   Eigen::MatrixXd B = Eigen::MatrixXd::Zero(n_old + n_new, k);

  //   ALBATROSS_ASSERT(n_old == k);

  //   // Form:
  //   //   B = |R_old P_old^T| = |Q_1| R P^T
  //   //       |A^{-1/2} K_fu|   |Q_2|
  //   for (Eigen::Index i = 0; i < old_fit.permutation_indices.size(); ++i) {
  //     const Eigen::Index &pi = old_fit.permutation_indices.coeff(i);
  //     B.col(pi).topRows(i + 1) = old_fit.sigma_R.col(i).topRows(i + 1);
  //   }
  //   B.bottomRows(n_new) = A_ldlt.sqrt_solve(K_fu);
  //   const auto B_qr = QRImplementation::compute(B, Base::threads_.get());

  //   // Form:
  //   //   y_aug = |R_old P_old^T v_old|
  //   //           |A^{-1/2} y         |
  //   ALBATROSS_ASSERT(old_fit.information.size() == n_old);
  //   Eigen::VectorXd y_augmented(n_old + n_new);
  //   for (Eigen::Index i = 0; i < old_fit.permutation_indices.size(); ++i) {
  //     y_augmented[i] =
  //         old_fit.information[old_fit.permutation_indices.coeff(i)];
  //   }
  //   y_augmented.topRows(n_old) =
  //       old_fit.sigma_R.template triangularView<Eigen::Upper>() *
  //       y_augmented.topRows(n_old);

  //   y_augmented.bottomRows(n_new) = A_ldlt.sqrt_solve(y,
  //   Base::threads_.get()); const Eigen::VectorXd v =
  //   B_qr->solve(y_augmented);

  //   Eigen::MatrixXd R = get_R(*B_qr);
  //   if (B_qr->rank() < B_qr->cols()) {
  //     // Inflate the diagonal of R in an attempt to avoid singularity
  //     R.diagonal() +=
  //         Eigen::VectorXd::Constant(B_qr->cols(), details::cSparseRNugget);
  //   }
  //   using FitType = Fit<SparseGPFit<InducingPointFeatureType>>;
  //   return FitType(
  //       old_fit.train_features, old_fit.train_covariance, R,
  //       B_qr->colsPermutation().indices().template cast<Eigen::Index>(), v,
  //       B_qr->rank());
  // }

  template <
      typename FeatureType,
      std::enable_if_t<
          has_call_operator<CovFunc, FeatureType, FeatureType>::value, int> = 0>
  auto _fit_impl(const std::vector<FeatureType> &features,
                 const MarginalDistribution &targets) const {
    // Determine the set of inducing points, u.

    const auto u = inducing_point_strategy_(this->covariance_function_, features);

    ALBATROSS_ASSERT(u.global.size() > 0 && "Empty global inducing points!");
    ALBATROSS_ASSERT(u.local.size() > 0 && "Empty local inducing points!");

    auto build_block = [&](const auto &, const auto &u_i) -> Eigen::MatrixXd {
      return covariance_function_(u_i);
    };

    auto K_ll_blocks = albatross::apply(u.local, build_block);
    std::size_t i = 0;

    using KeyType = typename decltype(u.local)::key_t;
    std::map<KeyType, std::size_t> key_to_index;
    for (const auto &pair : K_ll_blocks) {
      key_to_index[pair.first] = i;
      ++i;
    };

    const auto K_ll = BlockDiagonal{K_ll_blocks.values()};
    const auto K_gg = covariance_function_(u.global);
    const Eigen::MatrixXd K_lg = Eigen::MatrixXd::Zero(K_ll.rows(), u.global.size());
    const auto K_uu_ldlt = block_symmetric_arrow_ldlt(K_ll, K_lg, K_gg);

    BlockDiagonalLDLT A_ldlt;
    StructuredQR sigma_qr;
    Eigen::VectorXd y;
    compute_internal_components(u, features, targets, &A_ldlt, &K_uu_ldlt,
                                &sigma_qr, &y);

    Eigen::VectorXd y_augmented = Eigen::VectorXd::Zero(sigma_R.rows());
    y_augmented.topRows(y.size()) = A_ldlt.sqrt_solve(y, Base::threads_.get());
    const Eigen::VectorXd v = sigma_qr->solve(y_augmented);

    using InducingPointFeatureType = typename std::decay<decltype(u[0])>::type;

    using FitType = Fit<SparseGPFit<InducingPointFeatureType>>;
    return FitType(
        u, K_uu_ldlt, get_R(*B_qr),
        B_qr->colsPermutation().indices().template cast<Eigen::Index>(), v,
        B_qr->rank());
  }

  template <typename FeatureType>
  auto fit_from_prediction(const std::vector<FeatureType> &new_inducing_points,
                           const JointDistribution &prediction_) const {
    FitModel<SparseStructuredGaussianProcessRegression, Fit<SparseGPFit<FeatureType>>>
        output(*this, Fit<SparseGPFit<FeatureType>>());
    Fit<SparseGPFit<FeatureType>> &new_fit = output.get_fit();

    new_fit.train_features = new_inducing_points;

    const Eigen::MatrixXd K_zz =
        this->covariance_function_(new_inducing_points, Base::threads_.get());
    new_fit.train_covariance = Eigen::SerializableLDLT(K_zz);

    // We're going to need to take the sqrt of the new covariance which
    // could be extremely small, so here we add a small nugget to avoid
    // numerical instability
    JointDistribution prediction(prediction_);
    prediction.covariance.diagonal() += Eigen::VectorXd::Constant(
        cast::to_index(prediction.size()), 1, details::DEFAULT_NUGGET);
    new_fit.information = new_fit.train_covariance.solve(prediction.mean);

    // Here P is the posterior covariance at the new inducing points.  If
    // we consider the case where we rebase and then use the resulting fit
    // to predict the new inducing points then we see that the predictive
    // covariance (see documentation above) would be,:
    //
    //    C = K_zz - Q_zz + K_zz Sigma K_zz
    //
    // We can use this, knowing that at the inducing points K_zz = Q_zz, to
    // derive our updated Sigma,
    //
    //    C = K_zz - K_zz + K_zz Sigma K_zz
    //    C  = K_zz Sigma K_zz
    //    Sigma = K_zz^-1 C K_zz^-1
    //
    // And since we need to store Sigma in sqrt form we get,
    //
    //    Sigma = (B_z^T B_z)^-1
    //          = K_zz^-1 C K_zz^-1
    //
    // So by setting:
    //
    //    B_z = C^{-1/2} K_z
    //
    // We can then compute and store the QR decomposition of B
    // as we do in a normal fit.
    const Eigen::SerializableLDLT C_ldlt(prediction.covariance);
    const Eigen::MatrixXd sigma_inv_sqrt = C_ldlt.sqrt_solve(K_zz);
    const auto B_qr = QRImplementation::compute(sigma_inv_sqrt, nullptr);

    new_fit.permutation_indices =
        B_qr->colsPermutation().indices().template cast<Eigen::Index>();
    new_fit.sigma_R = get_R(*B_qr);
    new_fit.numerical_rank = B_qr->rank();

    return output;
  }

  template <typename FeatureType, typename FitFeaturetype>
  Eigen::VectorXd
  _predict_impl(const std::vector<FeatureType> &features,
                const Fit<SparseGPFit<FitFeaturetype>> &sparse_gp_fit,
                PredictTypeIdentity<Eigen::VectorXd> &&) const {
    const auto cross_cov =
        this->covariance_function_(sparse_gp_fit.train_features, features);
    Eigen::VectorXd mean =
        gp_mean_prediction(cross_cov, sparse_gp_fit.information);
    this->mean_function_.add_to(features, &mean);
    return mean;
  }

  template <typename FeatureType, typename FitFeaturetype>
  MarginalDistribution
  _predict_impl(const std::vector<FeatureType> &features,
                const Fit<SparseGPFit<FitFeaturetype>> &sparse_gp_fit,
                PredictTypeIdentity<MarginalDistribution> &&) const {
    const auto cross_cov =
        this->covariance_function_(sparse_gp_fit.train_features, features);
    Eigen::VectorXd mean =
        gp_mean_prediction(cross_cov, sparse_gp_fit.information);
    this->mean_function_.add_to(features, &mean);

    Eigen::VectorXd marginal_variance(cast::to_index(features.size()));
    for (Eigen::Index i = 0; i < marginal_variance.size(); ++i) {
      marginal_variance[i] = this->covariance_function_(
          features[cast::to_size(i)], features[cast::to_size(i)]);
    }

    const Eigen::MatrixXd Q_sqrt =
        sparse_gp_fit.train_covariance.sqrt_solve(cross_cov);
    const Eigen::VectorXd Q_diag =
        Q_sqrt.cwiseProduct(Q_sqrt).array().colwise().sum();
    marginal_variance -= Q_diag;

    const Eigen::MatrixXd S_sqrt = sqrt_solve(
        sparse_gp_fit.sigma_R, sparse_gp_fit.permutation_indices, cross_cov);
    const Eigen::VectorXd S_diag =
        S_sqrt.cwiseProduct(S_sqrt).array().colwise().sum();
    marginal_variance += S_diag;

    return MarginalDistribution(mean, marginal_variance);
  }

  template <typename FeatureType, typename FitFeaturetype>
  JointDistribution
  _predict_impl(const std::vector<FeatureType> &features,
                const Fit<SparseGPFit<FitFeaturetype>> &sparse_gp_fit,
                PredictTypeIdentity<JointDistribution> &&) const {
    const auto cross_cov =
        this->covariance_function_(sparse_gp_fit.train_features, features);
    const Eigen::MatrixXd prior_cov = this->covariance_function_(features);

    const Eigen::MatrixXd S_sqrt = sqrt_solve(
        sparse_gp_fit.sigma_R, sparse_gp_fit.permutation_indices, cross_cov);

    const Eigen::MatrixXd Q_sqrt =
        sparse_gp_fit.train_covariance.sqrt_solve(cross_cov);

    const Eigen::MatrixXd max_explained = Q_sqrt.transpose() * Q_sqrt;
    const Eigen::MatrixXd unexplained = S_sqrt.transpose() * S_sqrt;
    const Eigen::MatrixXd covariance = prior_cov - max_explained + unexplained;

    JointDistribution pred(cross_cov.transpose() * sparse_gp_fit.information,
                           covariance);

    this->mean_function_.add_to(features, &pred.mean);
    return pred;
  }

  template <typename FeatureType>
  double log_likelihood(const RegressionDataset<FeatureType> &dataset) const {
    const auto u =
        inducing_point_strategy_(this->covariance_function_, dataset.features);

    BlockDiagonalLDLT A_ldlt;
    Eigen::SerializableLDLT K_uu_ldlt;
    Eigen::MatrixXd K_fu;
    Eigen::VectorXd y;
    compute_internal_components(u, dataset.features, dataset.targets, &A_ldlt,
                                &K_uu_ldlt, &K_fu, &y);
    const auto B_qr = compute_sigma_qr(K_uu_ldlt, A_ldlt, K_fu);
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
    //       = |P R^T Q^T Q R P^T| |K_uu^-1| |A|
    //       = |R^T R| |K_uu^-1| |A|
    //       = |R|^2 |A| / |K_uu|
    //
    // Where the first equality comes from the matrix determinant lemma.
    // https://en.wikipedia.org/wiki/Matrix_determinant_lemma#Generalization
    //
    // After which we can take the log:
    //
    //   log(|K|) = 2 log(|R|) + log(|A|) - log(|K_uu|)
    //
    const double log_det_a = A_ldlt.log_determinant();

    const double log_det_r =
        B_qr->matrixR().diagonal().array().cwiseAbs().log().sum();
    const double log_det_K_uu = K_uu_ldlt.log_determinant();
    const double log_det = log_det_a + 2 * log_det_r - log_det_K_uu;

    // q = y^T K^-1 y
    //   = y^T (A + Q_ff)^-1 y
    //   = y^T (A^-1 - A^-1 K_fu (K_uu + K_uf A^-1 K_fu)^-1 K_uf A^-1) y
    //   = y^T A^-1 y - y^T A^-1 K_fu (K_uu + K_uf A^-1 K_fu)^-1 K_uf A^-1) y
    //   = y^T A^-1 y - y^T A^-1 K_fu (R^T R)^-1 K_uf A^-1) y
    //   = y^T A^-1 y - (R^-T K_uf A^-1 y)^T (R^-T K_uf A^-1 y)
    //   = y^T y_a - y_b^T y_b
    //
    // with y_b = R^-T K_uf y_a
    const Eigen::VectorXd y_a = A_ldlt.solve(y);

    Eigen::VectorXd y_b = K_fu.transpose() * y_a;
    y_b = sqrt_solve(*B_qr, y_b);

    double log_quadratic = y.transpose() * y_a;
    log_quadratic -= y_b.transpose() * y_b;

    const double rank = cast::to_double(y.size());
    const double log_dimension = rank * log(2 * M_PI);

    return -0.5 * (log_det + log_quadratic + log_dimension) +
           this->prior_log_likelihood();
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
  template <typename Key, typename InducingFeatureType, typename FeatureType>
  void compute_internal_components(
                                   const LocalGlobalInducingPoints<Key, InducingFeatureType> &inducing_points,
      const std::vector<FeatureType> &out_of_order_features,
      const MarginalDistribution &out_of_order_targets,
      BlockDiagonalLDLT *A_ldlt,
      BlockSymmetricArrowLDLT *K_uu_ldlt,
      StructuredQR *sigma_qr,
      Eigen::VectorXd *y) const {

    ALBATROSS_ASSERT(A_ldlt != nullptr);
    ALBATROSS_ASSERT(K_uu_ldlt != nullptr);
    ALBATROSS_ASSERT(K_fu != nullptr);
    ALBATROSS_ASSERT(y != nullptr);

    const auto indexer =
        group_by(out_of_order_features, independent_group_function_).indexers();

    static_assert(std::is_same<decltype(indexer)::key_t, Key>::value);
    for (const auto &pair : indexer) {
      if (!map_contains(inducing_points.local, pair.first)) {
        std::cerr << "Observations had a group " << pair.first << " which does not exist in the inducing points";
        assert(false && "observations had a group which does not exist in inducing points");
      }
    }

    const auto out_of_order_measurement_features =
        as_measurements(out_of_order_features);

    std::vector<std::size_t> reordered_inds;
    BlockDiagonal K_ff;
    // TODO(@peddie): compute these blocks asynchronously?
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
};

// rebase_inducing_points takes a Sparse GP which was fit using some set of
// inducing points and creates a new fit relative to new inducing points.
// Note that this will NOT be the equivalent to having fit the model with
// the new inducing points since some information may have been lost in
// the process.
template <typename ModelType, typename FeatureType, typename NewFeatureType>
auto rebase_inducing_points(
    const FitModel<ModelType, Fit<SparseGPFit<FeatureType>>> &fit_model,
    const std::vector<NewFeatureType> &new_inducing_points) {
  return fit_model.get_model().fit_from_prediction(
      new_inducing_points, fit_model.predict(new_inducing_points).joint());
}

template <typename CovFunc, typename MeanFunc, typename GrouperFunction,
          typename InducingPointStrategy,
          typename QRImplementation = DenseQRImplementation>
auto sparse_gp_from_covariance_and_mean(
    CovFunc &&covariance_function, MeanFunc &&mean_function,
    GrouperFunction &&grouper_function, InducingPointStrategy &&strategy,
    const std::string &model_name,
    QRImplementation qr __attribute__((unused)) = DenseQRImplementation{}) {
  return SparseStructuredGaussianProcessRegression<
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
auto sparse_gp_from_covariance_and_mean(
    CovFunc &&covariance_function, MeanFunc &&mean_function,
    GrouperFunction &&grouper_function, const std::string &model_name,
    QRImplementation qr = DenseQRImplementation{}) {
  return sparse_gp_from_covariance_and_mean(
      std::forward<CovFunc>(covariance_function),
      std::forward<MeanFunc>(mean_function),
      std::forward<GrouperFunction>(grouper_function),
      StateSpaceInducingPointStrategy(), model_name, qr);
};

template <typename CovFunc, typename GrouperFunction,
          typename InducingPointStrategy,
          typename QRImplementation = DenseQRImplementation>
auto sparse_gp_from_covariance(CovFunc &&covariance_function,
                               GrouperFunction &&grouper_function,
                               InducingPointStrategy &&strategy,
                               const std::string &model_name,
                               QRImplementation qr = DenseQRImplementation{}) {
  return sparse_gp_from_covariance_and_mean<
      CovFunc, decltype(ZeroMean()), GrouperFunction, InducingPointStrategy,
      QRImplementation>(std::forward<CovFunc>(covariance_function), ZeroMean(),
                        std::forward<GrouperFunction>(grouper_function),
                        std::forward<InducingPointStrategy>(strategy),
                        model_name, qr);
};

template <typename CovFunc, typename GrouperFunction,
          typename QRImplementation = DenseQRImplementation>
auto sparse_gp_from_covariance(CovFunc covariance_function,
                               GrouperFunction grouper_function,
                               const std::string &model_name,
                               QRImplementation qr = DenseQRImplementation{}) {
  return sparse_gp_from_covariance_and_mean<
      CovFunc, decltype(ZeroMean()), GrouperFunction,
      decltype(StateSpaceInducingPointStrategy()), QRImplementation>(
      std::forward<CovFunc>(covariance_function), ZeroMean(),
      std::forward<GrouperFunction>(grouper_function),
      StateSpaceInducingPointStrategy(), model_name, qr);
};

} // namespace albatross

#endif /* INCLUDE_ALBATROSS_MODELS_SPARSE_GP_H_ */
