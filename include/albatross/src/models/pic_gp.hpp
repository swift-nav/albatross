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
  // std::vector<Eigen::MatrixXd> covariance_W;
  std::vector<Eigen::MatrixXd> covariance_Y;
  Eigen::MatrixXd Z;
  BlockDiagonalLDLT A_ldlt;
  GroupMap measurement_groups;
  Eigen::VectorXd information;
  Eigen::Index numerical_rank;

  // debug stuff
  // std::vector<Eigen::MatrixXd> covariance_Ynot;
  // Eigen::SerializableLDLT K_PITC_ldlt;
  std::vector<Eigen::SparseMatrix<double>> cols_Bs;
  // std::vector<Eigen::SparseMatrix<double>> cols_Cs;

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
      const std::vector<Eigen::SparseMatrix<double>> cols_Bs_)
      : train_features(features_), inducing_features(inducing_features_),
        train_covariance(train_covariance_), sigma_R(sigma_R_), P(P_),
        mean_w(mean_w_), W(W_), covariance_Y(covariance_Y_), Z(Z_),
        A_ldlt(A_ldlt_), measurement_groups(measurement_groups_),
        information(information_), numerical_rank(numerical_rank_),
        cols_Bs(cols_Bs_) {}

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
            numerical_rank == other.numerical_rank &&
            // Debug stuff
            cols_Bs == other.cols_Bs);
  }
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
 * to improve numerical stability.  Here we use an approach based off the QR
 * decomposition which is described in
 *
 *     Stable and Efficient Gaussian Process Calculations
 *     http://www.jmlr.org/papers/volume10/foster09a/foster09a.pdf
 *
 * A more detailed (but more likely to be out of date) description of
 * the details can be found on the albatross documentation.  A short
 * description follows.  It starts by setting up the Sparse Gaussian process
 * covariances
 *
 *   [f|u] ~ N(K_fu K_uu^-1 u, K_ff - Q_ff)
 *
 * We then set,
 *
 *   A = K_ff - Q_ff
 *     = K_ff - K_fu K_uu^-1 K_uf
 *
 * which can be thought of as the covariance in the training data which
 * is not be explained by the inducing points.  The fundamental
 * assumption in these sparse Gaussian processes is that A is sparse, in
 * this case block diagonal.
 *
 * We then build a matrix B and use its QR decomposition (with pivoting P)
 *
 *   B = |A^-1/2 K_fu| = |Q_1| R P^T
 *       |K_uu^{T/2} |   |Q_2|
 *
 * After which we can get the information vector (see _fit_impl)
 *
 *   v = (K_uu + K_uf A^-1 K_fu)^-1 K_uf A^-1 y
 *     = (B^T B) B^T A^-1/2 y
 *     = P R^-1 Q_1^T A^-1/2 y
 *
 * and can make predictions for new locations (see _predict_impl),
 *
 *   [f*|f=y] ~ N(K_*u S K_uf A^-1 y, K_** - Q_** + K_*u S K_u*)
 *            ~ N(m, C)
 *
 *  where we have
 *
 *    m = K_*u S K_uf A^-1 y
 *      = K_*u v
 *
 *  and
 *
 *    C = K_** - Q_** + K_*u S K_u*
 *
 *  using
 *
 *    Q_** = K_*u K_uu^-1 K_u*
 *         = (K_uu^{-1/2}  K_u*)^T (K_uu^{-1/2}  K_u*)
 *         = Q_sqrt^T Q_sqrt
 *  and
 *
 *    K_*u S K_u* = K_*u (K_uu + K_uf A^-1 K_fu)^-1   K_u*
 *                = K_*u (B^T B)^-1 K_u*
 *                = K_*u (P R R^T P^T)^-1 K_u*
 *                = (P R^-T K_u*)^T (P R^-T K_u*)
 *                = S_sqrt^T S_sqrt
 */
template <typename CovFunc, typename MeanFunc, typename GrouperFunction,
          typename InducingPointStrategy,
          typename QRImplementation = DenseQRImplementation>
class PICGaussianProcessRegression
    : public GaussianProcessBase<CovFunc, MeanFunc,
                                 PICGaussianProcessRegression<
                                     CovFunc, MeanFunc, GrouperFunction,
                                     InducingPointStrategy, QRImplementation>> {
  InducingPointStrategy inducing_point_strategy_;

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

  template <typename FeatureType, typename InducingPointFeatureType>
  auto
  _update_impl(const Fit<PICGPFit<GrouperFunction, InducingPointFeatureType,
                                  FeatureType>> &old_fit,
               const std::vector<FeatureType> &features,
               const MarginalDistribution &targets) const {

    BlockDiagonalLDLT A_ldlt;
    Eigen::SerializableLDLT K_uu_ldlt;
    Eigen::MatrixXd K_fu;
    Eigen::VectorXd y;
    Eigen::SerializableLDLT K_PITC_ldlt;
    compute_internal_components(old_fit.train_features, features, targets,
                                &A_ldlt, &K_uu_ldlt, &K_fu, &y, &K_PITC_ldlt);

    const Eigen::Index n_old = old_fit.sigma_R.rows();
    const Eigen::Index n_new = A_ldlt.rows();
    const Eigen::Index k = old_fit.sigma_R.cols();
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(n_old + n_new, k);

    ALBATROSS_ASSERT(n_old == k);

    // Form:
    //   B = |R_old P_old^T| = |Q_1| R P^T
    //       |A^{-1/2} K_fu|   |Q_2|
    for (Eigen::Index i = 0; i < old_fit.permutation_indices.size(); ++i) {
      const Eigen::Index &pi = old_fit.permutation_indices.coeff(i);
      B.col(pi).topRows(i + 1) = old_fit.sigma_R.col(i).topRows(i + 1);
    }
    B.bottomRows(n_new) = A_ldlt.sqrt_solve(K_fu);
    const auto B_qr = QRImplementation::compute(B, Base::threads_.get());

    // Form:
    //   y_aug = |R_old P_old^T v_old|
    //           |A^{-1/2} y         |
    ALBATROSS_ASSERT(old_fit.information.size() == n_old);
    Eigen::VectorXd y_augmented(n_old + n_new);
    for (Eigen::Index i = 0; i < old_fit.permutation_indices.size(); ++i) {
      y_augmented[i] =
          old_fit.information[old_fit.permutation_indices.coeff(i)];
    }
    y_augmented.topRows(n_old) =
        old_fit.sigma_R.template triangularView<Eigen::Upper>() *
        y_augmented.topRows(n_old);

    y_augmented.bottomRows(n_new) = A_ldlt.sqrt_solve(y, Base::threads_.get());
    const Eigen::VectorXd v = B_qr->solve(y_augmented);

    Eigen::MatrixXd R = get_R(*B_qr);
    if (B_qr->rank() < B_qr->cols()) {
      // Inflate the diagonal of R in an attempt to avoid singularity
      R.diagonal() +=
          Eigen::VectorXd::Constant(B_qr->cols(), details::cSparseRNugget);
    }
    using FitType =
        Fit<PICGPFit<GrouperFunction, InducingPointFeatureType, FeatureType>>;
    return FitType(old_fit.train_features, old_fit.train_covariance, R,
                   get_P(*B_qr), v, B_qr->rank());
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
    // std::cerr << "features: " << features.size() << std::endl;
    // std::cerr << "u: " << u.size() << std::endl;

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
        // get_R(*B_qr).template triangularView<Eigen::Upper>().transpose().solve(
        //     get_P(*B_qr).transpose() * A_ldlt.solve(K_fu).transpose());
    // std::cout << "Z (" << Z.rows() << "x" << Z.cols() << "):\n"
    //           << Z << std::endl;

    Eigen::VectorXd y_augmented = Eigen::VectorXd::Zero(B_qr->rows());
    y_augmented.topRows(y.size()) = A_ldlt.sqrt_solve(y, Base::threads_.get());
    const Eigen::VectorXd v = B_qr->solve(y_augmented);

    using InducingPointFeatureType = typename std::decay<decltype(u[0])>::type;

    const Eigen::MatrixXd W =
      K_uu_ldlt.solve((A_ldlt.solve(K_fu) - Z.transpose() * Z * K_fu).transpose());
    // std::cout << "W (" << W.rows() << "x" << W.cols() << "):\n"
    //           << W.format(Eigen::FullPrecision) << std::endl;

    // const Eigen::MatrixXd W2 =
    //     K_uu_ldlt.solve(K_PITC_ldlt.solve(K_fu).transpose());
    // std::cout << "W2 (" << W2.rows() << "x" << W2.cols() << "):\n"
    //           << W2.format(Eigen::FullPrecision) << std::endl;

    // TODO(@peddie): a lot of this can be batched
    std::vector<Eigen::MatrixXd> covariance_Y(measurement_groups.size());
    std::vector<Eigen::VectorXd> mean_w(measurement_groups.size());
    std::vector<Eigen::SparseMatrix<double>> cols_Bs(measurement_groups.size());
    const auto block_start_indices = A_ldlt.block_to_row_map();

    const auto precompute_block = [&A_ldlt, &features, &y, &K_fu, &v,
                                   &K_uu_ldlt, &mean_w, &covariance_Y,
                                   &cols_Bs](std::size_t block_number,
                                             Eigen::Index start_row) -> void {
      // const std::size_t block_number = block_start.first;
      // const Eigen::Index start_row = block_start.second;
      const Eigen::Index block_size = A_ldlt.blocks[block_number].rows();
      // K_fu is already computed, so we can form K_uB and K_uA by
      // appropriate use of sparse indexing matrices and avoid an O(N
      // M) operation.
      //
      // This nonsense would be a lot more straightforward with Eigen
      // 3.4's slicing and indexing API.
      Eigen::SparseMatrix<double> cols_B(cast::to_index(features.size()), block_size);
      cols_B.reserve(block_size);
      Eigen::SparseMatrix<double> cols_C(cast::to_index(features.size()),
                                         cast::to_index(features.size()) - block_size);
      cols_C.reserve(cast::to_index(features.size()) - block_size);

      for (Eigen::Index i = 0; i < block_size; ++i) {
        cols_B.insert(start_row + i, i) = 1.;
      }
      cols_B.makeCompressed();

      // std::cout << "block " << block_number << " -- start_row: " << start_row
      //           << "; block_size: " << block_size << std::endl;
      // std::cout << "cols_B (" << cols_B.rows() << "x" << cols_B.cols() << "):\n"
      //           << Eigen::MatrixXd(cols_B) << std::endl;

      for (Eigen::Index i = 0, j = 0; i < cast::to_index(features.size());) {
        if (i == start_row) {
          i += block_size;
          continue;
        }
        cols_C.insert(i, j) = 1.;
        i++;
        j++;
      }
      cols_C.makeCompressed();

      Eigen::Index col = 0;
      for (Eigen::Index k = 0; k < cols_C.outerSize(); ++k) {
        for (decltype(cols_C)::InnerIterator it(cols_C, k); it; ++it) {
          assert(it.col() == col++);
        }
      }

      cols_Bs[block_number] = cols_B;

      // std::cout << "cols_C (" << cols_C.rows() << "x" << cols_C.cols() << "):\n"
      //           << Eigen::MatrixXd(cols_C) << std::endl;
      // v_b \in R^b = A_BB^-1 (y_b - K_Bu v)
      Eigen::MatrixXd ydiff_b = cols_B.transpose() * (y - K_fu * v);
      // std::cerr << "ydiff_b: " << ydiff_b.rows() << " x " << ydiff_b.cols()
      //           << std::endl;
      // std::cerr << "cols_B: " << cols_B.rows() << " x " << cols_B.cols()
      //           << std::endl;
      // std::cerr << "A_ldlt: " << A_ldlt.rows() << " x " << A_ldlt.cols()
      //           << " in " << A_ldlt.blocks.size() << " blocks: { ";
      // for (const auto &block : A_ldlt.blocks) {
      //   std::cerr << block.rows() << ", ";
      // }
      // std::cerr << " }" << std::endl;
      const Eigen::MatrixXd mean_w_full =
          A_ldlt.blocks[block_number].solve(ydiff_b);

      // if (A_ldlt.blocks.size() > 1) {
      //   // std::cout << "K_fu (" << K_fu.rows() << "x" << K_fu.cols() <<
      //   // "):\n"
      //   //           << K_fu << std::endl;
      //   const Eigen::MatrixXd KufC = K_fu.transpose() * cols_C;
      //   // std::cout << "KufC (" << KufC.rows() << "x" << KufC.cols() << "):\n"
      //   //           << KufC << std::endl;
      //   // const Eigen::MatrixXd ZC = Z * cols_C;
      //   // // std::cout << "ZC.transpose() (" << ZC.transpose().rows() << "x"
      //   // //           << ZC.transpose().cols() << "):\n"
      //   // //           << ZC.transpose() << std::endl;

      //   // const Eigen::MatrixXd KuZT = KufC * ZC.transpose();
      //   // const Eigen::MatrixXd KuZTZ = KuZT * Z;
      //   // const Eigen::MatrixXd KuZTZB = -KuZTZ * cols_B;
      //   // covariance_W[block_number] = K_uu_ldlt.solve(KuZTZB);
      //   // std::cout << "covariance_W[" << block_number << "]:\n"
      //   //           << covariance_W[block_number].format(Eigen::FullPrecision)
      //   //           << std::endl;
      //   covariance_Ynot[block_number] = K_uu_ldlt.solve(KufC);
      //   // covariance_W.emplace_back(K_uu_ldlt.solve(
      //   //     (-(K_fu.transpose() * cols_C) * Z.transpose() * Z) *
      //   //     cols_B));
      // }

      // std::cout << "mean_w_full (" << mean_w_full.rows() << "x"
      //           << mean_w_full.cols() << "):\n"
      //           << mean_w_full << std::endl;
      // const Eigen::MatrixXd mean_w_block = cols_B.transpose() *
      // mean_w_full; std::cout << "mean_w_block (" << mean_w_block.rows()
      // << "x"
      //           << mean_w_block.cols() << "):\n"
      //           << mean_w_block << std::endl;
      // mean_w.emplace_back(cols_B.transpose() *
      //                     A_ldlt.blocks[block_number].solve(ydiff_b));
      mean_w[block_number] = mean_w_full;
      // Y \in R^(u x b) = K_uu^-1 K_uB
      covariance_Y[block_number] = K_uu_ldlt.solve(K_fu.transpose() * cols_B);
      // std::cout << "covariance_Y[" << block_number << "]:\n"
      //           << covariance_Y[block_number].format(Eigen::FullPrecision)
      //           << std::endl;
      // W \in R^(u x b) = K_uu^-1 K_uC S_CB^-1
      //                 = K_uu^-1 K_uC (A^-1 - Z^T Z)
      //                 [A^-1 is block diagonal; C and B are disjoint]
      //                 = K_uu^-1 K_uC (- Z^T Z)
    };

    // for (const auto &block_start : block_start_indices) {
    //   const std::size_t block_number = block_start.first;
    //   const Eigen::Index start_row = block_start.second;
    //   const Eigen::Index block_size = A_ldlt.blocks[block_number].rows();
    //   // K_fu is already computed, so we can form K_uB and K_uA by
    //   // appropriate use of sparse indexing matrices and avoid an O(N
    //   // M) operation.
    //   //
    //   // This nonsense would be a lot more straightforward with Eigen
    //   // 3.4's slicing and indexing API.
    //   Eigen::SparseMatrix<double> cols_B(features.size(), block_size);
    //   cols_B.reserve(block_size);
    //   Eigen::SparseMatrix<double> cols_C(features.size(),
    //                                      features.size() - block_size);
    //   cols_C.reserve(features.size() - block_size);

    //   for (Eigen::Index i = 0; i < block_size; ++i) {
    //     cols_B.insert(start_row + i, i) = 1.;
    //   }
    //   cols_B.makeCompressed();

    //   // std::cout << "cols_B (" << cols_B.rows() << "x" << cols_B.cols() <<
    //   "):\n"
    //   //           << Eigen::MatrixXd(cols_B) << std::endl;

    //   // std::cout << "start_row: " << start_row << "; block_size: " <<
    //   block_size
    //   //           << std::endl;
    //   for (Eigen::Index i = 0, j = 0; i < features.size();) {
    //     if (i == start_row) {
    //       i += block_size;
    //       continue;
    //     }
    //     cols_C.insert(i, j) = 1.;
    //     i++;
    //     j++;
    //   }
    //   cols_C.makeCompressed();

    //   Eigen::Index col = 0;
    //   for (Eigen::Index k = 0; k < cols_C.outerSize(); ++k) {
    //     for (decltype(cols_C)::InnerIterator it(cols_C, k); it; ++it) {
    //       assert(it.col() == col++);
    //     }
    //   }
    //   // std::cout << "cols_C (" << cols_C.rows() << "x" << cols_C.cols() <<
    //   "):\n"
    //   //           << Eigen::MatrixXd(cols_C) << std::endl;
    //   // v_b \in R^b = A_BB^-1 (y_b - K_Bu v)
    //   Eigen::MatrixXd ydiff_b = cols_B.transpose() * (y - K_fu * v);
    //   // std::cerr << "ydiff_b: " << ydiff_b.rows() << " x " <<
    //   ydiff_b.cols()
    //   //           << std::endl;
    //   // std::cerr << "cols_B: " << cols_B.rows() << " x " << cols_B.cols()
    //   //           << std::endl;
    //   // std::cerr << "A_ldlt: " << A_ldlt.rows() << " x " << A_ldlt.cols()
    //   //           << " in " << A_ldlt.blocks.size() << " blocks: { ";
    //   // for (const auto &block : A_ldlt.blocks) {
    //   //   std::cerr << block.rows() << ", ";
    //   // }
    //   // std::cerr << " }" << std::endl;
    //   const Eigen::MatrixXd mean_w_full =
    //       A_ldlt.blocks[block_number].solve(ydiff_b);
    //   // std::cout << "mean_w_full (" << mean_w_full.rows() << "x"
    //   //           << mean_w_full.cols() << "):\n"
    //   //           << mean_w_full << std::endl;
    //   // const Eigen::MatrixXd mean_w_block = cols_B.transpose() *
    //   mean_w_full;
    //   // std::cout << "mean_w_block (" << mean_w_block.rows() << "x"
    //   //           << mean_w_block.cols() << "):\n"
    //   //           << mean_w_block << std::endl;
    //   // mean_w.emplace_back(cols_B.transpose() *
    //   //                     A_ldlt.blocks[block_number].solve(ydiff_b));
    //   mean_w.push_back(mean_w_full);
    //   // Y \in R^(u x b) = K_uu^-1 K_uB
    //   covariance_Y.emplace_back(K_uu_ldlt.solve(K_fu.transpose() * cols_B));
    //   // W \in R^(u x b) = K_uu^-1 K_uC S_CB^-1
    //   //                 = K_uu^-1 K_uC (A^-1 - Z^T Z)
    //   //                 [A^-1 is block diagonal; C and B are disjoint]
    //   //                 = K_uu^-1 K_uC (- Z^T Z)

    //   if (A_ldlt.blocks.size() > 1) {
    //     // std::cout << "K_fu (" << K_fu.rows() << "x" << K_fu.cols() <<
    //     "):\n"
    //     //           << K_fu << std::endl;
    //     const Eigen::MatrixXd KufC = K_fu.transpose() * cols_C;
    //     // std::cout << "KufC (" << KufC.rows() << "x" << KufC.cols() <<
    //     "):\n"
    //     //           << KufC << std::endl;
    //     const Eigen::MatrixXd ZC = Z * cols_C;
    //     // std::cout << "ZC.transpose() (" << ZC.transpose().rows() << "x"
    //     //           << ZC.transpose().cols() << "):\n"
    //     //           << ZC.transpose() << std::endl;

    //     const Eigen::MatrixXd KuZT = KufC * ZC.transpose();
    //     const Eigen::MatrixXd KuZTZ = KuZT * Z;
    //     const Eigen::MatrixXd KuZTZB = -KuZTZ * cols_B;
    //     covariance_W.emplace_back(K_uu_ldlt.solve(KuZTZB));
    //     // covariance_W.emplace_back(K_uu_ldlt.solve(
    //     //     (-(K_fu.transpose() * cols_C) * Z.transpose() * Z) * cols_B));
    //   }
    // }

    apply_map(block_start_indices, precompute_block, Base::threads_.get());

    using FitType =
        Fit<PICGPFit<GrouperFunction, InducingPointFeatureType, FeatureType>>;
    return FitType(features, u, K_uu_ldlt, get_R(*B_qr), get_P(*B_qr), mean_w,
                   W, covariance_Y, Z, A_ldlt, measurement_groups, v,
                   B_qr->rank(), cols_Bs);
  }

  template <typename FeatureType>
  auto fit_from_prediction(const std::vector<FeatureType> &new_inducing_points,
                           const JointDistribution &prediction_) const {
    FitModel<PICGaussianProcessRegression,
             Fit<PICGPFit<GrouperFunction, FeatureType, FeatureType>>>
        output(*this,
               Fit<PICGPFit<GrouperFunction, FeatureType, FeatureType>>());
    Fit<PICGPFit<GrouperFunction, FeatureType, FeatureType>> &new_fit =
        output.get_fit();

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

    new_fit.P = get_P(*B_qr);
    new_fit.sigma_R = get_R(*B_qr);
    new_fit.numerical_rank = B_qr->rank();

    return output;
  }

  // This is included to allow the SparseGP to be compatible with fits
  // generated using a standard GP.
  using Base::_predict_impl;

  template <typename FeatureType, typename FitFeaturetype>
  Eigen::VectorXd _predict_impl(
      const std::vector<FeatureType> &features,
      const Fit<PICGPFit<GrouperFunction, FitFeaturetype, FeatureType>>
          &sparse_gp_fit,
      PredictTypeIdentity<Eigen::VectorXd> &&) const {
    const auto cross_cov =
        this->covariance_function_(sparse_gp_fit.train_features, features);
    Eigen::VectorXd mean =
        gp_mean_prediction(cross_cov, sparse_gp_fit.information);
    this->mean_function_.add_to(features, &mean);
    return mean;
  }

  template <typename FeatureType, typename FitFeaturetype>
  MarginalDistribution _predict_impl(
      const std::vector<FeatureType> &features,
      const Fit<PICGPFit<GrouperFunction, FitFeaturetype, FeatureType>>
          &sparse_gp_fit,
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

    const Eigen::MatrixXd S_sqrt =
        sqrt_solve(sparse_gp_fit.sigma_R, sparse_gp_fit.P.indices(), cross_cov);
    const Eigen::VectorXd S_diag =
        S_sqrt.cwiseProduct(S_sqrt).array().colwise().sum();
    marginal_variance += S_diag;

    return MarginalDistribution(mean, marginal_variance);
  }

  // template <typename FeatureType, typename FitType>
  // auto find_group(const FitType &fit, const FeatureType &feature) const {
  //   const auto group = fit.measurement_groups.find(
  //       independent_group_function_(without_measurement(feature)));
  //   if (group == fit.measurement_groups.end()) {
  //     std::cerr << "Group mapping failure for feature '"
  //               << without_measurement(feature) << "' (group index '"
  //               << independent_group_function_(without_measurement(feature))
  //               << "')!" << std::endl;
  //     assert(group != fit.measurement_groups.end() &&
  //            "TODO(@peddie): the group function in a PIC GP model must cover
  //            " "the entire feature domain in any fit.");
  //   }
  //   return group;
  // }
  // template <typename FitFeatureType>
  // using InducingFeatureType =
  //     typename std::decay<typename std::result_of<InducingPointStrategy(
  //         const CovFunc &,
  //         const std::vector<FitFeatureType> &)>::type::value_type>::type;

  // typename std::decay<decltype(InducingPointStrategy>(
  //     std::declval<CovFunc>(), std::declval<FitFeatureType>()))>::type;

  // template <typename FitFeatureType>
  // using InducingFeatureType =
  //     typename std::decay<typename decltype(inducing_point_strategy_(
  //         std::declval<const CovFunc &>(),
  //         std::declval<const std::vector<FitFeatureType>
  //         &>()))::value_type>:: type;

  template <typename FeatureType, typename InducingFeatureType,
            typename FitFeatureType>
  JointDistribution _predict_impl(
      const std::vector<FeatureType> &features,
      const Fit<PICGPFit<
          GrouperFunction,
          InducingFeatureType, // <FitFeatureType>,
                               // typename std::result_of<InducingPointStrategy(
                               //     const CovFunc &, const
                               //     std::vector<FitFeatureType> &)>::type,
          FitFeatureType>> &sparse_gp_fit,
      // const Fit<PICGPFit<GrouperFunction,
      // InducingFeatureType<FitFeatureType>,
      //                    FitFeatureType>> &sparse_gp_fit,
      PredictTypeIdentity<JointDistribution> &&) const {

    const auto find_group = [this, &sparse_gp_fit](const auto &feature) {
      const auto group = sparse_gp_fit.measurement_groups.find(
          independent_group_function_(without_measurement(feature)));
      if (group == sparse_gp_fit.measurement_groups.end()) {
        std::cerr << "Group mapping failure for feature '"
                  << without_measurement(feature) << "' (group index '"
                  << independent_group_function_(without_measurement(feature))
                  << "')!" << std::endl;
        assert(group != sparse_gp_fit.measurement_groups.end() &&
               "TODO(@peddie): the group function in a PIC GP model must "
               "cover the entire feature domain in any fit.");
      }
      return group;
    };
    // using CalculatedInducingFeatureType =
    //     typename std::decay<typename std::result_of<InducingPointStrategy(
    //         const CovFunc &,
    //         const std::vector<FeatureType> &)>::type::value_type>::type;
    // static_assert(
    //     std::is_same<CalculatedInducingFeatureType,
    //                  InducingFeatureType>::value &&
    //     "A fitted PIC model must be able to compute the covariance between
    //     its " "inducing point feature type and the feature type to be
    //     predicted.");
    // K_up
    const Eigen::MatrixXd cross_cov =
        this->covariance_function_(sparse_gp_fit.inducing_features, features);
    const Eigen::MatrixXd prior_cov = this->covariance_function_(features);

    Eigen::MatrixXd WV = Eigen::MatrixXd::Zero(
        cast::to_index(sparse_gp_fit.inducing_features.size()), cast::to_index(features.size()));

    Eigen::VectorXd mean_correction = Eigen::VectorXd::Zero(cast::to_index(features.size()));
    // std::cout << "mean_correction before (" << mean_correction.size() << "): "
    //           << mean_correction.transpose().format(Eigen::FullPrecision)
    //           << std::endl;
    std::vector<std::size_t> feature_to_block;
    for (Eigen::Index j = 0; j < cast::to_index(features.size()); ++j) {
      const auto group = find_group(features[cast::to_size(j)]);
      // const auto group = sparse_gp_fit.measurement_groups.find(
      //     independent_group_function_(without_measurement(features[j])));
      // assert(group != sparse_gp_fit.measurement_groups.end() &&
      //        "TODO(@peddie): the group function in a PIC GP model must
      //        cover " "the entire feature domain in any fit.");
      feature_to_block.push_back(
          cast::to_size(std::distance(sparse_gp_fit.measurement_groups.begin(), group)));
      // if (sparse_gp_fit.A_ldlt.blocks.size() > 1) {
        const std::vector<FeatureType> fvec = {features[cast::to_size(j)]};

        const Eigen::VectorXd features_cov =
            this->covariance_function_(group->second.dataset.features, fvec);
        // std::cout << "Feature " << j << "(" <<
        // without_measurement(features[j])
        //           << ") in group '" << group->first << "' ("
        //           << group->second.block_size << ": "
        //           << group->second.initial_row << " -> "
        //           << group->second.initial_row + group->second.block_size <<
        //           ")"
        //           << std::endl;
        // std::cout << "inducing_points: "
        //           << sparse_gp_fit.inducing_features.size() << std::endl;
        // std::cout << "cross_cov (" << cross_cov.rows() << "x"
        //           << cross_cov.cols() << "):\n"
        //           << cross_cov << std::endl;
        // std::cout << "features_cov (" << features_cov.rows() << "x"
        //           << features_cov.cols() << "):\n"
        //           << features_cov << std::endl;
        // std::cout
        //     << "sparse_gp_fit.covariance_Y[group->second.block_index] ("
        //     << sparse_gp_fit.covariance_Y[group->second.block_index].rows()
        //     << "x"
        //     << sparse_gp_fit.covariance_Y[group->second.block_index].cols()
        //     << "):\n"
        //     << sparse_gp_fit.covariance_Y[group->second.block_index]
        //     << std::endl;
        const Eigen::VectorXd kpuy =
            cross_cov.transpose().row(j) *
            sparse_gp_fit.covariance_Y[cast::to_size(group->second.block_index)];
        // std::cout << "kpuy (" << kpuy.rows() << "x" << kpuy.cols() << "):\n"
        //           << kpuy << std::endl;
        const Eigen::VectorXd Vbp = features_cov - kpuy;
        mean_correction[j] =
            Vbp.dot(sparse_gp_fit.mean_w[cast::to_size(group->second.block_index)]);
        const Eigen::VectorXd wvj =
            sparse_gp_fit.W * sparse_gp_fit.cols_Bs[feature_to_block[cast::to_size(j)]] * Vbp;
        WV.col(j) = wvj;

        // std::cout << "Vbp[" << j << "] (" << Vbp.size()
        //           << "): " << Vbp.transpose().format(Eigen::FullPrecision)
        //           << std::endl;
        // std::cout << "sparse_gp_fit.mean_w[" << group->second.block_index
        //           << "] ("
        //           << sparse_gp_fit.mean_w[group->second.block_index].size()
        //           << "): "
        //           << sparse_gp_fit.mean_w[group->second.block_index]
        //                  .transpose()
        //                  .format(Eigen::FullPrecision)
        //           << std::endl;

        // WV.col(j) =
        //     sparse_gp_fit.covariance_W[group->second.block_index] *
        //     (this->covariance_function_(group->second.dataset.features, fvec)
        //     -
        //      cross_cov.transpose() *
        //          sparse_gp_fit.covariance_Y[group->second.block_index]);
      // }
    }

    // std::cout << "mean_correction after (" << mean_correction.size() << "): "
    //           << mean_correction.transpose().format(Eigen::FullPrecision)
    //           << std::endl;
    // std::cout << "WV (" << WV.rows() << "x" << WV.cols() << "):\n"
    //           << WV << std::endl;

    // std::cout << "mean_correction (" << mean_correction.size()
    //           << "): " << mean_correction.transpose() << std::endl;

    Eigen::MatrixXd VSV(features.size(), features.size());

    for (std::size_t row = 0; row < features.size(); ++row) {
      for (std::size_t col = 0; col <= row; ++col) {
        Eigen::Index row_index = cast::to_index(row);
        Eigen::Index col_index = cast::to_index(col);
        const auto row_group = find_group(features[row]);
        const auto column_group = find_group(features[col]);
        // const auto row_group = sparse_gp_fit.measurement_groups.find(
        //     independent_group_function_(without_measurement(features[row])));
        // assert(row_group != sparse_gp_fit.measurement_groups.end() &&
        //        "TODO(@peddie): the group function in a PIC GP model must
        //        cover " "the entire feature domain in any fit.");

        // const auto column_group = sparse_gp_fit.measurement_groups.find(
        //     independent_group_function_(without_measurement(features[col])));
        // assert(column_group != sparse_gp_fit.measurement_groups.end() &&
        //        "TODO(@peddie): the group function in a PIC GP model must
        //        cover " "the entire feature domain in any fit.");

        // TODO(@peddie): these are K, not V!
        const Eigen::VectorXd Q_row_p =
            cross_cov.transpose().row(row_index) *
            sparse_gp_fit.covariance_Y[cast::to_size(row_group->second.block_index)];
        const std::vector<FeatureType> row_fvec = {features[row]};
        const Eigen::VectorXd V_row_p =
            this->covariance_function_(row_group->second.dataset.features,
                                       row_fvec) -
            Q_row_p;
        const Eigen::VectorXd Q_column_p =
            cross_cov.transpose().row(col_index) *
            sparse_gp_fit.covariance_Y[cast::to_size(column_group->second.block_index)];
        const std::vector<FeatureType> column_fvec = {features[col]};
        const Eigen::VectorXd V_column_p =
            this->covariance_function_(column_group->second.dataset.features,
                                       column_fvec) -
            Q_column_p;

        VSV(row_index, col_index) = 0.;

        assert(row < feature_to_block.size());
        assert(col < feature_to_block.size());
        if (feature_to_block[row] == feature_to_block[col]) {
          VSV(row_index, col_index) =
              sparse_gp_fit.A_ldlt.blocks[feature_to_block[row]]
                  .sqrt_solve(V_row_p)
                  .col(0)
                  .dot(sparse_gp_fit.A_ldlt.blocks[feature_to_block[col]]
                           .sqrt_solve(V_column_p)
                           .col(0));
          // std::cout << "VSV(" << row << ", " << col << ") same block ("
          //           << feature_to_block[row] << "):\n"
          //           << VSV(row, col) << std::endl;
        }

        const Eigen::MatrixXd rowblock = sparse_gp_fit.Z.block(
            0, row_group->second.initial_row, sparse_gp_fit.Z.rows(),
            row_group->second.block_size);
        const Eigen::MatrixXd columnblock = sparse_gp_fit.Z.block(
            0, column_group->second.initial_row, sparse_gp_fit.Z.rows(),
            column_group->second.block_size);

        // std::cout << "VSV(" << row << ", " << col << "):\n"
        //           << "rowblock (" << rowblock.rows() << "x" << rowblock.cols()
        //           << "):\n"
        //           << rowblock << "\ncolblock (" << columnblock.rows() << "x"
        //           << columnblock.cols() << "):\n"
        //           << columnblock << "\nV_row_p (" << V_row_p.size()
        //           << "): " << V_row_p.transpose() << "\nV_column_p ("
        //           << V_column_p.size() << "): " << V_column_p.transpose()
        //           << "\nvalue: "
        //           << (rowblock * V_row_p).dot(columnblock * V_column_p)
        //           << std::endl;

        VSV(row_index, col_index) -= (rowblock * V_row_p).dot(columnblock * V_column_p);
      }
    }

    VSV.triangularView<Eigen::Upper>() = VSV.transpose();

    // std::cout << "VSV (" << VSV.rows() << "x" << VSV.cols() << "):\n"
    //           << VSV << std::endl;
    const Eigen::MatrixXd U = cross_cov.transpose() * WV;

    // std::cout << "U (" << U.rows() << "x" << U.cols() << "):\n"
    //           << U << std::endl;

    const Eigen::MatrixXd S_sqrt =
        sqrt_solve(sparse_gp_fit.sigma_R, sparse_gp_fit.P, cross_cov);

    const Eigen::MatrixXd Q_sqrt =
        sparse_gp_fit.train_covariance.sqrt_solve(cross_cov);

    const Eigen::MatrixXd max_explained = Q_sqrt.transpose() * Q_sqrt;
    const Eigen::MatrixXd unexplained = S_sqrt.transpose() * S_sqrt;

    const Eigen::MatrixXd pitc_covariance =
        prior_cov - max_explained + unexplained;

    // std::cout << "pitc_covariance (" << pitc_covariance.rows() << "x"
    //           << pitc_covariance.cols() << "):\n"
    //           << pitc_covariance << std::endl;

    const Eigen::MatrixXd pic_correction = U + U.transpose() + VSV;

    // std::cout << "pic_correction (" << pic_correction.rows() << "x"
    //           << pic_correction.cols() << "):\n"
    //           << pic_correction << std::endl;

    JointDistribution pred(cross_cov.transpose() * sparse_gp_fit.information +
                               mean_correction,
                           pitc_covariance - pic_correction);

    this->mean_function_.add_to(features, &pred.mean);

    // Debug comparison

    // Eigen::MatrixXd K_pic(sparse_gp_fit.train_features.size(),
    // features.size()); for (std::size_t i = 0; i < features.size(); ++i) {
    //   const auto group = find_group(features[i]);
    //   K_pic.col(i) = sparse_gp_fit.cols_Cs[feature_to_block[i]] *
    //                  cross_cov.transpose() *
    //                  sparse_gp_fit.covariance_Ynot[feature_to_block[i]];
    //   std::cout << "Ynot: K_pic.col(" << i << "): " <<
    //   K_pic.col(i).transpose()
    //             << std::endl;
    //   K_pic.col(i).segment(group->second.initial_row,
    //                        group->second.block_size) =
    //       this->covariance_function_(std::vector<FeatureType>{features[i]},
    //                                  group->second.dataset.features);
    //   std::cout << "K*: K_pic.col(" << i << "): " << K_pic.col(i).transpose()
    //             << std::endl;
    // }

    // std::cout << "K_pic (" << K_pic.rows() << "x" << K_pic.cols() << "):\n"
    //           << K_pic << std::endl;

    // const Eigen::MatrixXd K_pic_pitc =
    //     sparse_gp_fit.K_PITC_ldlt.sqrt_solve(K_pic);
    // JointDistribution alt_pred(cross_cov.transpose() *
    //                                sparse_gp_fit.information,
    //                            prior_cov - K_pic_pitc.transpose() *
    //                            K_pic_pitc);

    // std::cout << "alt covariance (" << alt_pred.covariance.rows() << "x"
    //           << alt_pred.covariance.cols() << "):\n"
    //           << alt_pred.covariance << std::endl;

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
      // measurement_groups->emplace(
      //     std::piecewise_construct, std::forward_as_tuple(pair.first),
      //     std::forward_as_tuple(
      //         begin_row_index, static_cast<Eigen::Index>(pair.second.size()),
      //         block_index,
      //         albatross::RegressionDataset<FeatureType>(
      //             subset(out_of_order_features, pair.second),
      //             out_of_order_targets.subset(pair.second))));
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

    // const Eigen::MatrixXd K_uuuf = K_uu_ldlt->sqrt_solve(K_fu->transpose());
    // Eigen::MatrixXd Kfuf = K_uuuf.transpose() * K_uuuf;
    // Eigen::Index row_offset = 0;
    // Eigen::Index col_offset = 0;
    // for (auto &b : A.blocks) {
    //   Kfuf.block(row_offset, col_offset, b.rows(), b.cols()) += b;
    //   row_offset += b.rows();
    //   col_offset += b.cols();
    // }

    // // std::cout << "Kfuf (" << Kfuf.rows() << "x" << Kfuf.cols() << "):\n"
    // //           << Kfuf.format(Eigen::FullPrecision) << std::endl;

    // *K_PITC_ldlt = Kfuf.ldlt();
  }

  Parameter measurement_nugget_;
  Parameter inducing_nugget_;
  // InducingPointStrategy inducing_point_strategy_;
  GrouperFunction independent_group_function_;
};

// rebase_inducing_points takes a Sparse GP which was fit using some set of
// inducing points and creates a new fit relative to new inducing points.
// Note that this will NOT be the equivalent to having fit the model with
// the new inducing points since some information may have been lost in
// the process.
template <typename ModelType, typename GrouperFunction, typename FeatureType,
          typename NewFeatureType>
auto rebase_inducing_points(
    const FitModel<ModelType,
                   Fit<PICGPFit<GrouperFunction, NewFeatureType, FeatureType>>>
        &fit_model,
    const std::vector<NewFeatureType> &new_inducing_points) {
  return fit_model.get_model().fit_from_prediction(
      new_inducing_points, fit_model.predict(new_inducing_points).joint());
}

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
