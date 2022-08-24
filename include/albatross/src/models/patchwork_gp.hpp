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

#ifndef INCLUDE_ALBATROSS_MODELS_PATCHWORK_GP_H_
#define INCLUDE_ALBATROSS_MODELS_PATCHWORK_GP_H_

/*
 * Patchwork Gaussian Process is based off the paper:
 *
 *   Chiwoo Park and Daniel Apley. 2018. Patchwork Kriging for large-scale
 * Gaussian process regression. J. Mach. Learn. Res. 19, 1 (January 2018),
 * 269-311.
 *
 *   http://www.jmlr.org/papers/volume19/17-042/17-042.pdf
 *
 * This technique works by splitting a problem (which may contain too much
 * data for a single GP) into tractible sized domains which are then stitched
 * together by forcing their predictions to be equal along boundaries.
 *
 * The implementation requires that you define a class which contains three
 * methods:
 *
 * grouper :
 *
 *     GroupKey grouper(const FeatureType &f) const
 *
 *   This method should take a FeatureType and return which group it belongs to.
 *
 * boundary:
 *
 *     std::vector<BoundaryFeature> boundary(const GroupKey &x, const
 * GroupKey&y) const
 *
 *   This method should take two groups and return the features which represent
 * the boundary between two groups that will be constrained to be equal.
 *
 * nearest_group:
 *
 *   GroupKey nearest_group(const std::vector<GroupKey> &groups, const GroupKey
 * &query) const
 *
 *   This method is used during prediction and takes a vector of all the
 * available groups and a query group.  If the query exists it should simply
 * return the query.  But if the query group doesn't exist is should return the
 * nearest group.
 *
 */

namespace albatross {

template <typename FitModelType, typename GroupKey,
          typename BoundaryFeatureType>
struct PatchworkGPFit {};

template <typename ModelType, typename FitType, typename GroupKey,
          typename BoundaryFeatureType>
struct Fit<PatchworkGPFit<FitModel<ModelType, FitType>, GroupKey,
                          BoundaryFeatureType>> {

  using CovarianceRepresentation = decltype(FitType::train_covariance);

  Grouped<GroupKey, FitModel<ModelType, FitType>> fit_models;
  Grouped<GroupKey, Eigen::MatrixXd> information;
  std::vector<BoundaryFeatureType> boundary_features;
  // These matrices make up the block elements in Equation 5
  Grouped<GroupKey, CovarianceRepresentation> C_dd;
  Grouped<GroupKey, Eigen::MatrixXd> C_db;
  Eigen::SerializableLDLT C_bb_ldlt;
  Eigen::SerializableLDLT S_bb_ldlt;

  Fit(){};

  Fit(const Grouped<GroupKey, FitModel<ModelType, FitType>> &fit_models_)
      : fit_models(fit_models_){};

  Fit(const Grouped<GroupKey, FitModel<ModelType, FitType>> &fit_models_,
      const Grouped<GroupKey, Eigen::MatrixXd> &information_,
      const std::vector<BoundaryFeatureType> &boundary_features_,
      const Grouped<GroupKey, CovarianceRepresentation> &C_dd_,
      const Grouped<GroupKey, Eigen::MatrixXd> &C_db_,
      const Eigen::SerializableLDLT &C_bb_ldlt_,
      const Eigen::SerializableLDLT &S_bb_ldlt_)
      : fit_models(fit_models_), information(information_),
        boundary_features(boundary_features_), C_dd(C_dd_), C_db(C_db_),
        C_bb_ldlt(C_bb_ldlt_), S_bb_ldlt(S_bb_ldlt_){};
};

template <typename BoundaryFunction, typename GroupKey>
auto build_boundary_features(const BoundaryFunction &boundary_function,
                             const std::vector<GroupKey> &groups) {
  /*
   * Loop through all combinations of groups (without permutations) and
   * assemble the boundary features.
   */
  using BoundarySubFeatureType =
      typename invoke_result<BoundaryFunction, GroupKey,
                             GroupKey>::type::value_type;
  using BoundaryFeatureType = BoundaryFeature<GroupKey, BoundarySubFeatureType>;

  std::vector<BoundaryFeatureType> boundary_features;
  for (std::size_t i = 0; i < groups.size(); ++i) {
    for (std::size_t j = i + 1; j < groups.size(); ++j) {
      const auto next_boundary_features = as_boundary_features(
          groups[i], groups[j], boundary_function(groups[i], groups[j]));
      if (next_boundary_features.size() > 0) {
        boundary_features.insert(boundary_features.end(),
                                 next_boundary_features.begin(),
                                 next_boundary_features.end());
      }
    }
  }
  return boundary_features;
}

template <typename GroupKey, typename Solver, int Rows, int Cols>
auto patchwork_solver_from_v(
    const Grouped<GroupKey, Solver> &A,
    const Grouped<GroupKey, Eigen::MatrixXd> &C,
    const Eigen::SerializableLDLT &S,
    const Grouped<GroupKey, Eigen::Matrix<double, Rows, Cols>> &v) {
  // This solves a matrix which takes the form:
  //
  //     (A - C B^-1 C^T)^{-1}
  //
  // Where A is a block diagonal matrix, C and B are dense and
  //
  //     S = (B - C^T A^-1 C)
  //
  // The block structure is formulated using Grouped objects which
  // consist of std::map like key value pairs.  For example, `A`
  // in this case is a map which maps from a GroupKey to the
  // the corresponding solver for that particular block diagonal
  // portion.  Similarly `C` is a map from the group key to the
  // corresponding rows of the dense version of C.
  //
  // This matrix can be solved using the matrix inversion lemma,
  // which is re-iterated in Equation 9:
  //
  //   A^-1 + A^-1 C (B - C^T A^-1 C)^-1 C^T A^-1
  //   A^-1 + A^-1 C S^-1 C^T A^-1
  //
  // Here we break it up into a few steps:
  //
  //   v = A^-1 rhs    (grouped and precomputed)
  //   u = S^-1 C^T v  (dense)
  //   w = A^-1 C U    (grouped)
  //   output = w + v

  // u = S^-1 C^T v
  auto compute_u_block = [&](const Eigen::MatrixXd &C_i, const auto &v_i) {
    return Eigen::MatrixXd(S.solve(C_i.transpose() * v_i));
  };
  const Eigen::MatrixXd u = block_accumulate(C, v, compute_u_block);

  // w = A^-1 C u
  auto compute_Cu_block = [&](const auto &key, const auto &C_i) {
    return Eigen::MatrixXd(C_i * u);
  };
  const auto w = block_diag_solve(A, C.apply(compute_Cu_block));
  // output = w + z
  auto add_z = [&](const auto &key, const auto &w_block) {
    return Eigen::MatrixXd(w_block + v.at(key)).eval();
  };
  return w.apply(add_z);
};

template <typename GroupKey, typename Solver, int Rows, int Cols>
auto patchwork_solver(
    const Grouped<GroupKey, Solver> &A,
    const Grouped<GroupKey, Eigen::MatrixXd> &C,
    const Eigen::SerializableLDLT &S,
    const Grouped<GroupKey, Eigen::Matrix<double, Rows, Cols>> &rhs) {
  // v = A^-1 rhs
  const auto v = block_diag_solve(A, rhs);
  return patchwork_solver_from_v(A, C, S, v);
}

template <typename CovFunc, typename MeanFunc, typename PatchworkFunctions>
class PatchworkGaussianProcess
    : public GaussianProcessBase<
          CovFunc, MeanFunc,
          PatchworkGaussianProcess<CovFunc, MeanFunc, PatchworkFunctions>> {

public:
  using Base = GaussianProcessBase<
      CovFunc, MeanFunc,
      PatchworkGaussianProcess<CovFunc, MeanFunc, PatchworkFunctions>>;

  PatchworkGaussianProcess() : Base(){};
  PatchworkGaussianProcess(CovFunc &covariance_function)
      : Base(covariance_function){};
  PatchworkGaussianProcess(CovFunc &covariance_function,
                           PatchworkFunctions patchwork_functions)
      : Base(covariance_function), patchwork_functions_(patchwork_functions){};

  template <typename FitModelType, typename GroupKey>
  auto
  from_fit_models(const Grouped<GroupKey, FitModelType> &fit_models) const {

    auto boundary_function = [&](const GroupKey &x, const GroupKey &y) {
      return patchwork_functions_.boundary(x, y);
    };

    const auto boundary_features =
        build_boundary_features(boundary_function, fit_models.keys());

    using BoundaryFeatureType = typename std::decay<
        typename decltype(boundary_features)::value_type>::type;
    using PatchworkFitType =
        Fit<PatchworkGPFit<FitModelType, GroupKey, BoundaryFeatureType>>;
    using ReturnType = FitModel<PatchworkGaussianProcess, PatchworkFitType>;

    if (fit_models.size() == 1) {
      return ReturnType(*this, PatchworkFitType(fit_models));
    }

    ALBATROSS_ASSERT(boundary_features.size() > 0);

    // The following variable names are meant to approximately match the
    // notation used in Equation 5 (and the following matrices).  The
    // subscripts had to be changed.  In particular we've used:
    //
    //   d - the data, this is script D in the paper.
    //   b - the boundaries, this is \del in the paper.
    //   f - the prediction locations, this is * in the paper.

    // C_bb is the covariance matrix between all boundaries, it will
    // have a lot of zeros, so it could be decomposed more efficiently,
    // but here is treated as a dense matrix.
    const Eigen::MatrixXd C_bb =
        patchwork_covariance_matrix(boundary_features, boundary_features);
    const auto C_bb_ldlt = C_bb.ldlt();

    // C_dd is the large block diagonal matrix, with one block holding the
    // covariance matrix from each model.  Albatross stores fit
    // Gaussian processes using a pre-decomposed covariance matrix,
    // typically the LDLT, we can use directly in subsequent computations.
    // So C_dd here is actually a map to a "Solver", not an Eigen::MatrixXd.
    //
    // Also worth noting that in the paper they show C_dd as a full dense
    // matrix, but all of the off-diagonal terms are zero so it is indeed
    // a block diagonal matrix.
    auto get_train_covariance = [](const auto &fit_model) {
      return fit_model.get_fit().train_covariance;
    };
    const auto C_dd = fit_models.apply(get_train_covariance);

    // C_db holds the covariance between each model and all boundaries.
    // The actual storage is effectively a map with values which correspond
    // to the covariance between that model's features and the boundaries.
    auto C_db_one_group = [&](const auto &key, const auto &fit_model) {
      const auto group_features =
          as_group_features(key, fit_model.get_fit().train_features);
      return this->patchwork_covariance_matrix(group_features,
                                               boundary_features);
    };
    const auto C_db = fit_models.apply(C_db_one_group);

    // S_bb = C_bb - C_db^T * C_dd^-1 * C_db
    const Eigen::MatrixXd S_bb =
        C_bb - block_inner_product(C_db, block_diag_solve(C_dd, C_db));
    const Eigen::SerializableLDLT S_bb_ldlt(S_bb.ldlt());

    // Similar to with a Gaussian process we can precompute the "information"
    // vector which accelerates the prediction step.  In Patchwork
    // Krigging the "information" is the quantity:
    //
    //   information = (C_dd - C_db C_bb^-1 C_bd)^-1 y
    //               = patchwork_solve(y)
    //
    // where y are the measurements (or targets) from each model.  The fit
    // models don't directly store the measurements in favor of the
    // information vector (v) corresponding to model k which is given by:
    //
    //   v_k = C_k^-1 y_k
    //
    // So we could compute: y_k = C_k v_k then pass that on to the
    // patchwork_solver, but since the patchwork solver is subsequently going
    // to compute C_k^-1 y_k, we can get a computation speedup by directly
    // using v_k.
    auto get_v = [](const auto &fit_model) {
      return fit_model.get_fit().information;
    };
    const auto vs = fit_models.apply(get_v);
    // information = (C_dd - Cdb C_bb^-1 C_bd)^-1 ys
    const auto information = patchwork_solver_from_v(C_dd, C_db, S_bb_ldlt, vs);

    return ReturnType(*this, PatchworkFitType(fit_models, information,
                                              boundary_features, C_dd, C_db,
                                              C_bb_ldlt, S_bb_ldlt));
  };

  template <typename FeatureType>
  auto _fit_impl(const std::vector<FeatureType> &features,
                 const MarginalDistribution &targets) const {

    static_assert(details::patchwork_functions_are_valid<PatchworkFunctions,
                                                         FeatureType>::value,
                  "Invalid PatchworkFunctions for this FeatureType");

    const auto m = gp_from_covariance_and_mean(this->covariance_function_,
                                               this->mean_function_);

    auto create_fit_model = [&](const auto &dataset) { return m.fit(dataset); };

    const RegressionDataset<FeatureType> dataset(features, targets);

    auto grouper = [&](const auto &f) {
      return patchwork_functions_.grouper(f);
    };

    const auto fit_models = dataset.group_by(grouper).apply(create_fit_model);

    return from_fit_models(fit_models).get_fit();
  }

  template <typename FeatureType, typename FitModelType, typename GroupKey,
            typename BoundaryFeatureType>
  JointDistribution _predict_impl(
      const std::vector<FeatureType> &features,
      const Fit<PatchworkGPFit<FitModelType, GroupKey, BoundaryFeatureType>>
          &patchwork_fit,
      PredictTypeIdentity<JointDistribution> &&) const {

    if (patchwork_fit.fit_models.size() == 1) {
      // In this situation there are no boundaries, so predictions
      // can be made directly.
      return patchwork_fit.fit_models.values()[0].predict(features).joint();
    }

    // It's possible that the prediction features are in a group
    // that was unobserved during training in which case we want
    // to use the nearest group's model to make the prediction for
    // those features.
    const auto training_group_keys = patchwork_fit.C_db.keys();
    auto predict_grouper = [&](const auto &f) {
      return patchwork_functions_.nearest_group(
          training_group_keys, patchwork_functions_.grouper(f));
    };

    auto group_features = as_group_features(features, predict_grouper);

    // Making a prediction involves using Equations 6 and 7 which
    // take the form:
    //
    //   m = (C_fd - C_fb C_bb^-1 C_db^T) * information
    //     = cross * information
    // and
    //
    //   E = cross * (C_dd - C_db C_bb^-1 C_db^T) * cross^T
    //     = cross * patchwork_solve(cross^T)
    //
    // Giving:
    //
    //   prediction = N(m, P - E)
    //
    // Where P is the prior covariance and E the explained covariance
    const Eigen::MatrixXd C_fb = patchwork_covariance_matrix(
        group_features, patchwork_fit.boundary_features);

    // First we compute cross_transpose = cross^T which starts
    // by computing:
    //
    //   Q = C_bb^-1 C_fb^T
    //
    // Then forming the grouped matrix:
    //
    //   cross_transpose = C_fd^T - Cdb * Q
    //
    const Eigen::MatrixXd Q = patchwork_fit.C_bb_ldlt.solve(C_fb.transpose());

    auto compute_cross_transpose = [&](const auto &key, const auto &fit_model) {
      const auto train_features =
          as_group_features(key, fit_model.get_fit().train_features);

      // cross_transpose_group = C_df
      Eigen::MatrixXd cross_transpose_group =
          this->patchwork_covariance_matrix(train_features, group_features);
      // cross_transpose = C_df - C_db * Q
      cross_transpose_group -= patchwork_fit.C_db.at(key) * Q;
      return cross_transpose_group;
    };
    const auto cross_transpose =
        patchwork_fit.fit_models.apply(compute_cross_transpose);

    const Eigen::VectorXd mean =
        block_inner_product(cross_transpose, patchwork_fit.information);

    // This is Equation 7 but split into a few steps.
    const auto patchwork_solve_cross_transpose =
        patchwork_solver(patchwork_fit.C_dd, patchwork_fit.C_db,
                         patchwork_fit.S_bb_ldlt, cross_transpose);
    const Eigen::MatrixXd explained =
        block_inner_product(cross_transpose, patchwork_solve_cross_transpose);
    const Eigen::MatrixXd cov =
        patchwork_covariance_matrix(group_features, group_features) - C_fb * Q -
        explained;

    return JointDistribution(mean, cov);
  }

private:
  template <typename X, typename Y>
  Eigen::MatrixXd patchwork_covariance_matrix(const std::vector<X> &xs,
                                              const std::vector<Y> &ys) const {
    auto patchwork_caller = [&](const auto &x, const auto &y) {
      return PatchworkCaller::call(this->covariance_function_, x, y);
    };

    return compute_covariance_matrix(patchwork_caller, xs, ys);
  };

  struct PatchworkFunctionsWithMeasurement {

    PatchworkFunctionsWithMeasurement(){};

    PatchworkFunctionsWithMeasurement(PatchworkFunctions functions)
        : functions_(functions) {}

    template <typename X> auto grouper(const X &x) const {
      static_assert(
          details::patchwork_functions_are_valid<PatchworkFunctions, X>::value,
          "Invalid PatchworkFunctions for this FeatureType");
      return functions_.grouper(x);
    }

    template <typename X> auto grouper(const Measurement<X> &x) const {
      return functions_.grouper(x.value);
    }

    template <typename GroupKey>
    auto boundary(const GroupKey &x, const GroupKey &y) const {
      return functions_.boundary(x, y);
    }

    template <typename GroupKey>
    auto nearest_group(const std::vector<GroupKey> &groups,
                       const GroupKey &query) const {
      return functions_.nearest_group(groups, query);
    }

    PatchworkFunctions functions_;
  };

  PatchworkFunctionsWithMeasurement patchwork_functions_;
};

template <typename CovFunc, typename PatchworkFunctions>
inline PatchworkGaussianProcess<CovFunc, ZeroMean, PatchworkFunctions>
patchwork_gp_from_covariance(CovFunc covariance_function,
                             PatchworkFunctions patchwork_functions) {
  return PatchworkGaussianProcess<CovFunc, ZeroMean, PatchworkFunctions>(
      covariance_function, patchwork_functions);
};

} // namespace albatross

#endif /* INCLUDE_ALBATROSS_MODELS_SPARSE_GP_H_ */
