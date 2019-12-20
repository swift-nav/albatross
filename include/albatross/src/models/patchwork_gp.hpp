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
auto patchwork_solver(
    const Grouped<GroupKey, Solver> &A,
    const Grouped<GroupKey, Eigen::MatrixXd> &C,
    const Eigen::SerializableLDLT &S,
    const Grouped<GroupKey, Eigen::Matrix<double, Rows, Cols>> &rhs) {
  // A^-1 rhs + A^-1 C (B - C^T A^-1 C)^-1 C^T A^-1 rhs
  // A^-1 rhs + A^-1 C S^-1 C^T A^-1 rhs

  const auto Ai_rhs = block_diag_solve(A, rhs);

  // S^-1 C^T A^-1 rhs
  auto SiCtAi_rhs_block = [&](const Eigen::MatrixXd &C_i,
                              const auto &Ai_rhs_i) {
    return Eigen::MatrixXd(S.solve(C_i.transpose() * Ai_rhs_i));
  };
  const Eigen::MatrixXd SiCtAi_rhs =
      block_accumulate(C, Ai_rhs, SiCtAi_rhs_block);

  auto product_with_SiCtAi_rhs = [&](const auto &key, const auto &C_i) {
    return Eigen::MatrixXd(C_i * SiCtAi_rhs);
  };
  const auto CSiCtAi_rhs = C.apply(product_with_SiCtAi_rhs);

  auto output = block_diag_solve(A, CSiCtAi_rhs);
  // Adds A^-1 rhs to A^-1 C S^-1 C^T A^-1 rhs
  auto add_Ai_rhs = [&](const auto &key, const auto &group) {
    return Eigen::MatrixXd(group + Ai_rhs.at(key)).eval();
  };
  return output.apply(add_Ai_rhs);
};

template <typename CovFunc, typename PatchworkFunctions>
class PatchworkGaussianProcess
    : public GaussianProcessBase<
          CovFunc, PatchworkGaussianProcess<CovFunc, PatchworkFunctions>> {

public:
  using Base = GaussianProcessBase<
      CovFunc, PatchworkGaussianProcess<CovFunc, PatchworkFunctions>>;

  PatchworkGaussianProcess() : Base(){};
  PatchworkGaussianProcess(CovFunc &covariance_function)
      : Base(covariance_function){};
  PatchworkGaussianProcess(CovFunc &covariance_function,
                           PatchworkFunctions patchwork_functions)
      : Base(covariance_function), patchwork_functions_(patchwork_functions){};

  template <typename X, typename Y>
  Eigen::MatrixXd patchwork_covariance_matrix(const std::vector<X> &xs,
                                              const std::vector<Y> &ys) const {
    auto patchwork_caller = [&](const auto &x, const auto &y) {
      return PatchworkCaller::call(this->covariance_function_, x, y);
    };

    return compute_covariance_matrix(patchwork_caller, xs, ys);
  };

  template <typename FitModelType, typename GroupKey>
  auto
  from_fit_models(const Grouped<GroupKey, FitModelType> &fit_models) const {

    auto boundary_function = [&](const GroupKey &x, const GroupKey &y) {
      return patchwork_functions_.boundary(x, y);
    };

    const auto boundary_features =
        build_boundary_features(boundary_function, fit_models.keys());

    using BoundaryFeatureType = typename std::decay<typename decltype(
        boundary_features)::value_type>::type;
    using PatchworkFitType =
        Fit<PatchworkGPFit<FitModelType, GroupKey, BoundaryFeatureType>>;
    using ReturnType = FitModel<PatchworkGaussianProcess, PatchworkFitType>;

    if (fit_models.size() == 1) {
      return ReturnType(*this, PatchworkFitType(fit_models));
    }

    assert(boundary_features.size() > 0);

    // C_bb is the covariance matrix between all boundaries, it will
    // have a lot of zeros so it could be decomposed more efficiently
    const Eigen::MatrixXd C_bb =
        patchwork_covariance_matrix(boundary_features, boundary_features);
    const auto C_bb_ldlt = C_bb.ldlt();

    // C_dd is the large block diagonal matrix, with one block for each model
    // for which we already have an efficient way of computing the inverse.
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
    const auto C_dd_inv_C_db = block_diag_solve(C_dd, C_db);

    //    S_bb = C_bb - C_db * C_dd^-1 * C_db
    const Eigen::MatrixXd S_bb =
        C_bb - block_inner_product(C_db, C_dd_inv_C_db);
    const auto S_bb_ldlt = S_bb.ldlt();

    auto get_obs_vector = [](const auto &fit_model) {
      return fit_model.predict(fit_model.get_fit().train_features).mean();
    };

    const auto ys = fit_models.apply(get_obs_vector);
    const auto information = patchwork_solver(C_dd, C_db, S_bb_ldlt, ys);

    Eigen::VectorXd C_bb_inv_C_bd_information =
        Eigen::VectorXd::Zero(C_bb.rows());
    auto accumulate_C_bb_inv_C_bd_information = [&](const auto &key,
                                                    const auto &C_bd_i) {
      C_bb_inv_C_bd_information +=
          C_bb_ldlt.solve(C_bd_i.transpose() * information.at(key));
    };
    C_db.apply(accumulate_C_bb_inv_C_bd_information);

    return ReturnType(*this, PatchworkFitType(fit_models, information,
                                              boundary_features, C_dd, C_db,
                                              C_bb_ldlt, S_bb_ldlt));
  };

  template <typename FeatureType, typename FitModelType, typename GroupKey,
            typename BoundaryFeatureType>
  JointDistribution _predict_impl(
      const std::vector<FeatureType> &features,
      const Fit<PatchworkGPFit<FitModelType, GroupKey, BoundaryFeatureType>>
          &patchwork_fit,
      PredictTypeIdentity<JointDistribution> &&) const {

    if (patchwork_fit.fit_models.size() == 1) {
      return patchwork_fit.fit_models.values()[0].predict(features).joint();
    }

    auto predict_grouper = [&](const auto &f) {
      return patchwork_functions_.nearest_group(
          patchwork_fit.C_db.keys(), patchwork_functions_.grouper(f));
    };

    auto group_features = as_group_features(features, predict_grouper);

    const Eigen::MatrixXd C_fb = patchwork_covariance_matrix(
        group_features, patchwork_fit.boundary_features);
    const Eigen::MatrixXd C_fb_bb_inv =
        patchwork_fit.C_bb_ldlt.solve(C_fb.transpose()).transpose();

    auto compute_cross_block_transpose = [&](const auto &key,
                                             const auto &fit_model) {
      const auto train_features =
          as_group_features(key, fit_model.get_fit().train_features);
      Eigen::MatrixXd block =
          this->patchwork_covariance_matrix(train_features, group_features);
      block -= patchwork_fit.C_db.at(key) * C_fb_bb_inv.transpose();
      return block;
    };

    const auto cross_transpose =
        patchwork_fit.fit_models.apply(compute_cross_block_transpose);
    const auto C_dd_inv_cross =
        patchwork_solver(patchwork_fit.C_dd, patchwork_fit.C_db,
                         patchwork_fit.S_bb_ldlt, cross_transpose);

    const Eigen::VectorXd mean =
        block_inner_product(cross_transpose, patchwork_fit.information);
    const Eigen::MatrixXd explained =
        block_inner_product(cross_transpose, C_dd_inv_cross);

    const Eigen::MatrixXd cov =
        patchwork_covariance_matrix(group_features, group_features) -
        C_fb_bb_inv * C_fb.transpose() - explained;

    return JointDistribution(mean, cov);
  }

  template <typename FeatureType>
  auto _fit_impl(const std::vector<FeatureType> &features,
                 const MarginalDistribution &targets) const {

    static_assert(details::patchwork_functions_are_valid<PatchworkFunctions,
                                                         FeatureType>::value,
                  "Invalid PatchworkFunctions for this FeatureType");

    const auto m = gp_from_covariance(this->covariance_function_, "internal");

    auto create_fit_model = [&](const auto &dataset) { return m.fit(dataset); };

    const RegressionDataset<FeatureType> dataset(features, targets);

    auto grouper = [&](const auto &f) {
      return patchwork_functions_.grouper(f);
    };

    const auto fit_models = dataset.group_by(grouper).apply(create_fit_model);

    return from_fit_models(fit_models).get_fit();
  }

  struct PatchworkFunctionsWithMeasurement {

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
inline PatchworkGaussianProcess<CovFunc, PatchworkFunctions>
patchwork_gp_from_covariance(CovFunc covariance_function,
                             PatchworkFunctions patchwork_functions) {
  return PatchworkGaussianProcess<CovFunc, PatchworkFunctions>(
      covariance_function, patchwork_functions);
};

} // namespace albatross

#endif /* INCLUDE_ALBATROSS_MODELS_SPARSE_GP_H_ */
