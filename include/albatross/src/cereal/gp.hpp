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

#ifndef ALBATROSS_SRC_CEREAL_GP_HPP_
#define ALBATROSS_SRC_CEREAL_GP_HPP_

using albatross::Fit;
using albatross::GaussianProcessBase;
using albatross::GPFit;
using albatross::LinearCombination;

using albatross::SparseGPFit;
using albatross::PICGPFit;

#ifndef GP_SERIALIZATION_VERSION
#define GP_SERIALIZATION_VERSION 2
#endif

namespace cereal {

template <typename Archive, typename CovarianceRepresentation,
          typename FeatureType>
inline void serialize(Archive &archive,
                      Fit<GPFit<CovarianceRepresentation, FeatureType>> &fit,
                      const std::uint32_t) {
  archive(cereal::make_nvp("information", fit.information));
  archive(cereal::make_nvp("train_ldlt", fit.train_covariance));
  archive(cereal::make_nvp("train_features", fit.train_features));
}

template <typename Archive, typename FeatureType>
inline void serialize(Archive &archive, Fit<SparseGPFit<FeatureType>> &fit,
                      const std::uint32_t version) {
  archive(cereal::make_nvp("information", fit.information));
  archive(cereal::make_nvp("train_covariance", fit.train_covariance));
  archive(cereal::make_nvp("train_features", fit.train_features));
  archive(cereal::make_nvp("R", fit.R));
  archive(cereal::make_nvp("P", fit.P));
  if (version > 1) {
    archive(cereal::make_nvp("numerical_rank", fit.numerical_rank));
  } else {
    // Use a negative number to make it clear this is not a valid value.
    fit.numerical_rank = -1;
  }
}

template <typename Archive, typename GrouperFunction,
          typename InducingFeatureType, typename FeatureType>
inline void
serialize(Archive &archive,
          Fit<PICGPFit<GrouperFunction, InducingFeatureType, FeatureType>> &fit,
          const std::uint32_t ) {
  archive(cereal::make_nvp("train_features", fit.train_features));
  archive(cereal::make_nvp("inducing_features", fit.inducing_features));
  archive(cereal::make_nvp("train_covariance", fit.train_covariance));
  archive(cereal::make_nvp("sigma_R", fit.sigma_R));
  archive(cereal::make_nvp("P", fit.P));
  archive(cereal::make_nvp("mean_w", fit.mean_w));
  archive(cereal::make_nvp("W", fit.W));
  archive(cereal::make_nvp("covariance_Y", fit.covariance_Y));
  archive(cereal::make_nvp("Z", fit.Z));
  archive(cereal::make_nvp("A_ldlt", fit.A_ldlt));
  archive(cereal::make_nvp("measurement_groups", fit.measurement_groups));
  archive(cereal::make_nvp("information", fit.information));
  archive(cereal::make_nvp("numerical_rank", fit.numerical_rank));
  archive(cereal::make_nvp("cols_Bs", fit.cols_Bs));
}

template <typename Archive, typename CovFunc, typename MeanFunc,
          typename ImplType>
inline void save(Archive &archive,
                 const GaussianProcessBase<CovFunc, MeanFunc, ImplType> &gp,
                 const std::uint32_t) {
  archive(cereal::make_nvp("name", gp.get_name()));
  archive(cereal::make_nvp("params", gp.get_params()));
  archive(cereal::make_nvp("insights", gp.insights));
}

template <typename Archive, typename CovFunc, typename MeanFunc,
          typename ImplType>
inline void load(Archive &archive,
                 GaussianProcessBase<CovFunc, MeanFunc, ImplType> &gp,
                 const std::uint32_t version) {
  if (version > 0) {
    std::string model_name;
    archive(cereal::make_nvp("name", model_name));
    gp.set_name(model_name);
  }
  albatross::ParameterStore params;
  archive(cereal::make_nvp("params", params));
  gp.set_params(params);
  archive(cereal::make_nvp("insights", gp.insights));
}

template <typename Archive, typename FeatureType>
inline void serialize(Archive &archive, LinearCombination<FeatureType> &combo,
                      const std::uint32_t) {
  archive(cereal::make_nvp("values", combo.values));
  archive(cereal::make_nvp("coefficients", combo.coefficients));
}

} // namespace cereal

#endif /* ALBATROSS_SRC_CEREAL_GP_HPP_ */
