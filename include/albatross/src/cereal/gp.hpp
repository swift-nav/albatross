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

#ifndef GP_SERIALIZATION_VERSION
#define GP_SERIALIZATION_VERSION 1
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

template <typename Archive, typename CovFunc, typename MeanFunc,
          typename ImplType>
void save(Archive &archive,
          const GaussianProcessBase<CovFunc, MeanFunc, ImplType> &gp,
          const std::uint32_t) {
  archive(cereal::make_nvp("name", gp.get_name()));
  archive(cereal::make_nvp("params", gp.get_params()));
  archive(cereal::make_nvp("insights", gp.insights));
}

template <typename Archive, typename CovFunc, typename MeanFunc,
          typename ImplType>
void load(Archive &archive,
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

// Here we define the version for variant serialization following the
// example given here: https://github.com/USCiLab/cereal/issues/319
namespace detail {
template <typename CovFunc, typename MeanFunc, typename ImplType>
struct Version<GaussianProcessBase<CovFunc, MeanFunc, ImplType>> {
  static const std::uint32_t version;
  static std::uint32_t registerVersion() {
    ::cereal::detail::StaticObject<Versions>::getInstance().mapping.emplace(
        std::type_index(
            typeid(GaussianProcessBase<CovFunc, MeanFunc, ImplType>))
            .hash_code(),
        GP_SERIALIZATION_VERSION);
    return GP_SERIALIZATION_VERSION;
  }
  static void unused() { (void)version; }
};

template <typename CovFunc, typename MeanFunc, typename ImplType>
const std::uint32_t
    Version<GaussianProcessBase<CovFunc, MeanFunc, ImplType>>::version =
        Version<GaussianProcessBase<CovFunc, MeanFunc,
                                    ImplType>>::registerVersion();
} // namespace detail

} // namespace cereal

#endif /* ALBATROSS_SRC_CEREAL_GP_HPP_ */
