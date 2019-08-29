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

#ifndef THIRD_PARTY_ALBATROSS_INCLUDE_ALBATROSS_SRC_CEREAL_PARAMETERS_HPP_
#define THIRD_PARTY_ALBATROSS_INCLUDE_ALBATROSS_SRC_CEREAL_PARAMETERS_HPP_

namespace cereal {

template <class Archive>
inline void serialize(Archive &archive, albatross::Parameter &param,
                      const std::uint32_t) {
  archive(cereal::make_nvp("value", param.value));
  archive(cereal::make_nvp("prior", param.prior));
};

// template <class Archive>
// inline void save(Archive &archive, const albatross::ParameterHandlingMixin
// &param_handler, const std::uint32_t) {
//  archive(cereal::make_nvp("parameters", param_handler.get_params()));
//};
//
// template <class Archive>
// inline void load(Archive &archive, albatross::ParameterHandlingMixin
// &param_handler, const std::uint32_t) {
//  albatross::ParameterStore params;
//  archive(cereal::make_nvp("parameters", params));
//  param_handler.set_params(params);
//};

} // namespace cereal

#endif /* THIRD_PARTY_ALBATROSS_INCLUDE_ALBATROSS_SRC_CEREAL_PARAMETERS_HPP_   \
        */
