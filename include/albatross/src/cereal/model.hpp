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

#ifndef ALBATROSS_SRC_CEREAL_MODEL_HPP_
#define ALBATROSS_SRC_CEREAL_MODEL_HPP_

using albatross::ModelBase;

namespace cereal {

template <class Archive, typename ModelType>
void save(Archive &archive, const ModelBase<ModelType> &model,
          const std::uint32_t) {
  archive(cereal::make_nvp("params", model.derived().get_params()));
  archive(cereal::make_nvp("insights", model.derived().insights));
}

template <class Archive, typename ModelType>
void load(Archive &archive, ModelBase<ModelType> &model, const std::uint32_t) {
  albatross::ParameterStore params;
  archive(cereal::make_nvp("params", params));
  model.derived().set_params(params);
  archive(cereal::make_nvp("insights", model.derived().insights));
}

} // namespace cereal

#endif /* ALBATROSS_SRC_CEREAL_MODEL_HPP_ */
