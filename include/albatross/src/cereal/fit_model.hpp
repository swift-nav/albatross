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

#ifndef ALBATROSS_SRC_CEREAL_FIT_MODEL_HPP_
#define ALBATROSS_SRC_CEREAL_FIT_MODEL_HPP_

namespace cereal {

template <typename Archive, typename ModelType, typename Fit>
inline void serialize(Archive &archive,
                      albatross::FitModel<ModelType, Fit> &fit_model,
                      const std::uint32_t) {
  archive(cereal::make_nvp("model", fit_model.get_model()));
  archive(cereal::make_nvp("fit", fit_model.get_fit()));
};

}  // namespace cereal

#endif /* ALBATROSS_SRC_CEREAL_FIT_MODEL_HPP_ */
