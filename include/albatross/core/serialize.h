/*
 * Copyright (C) 2018 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef ALBATROSS_CORE_SERIALIZE_H
#define ALBATROSS_CORE_SERIALIZE_H

namespace albatross {

template <typename FeatureType, typename ModelFit>
class SerializableRegressionModel : public RegressionModel<FeatureType> {

public:
  using FitType = ModelFit;
  SerializableRegressionModel() : model_fit_(){};
  virtual ~SerializableRegressionModel(){};

  bool operator==(
      const SerializableRegressionModel<FeatureType, ModelFit> &other) const {
    return (this->get_name() == other.get_name() &&
            this->get_params() == other.get_params() &&
            this->has_been_fit() == other.has_been_fit() &&
            model_fit_ == other.get_fit());
  }

  /*
   * Include save/load methods conditional on the ability to serialize
   * ModelFit.
   */
  template <class Archive>
  typename std::enable_if<valid_output_serializer<ModelFit, Archive>::value,
                          void>::type
  save(Archive &archive) const {
    archive(cereal::make_nvp(
        "model_definition",
        cereal::base_class<RegressionModel<FeatureType>>(this)));
    archive(cereal::make_nvp("model_fit", this->model_fit_));
    archive(cereal::make_nvp("insights", this->insights_));
  }

  template <class Archive>
  typename std::enable_if<valid_input_serializer<ModelFit, Archive>::value,
                          void>::type
  load(Archive &archive) {
    archive(cereal::make_nvp(
        "model_definition",
        cereal::base_class<RegressionModel<FeatureType>>(this)));
    archive(cereal::make_nvp("model_fit", this->model_fit_));
    archive(cereal::make_nvp("insights", this->insights_));
  }

  /*
   * If ModelFit does not have valid serialization methods and you attempt to
   * (de)serialize a SerializableRegressionModel you'll get an error.
   */
  template <class Archive>
  typename std::enable_if<!valid_output_serializer<ModelFit, Archive>::value,
                          void>::type
  save(Archive &) const {
    static_assert(delay_static_assert<Archive>::value,
                  "SerializableRegressionModel requires a ModelFit type which "
                  "is serializable.");
  }

  template <class Archive>
  typename std::enable_if<!valid_input_serializer<ModelFit, Archive>::value,
                          void>::type
  load(Archive &) const {
    static_assert(delay_static_assert<Archive>::value,
                  "SerializableRegressionModel requires a ModelFit type which "
                  "is serializable.");
  }

  virtual ModelFit get_fit() const { return model_fit_; }

protected:
  void fit_(const std::vector<FeatureType> &features,
            const MarginalDistribution &targets) {
    model_fit_ = serializable_fit_(features, targets);
  }

  virtual ModelFit
  serializable_fit_(const std::vector<FeatureType> &features,
                    const MarginalDistribution &targets) const = 0;

  ModelFit model_fit_;
};
} // namespace albatross

#endif
