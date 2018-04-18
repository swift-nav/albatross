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

#ifndef ALBATROSS_CORE_MODEL_ADAPTER_H
#define ALBATROSS_CORE_MODEL_ADAPTER_H

namespace albatross {

/*
 * This provides a way of creating a RegressionModel<X> which
 * wraps a RegressionModel<Y> as long as you provide a way of
 * converting from X to Y.
 *
 * A good example can be found in the definition of LinearRegression.
 */
template <typename FeatureType, typename SubModelType>
class AdaptedRegressionModel : public RegressionModel<FeatureType> {
 public:
  typedef typename SubModelType::Feature SubFeature;

  AdaptedRegressionModel() : sub_model_(){};
  AdaptedRegressionModel(SubModelType& sub_model) : sub_model_(sub_model){};
  virtual ~AdaptedRegressionModel() {};

  virtual SubFeature convert_feature(const FeatureType& parent_feature) const = 0;

  std::string get_name() const override { return sub_model_.get_name(); };

  void fit_(const std::vector<FeatureType>& features,
            const Eigen::VectorXd& targets) override {
    sub_model_.fit(convert_features(features), targets);
  }

  bool has_been_fit() const override { return sub_model_.has_been_fit(); }

  PredictionDistribution predict_(
      const std::vector<FeatureType>& features) const override {
    auto predictions = sub_model_.predict(convert_features(features));
    return predictions;
  };

  ParameterStore get_params() const override { return sub_model_.get_params(); }

  void unchecked_set_param(const std::string& name,
                           const double value) override {
    sub_model_.set_param(name, value);
  }

  template<class Archive>
  void save(Archive & archive) const
  {
    archive(cereal::make_nvp("sub_model", cereal::base_class<RegressionModel<FeatureType>>(this)));
  }

  template<class Archive>
  void load(Archive & archive) const
  {
    archive(cereal::make_nvp("sub_model", cereal::base_class<RegressionModel<FeatureType>>(this)));
  }

 protected:

  std::vector<SubFeature> convert_features(const std::vector<FeatureType> &parent_features) const {
    std::vector<SubFeature> converted;
    for (const auto &f : parent_features) {
      converted.push_back(convert_feature(f));
    }
    return converted;
  }

  SubModelType sub_model_;
};


}

#endif
