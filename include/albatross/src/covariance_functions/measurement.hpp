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

#ifndef INCLUDE_ALBATROSS_SRC_COVARIANCE_FUNCTIONS_MEASUREMENT_HPP_
#define INCLUDE_ALBATROSS_SRC_COVARIANCE_FUNCTIONS_MEASUREMENT_HPP_

namespace albatross {

template <typename X> struct Measurement {

  Measurement() : value(){};

  Measurement(const X &x) { value = x; }

  X value;
};

// A simple helper function which aids the compiler with type deduction.
template <typename FeatureType>
Measurement<FeatureType> as_measurement(const FeatureType &f) {
  return Measurement<FeatureType>(f);
}

/*
 * Takes a vector of features and turns it into a vector of
 * measurements of those features.
 */
template <typename FeatureType>
auto as_measurements(const std::vector<FeatureType> &features) {
  std::vector<Measurement<FeatureType>> measurement_features;
  for (const auto &f : features) {
    measurement_features.emplace_back(f);
  }
  return measurement_features;
}

/*
 * This takes a vector of variants of features and turns it into
 * a vector of variants of MEASUREMENTS of features.
 */
template <typename... Ts>
auto as_measurements(const std::vector<variant<Ts...>> &features) {
  std::vector<variant<Measurement<Ts>...>> measurement_features;
  for (const auto &f : features) {
    measurement_features.emplace_back(f.match([](const auto &x) {
      return variant<Measurement<Ts>...>(as_measurement(x));
    }));
  }
  return measurement_features;
}

template <typename SubCovariance>
class MeasurementOnly
    : public CovarianceFunction<MeasurementOnly<SubCovariance>> {

public:
  MeasurementOnly() : sub_cov_(){};
  MeasurementOnly(const SubCovariance &sub_cov) : sub_cov_(sub_cov){};

  std::string name() const {
    return "measurement[" + sub_cov_.get_name() + "]";
  }

  ParameterStore get_params() const override { return sub_cov_.get_params(); }

  void unchecked_set_param(const ParameterKey &name,
                           const Parameter &param) override {
    sub_cov_.set_param(name, param);
  }

  template <
      typename X, typename Y,
      typename std::enable_if<
          has_valid_call_impl<SubCovariance, X &, Y &>::value, int>::type = 0>
  double _call_impl(const X &x, const Y &y) const {
    return 0.;
  };

  template <
      typename X, typename Y,
      typename std::enable_if<
          has_valid_call_impl<SubCovariance, X &, Y &>::value, int>::type = 0>
  double _call_impl(const Measurement<X> &x, const Measurement<Y> &y) const {
    return sub_cov_(x.value, y.value);
  };

private:
  SubCovariance sub_cov_;
};

/*
 * Utility function to act as a constructor but with template param resolution.
 */
template <typename SubCovariance>
MeasurementOnly<SubCovariance> measurement_only(const SubCovariance &cov) {
  return MeasurementOnly<SubCovariance>(cov);
}

} // namespace albatross

#endif /* INCLUDE_ALBATROSS_SRC_COVARIANCE_FUNCTIONS_MEASUREMENT_HPP_ */
