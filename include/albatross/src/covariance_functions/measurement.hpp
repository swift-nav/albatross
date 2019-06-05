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

template <typename SubCovariance>
class MeasurementOnly
    : public CovarianceFunction<MeasurementOnly<SubCovariance>> {

public:
  ~MeasurementOnly(){};

  std::string name() const {
    return "measurement[" + sub_cov_.get_name() + "]";
  }

  /*
   * This will create a scaled identity matrix, but only between
   * two different observations defined by the Observed type.
   */
  template <
      typename X, typename Y,
      typename std::enable_if<
          has_valid_call_impl<SubCovariance, X &, Y &>::value, int>::type = 0>
  double _call_impl(const X &x, const Y &y) const {
    return 0.;
  };

  /*
   * This will create a scaled identity matrix, but only between
   * two different observations defined by the Observed type.
   */
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

} // namespace albatross

#endif /* INCLUDE_ALBATROSS_SRC_COVARIANCE_FUNCTIONS_MEASUREMENT_HPP_ */
