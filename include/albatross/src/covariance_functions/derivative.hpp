/*
 * Copyright (C) 2020 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef INCLUDE_ALBATROSS_SRC_COVARIANCE_FUNCTIONS_DERIVATIVE_HPP_
#define INCLUDE_ALBATROSS_SRC_COVARIANCE_FUNCTIONS_DERIVATIVE_HPP_

namespace albatross {

template <typename T> struct Derivative {

  Derivative() : value(){};

  Derivative(const T &t) : value(t){};

  T value;
};

template <typename T> struct SecondDerivative {

  SecondDerivative() : value(){};

  SecondDerivative(const T &t) : value(t){};

  T value;
};

} // namespace albatross

#endif /* INCLUDE_ALBATROSS_SRC_COVARIANCE_FUNCTIONS_DERIVATIVE_HPP_ */
