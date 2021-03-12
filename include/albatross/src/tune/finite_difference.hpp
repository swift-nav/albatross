/*
 * Copyright (C) 2021 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef INCLUDE_ALBATROSS_SRC_TUNE_FINITE_DIFFERENCE_HPP_
#define INCLUDE_ALBATROSS_SRC_TUNE_FINITE_DIFFERENCE_HPP_

namespace albatross {

template <typename Function>
inline std::vector<double>
compute_gradient(Function f, const std::vector<double> &params, double f_val,
                 bool use_async = false) {

  std::vector<std::size_t> inds(params.size());
  std::iota(std::begin(inds), std::end(inds), 0);
  const double epsilon = 1e-6;

  auto compute_single_gradient = [&](const std::size_t &ind) {
    std::vector<double> perturbed(params);
    perturbed[ind] += epsilon;
    return (f(perturbed) - f_val) / epsilon;
  };

  if (use_async) {
    return albatross::async_apply(inds, compute_single_gradient);
  } else {
    return albatross::apply(inds, compute_single_gradient);
  }
}

template <typename Function>
inline std::vector<double>
compute_gradient(Function f, const ParameterStore &params, double f_val,
                 bool use_async = false) {

  TunableParameters tunable_params = get_tunable_parameters(params);

  std::vector<std::size_t> inds(tunable_params.values.size());
  std::iota(std::begin(inds), std::end(inds), 0);

  auto get_perturbed = [&](std::size_t i, double epsilon) {
    std::vector<double> perturbed(tunable_params.values);
    perturbed[i] = tunable_params.values[i] + epsilon;
    return set_tunable_params_values(params, perturbed);
  };

  std::vector<double> output(inds.size());

  auto compute_single_sub_gradient = [&](std::size_t i) {
    double epsilon = 1e-6;
    const double range =
        tunable_params.upper_bounds[i] - tunable_params.lower_bounds[i];
    if (std::isfinite(range)) {
      epsilon = 1e-8 * range;
    }
    auto perturbed_params = get_perturbed(i, epsilon);

    if (!params_are_valid(perturbed_params)) {
      epsilon *= -1;
      perturbed_params = get_perturbed(i, epsilon);
    }

    double grad_i = (f(perturbed_params) - f_val) / epsilon;

    if (tunable_params.values[i] >= tunable_params.upper_bounds[i] &&
        grad_i < 0) {
      grad_i = 0;
    }

    if (tunable_params.values[i] <= tunable_params.lower_bounds[i] &&
        grad_i > 0) {
      grad_i = 0;
    }

    return grad_i;
  };

  if (use_async) {
    return albatross::async_apply(inds, compute_single_sub_gradient);
  } else {
    return albatross::apply(inds, compute_single_sub_gradient);
  }
}

} // namespace albatross

#endif /* INCLUDE_ALBATROSS_SRC_TUNE_FINITE_DIFFERENCE_HPP_ */
