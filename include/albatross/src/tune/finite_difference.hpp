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

#include <sys/resource.h>

namespace albatross {

// Class which has the sole purpose of increasing stack size so that we don't
// crash
class StackInit {
 public:
  StackInit() {
    const rlim_t kStackSize = 80 * 1024L * 1024L;  // min stack size = 80 Mb
    struct rlimit rl;
    int result;

    result = getrlimit(RLIMIT_STACK, &rl);
    if (result == 0) {
      if (rl.rlim_cur < kStackSize) {
        rl.rlim_cur = kStackSize;
        if (kStackSize > rl.rlim_max) {
          fprintf(stderr, "Max stack size is limited to %llu bytes\n",
                  static_cast<unsigned long long>(rl.rlim_max));
          rl.rlim_cur = rl.rlim_max - 1;
        }

        result = setrlimit(RLIMIT_STACK, &rl);
        if (result != 0) {
          fprintf(stderr, "setrlimit returned result = %d\n", result);
        }
      }
    }
  }
};


template <typename Function>
double compute_single_gradient(Function f, const std::vector<double> &params,
                               double f_val,
                               std::size_t i) {
  const double epsilon = 1e-6;
  std::vector<double> perturbed(params);
  perturbed[i] += epsilon;
  return (f(perturbed) - f_val) / epsilon;
}

template <typename Function>
std::vector<double> compute_gradient(Function f,
                                     const std::vector<double> &params,
                                     double f_val) {

  std::vector<std::size_t> inds(params.size());
  std::iota(std::begin(inds), std::end(inds), 0);

  auto grad_for_index = [&](const std::size_t &ind) {
    return compute_single_gradient(f, params, f_val, ind);
  };

  return albatross::apply(inds, grad_for_index);
}


template <typename Function>
std::vector<double> compute_gradient(Function f,
                                     const ParameterStore &params,
                                     double f_val,
                                     std::size_t n_proc = 1) {

  TunableParameters tunable_params = get_tunable_parameters(params);

  std::vector<std::size_t> inds(tunable_params.values.size());
  std::iota(std::begin(inds), std::end(inds), 0);

  auto get_perturbed = [&](std::size_t i, double epsilon) {
    std::vector<double> perturbed(tunable_params.values);
    perturbed[i] = tunable_params.values[i] + epsilon;
    return set_tunable_params_values(params, perturbed);
  };

  std::mutex grad_mutex;

  std::vector<double> output(inds.size());

  auto compute_single_sub_gradient = [&](std::size_t i) {
    double epsilon = 1e-6;
    const double range = tunable_params.upper_bounds[i] - tunable_params.lower_bounds[i];
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

    {
      std::lock_guard<std::mutex> lock(grad_mutex);
      output[i] = grad_i;
    }
  };

  if (n_proc > 1) {
    albatross::async_apply(inds, compute_single_sub_gradient);
  } else {
    albatross::apply(inds, compute_single_sub_gradient);
  }

  return output;
}

} // namespace albatross

#endif /* INCLUDE_ALBATROSS_SRC_TUNE_FINITE_DIFFERENCE_HPP_ */
