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

#ifndef ALBATROSS_SRC_SAMPLERS_ENSEMBLE_HPP_
#define ALBATROSS_SRC_SAMPLERS_ENSEMBLE_HPP_

namespace albatross {

inline auto split_indices(std::size_t n, std::size_t k) {
  std::vector<std::size_t> inds(n);
  std::iota(inds.begin(), inds.end(), 0);
  return group_by(inds, KFoldGrouper(k)).indexers().values();
}

inline std::size_t random_complement(std::size_t n, std::size_t i,
                                     std::default_random_engine &gen) {
  // Draws a random integer between 0 and n - 1 excluding i.
  ALBATROSS_ASSERT(n > 1);
  std::uniform_int_distribution<std::size_t> uniform_int(0, n - 1);
  std::size_t output;
  while (true) {
    output = uniform_int(gen);
    if (output != i) {
      return output;
    }
  }
}

void assert_valid_states(const EnsembleSamplerState &ensembles) {
  for (std::size_t i = 0; i < ensembles.size(); ++i) {
    ALBATROSS_ASSERT(std::isfinite(ensembles[i].log_prob));
  }
}

template <typename ComputeLogProb>
EnsembleSamplerState stretch_move_step(const EnsembleSamplerState &ensembles,
                                       ComputeLogProb compute_log_prob,
                                       std::default_random_engine &gen,
                                       std::size_t n_splits = 2,
                                       double a = 2.) {

  const std::size_t n_ensembles = ensembles.size();
  const std::size_t n_dim = ensembles[0].params.size();

  assert_valid_states(ensembles);

  std::vector<ParameterStore> proposed(n_ensembles);
  std::vector<double> log_prob_z(n_ensembles);

  std::uniform_real_distribution<double> uniform_real(0.0, 1.0);
  std::uniform_int_distribution<std::size_t> uniform_int(0, n_ensembles - 1);

  const auto splits = split_indices(n_ensembles, n_splits);

  EnsembleSamplerState next_ensembles(ensembles);

  for (const auto &split : splits) {

    const auto complement = indices_complement(split, n_ensembles);
    std::uniform_int_distribution<std::size_t> random_complement_idx(
        0, complement.size() - 1);

    for (const std::size_t &k : split) {

      SamplerState current(next_ensembles[k]);
      SamplerState proposed(current);
      proposed.accepted = false;

      // Choose X_j from the complementary ensembles.
      std::size_t j = complement[random_complement_idx(gen)];
      // It's possible the sampler will get initialized with non-finite
      // probabilities, if that happens we want to only pick from the
      // set which are finite.
      while (!std::isfinite(next_ensembles[j].log_prob)) {
        j = complement[random_complement_idx(gen)];
      }

      // Draw a random z
      double p = uniform_real(gen);
      double z = pow((a - 1.0) * p + 1, 2.0) / a;
      log_prob_z[k] = (n_dim - 1.0) * log(z);

      // proposed = x_j + z * (x_k - x_j)
      for (std::size_t i = 0; i < n_dim; ++i) {
        const double v_j = next_ensembles[j].params[i];
        double delta = (proposed.params[i] - v_j);
        // Occasionally (especially with bounds) some
        // parameters can end up identical across samples
        // in this occasion we switch to a gaussian style
        // move with very small variance.
        if (delta == 0.) {
          delta = 1e-6;
        }
        proposed.params[i] = v_j + z * delta;
      }

      proposed.log_prob = compute_log_prob(proposed.params);
      const double log_diff =
          log_prob_z[k] + proposed.log_prob - current.log_prob;

      const double random = uniform_real(gen);
      const bool accepted = log_diff > log(random);
      const bool is_finite = std::isfinite(proposed.log_prob);
      if (accepted && is_finite) {
        proposed.accepted = true;
        next_ensembles[k] = proposed;
      } else {
        next_ensembles[k].accepted = false;
      }
    }
  }

  return next_ensembles;
}

template <typename ComputeLogProb, typename CallbackFunc = NullCallback>
std::vector<EnsembleSamplerState>
ensemble_sampler(ComputeLogProb &&compute_log_prob,
                 const EnsembleSamplerState &initial_state,
                 std::size_t max_iterations, std::default_random_engine &gen,
                 CallbackFunc &&callback = NullCallback()) {

  EnsembleSamplerState state = ensure_finite_initial_state(
      std::forward<ComputeLogProb>(compute_log_prob), initial_state, gen);

  std::vector<EnsembleSamplerState> output;
  output.push_back(state);
  callback(0, state);

  for (std::size_t iter = 1; iter <= max_iterations; ++iter) {
    const auto next_state = stretch_move_step<ComputeLogProb>(
        state, std::forward<ComputeLogProb>(compute_log_prob), gen);
    output.push_back(next_state);
    callback(iter, next_state);
    state = next_state;
  }
  return output;
}

template <typename ComputeLogProb, typename CallbackFunc = NullCallback>
std::vector<EnsembleSamplerState>
ensemble_sampler(ComputeLogProb &&compute_log_prob,
                 const std::vector<std::vector<double>> &params,
                 std::size_t max_iterations, std::default_random_engine &gen,
                 CallbackFunc &&callback = NullCallback()) {

  EnsembleSamplerState ensembles;
  for (const auto &p : params) {
    SamplerState state = {p, compute_log_prob(p), true};
    ensembles.push_back(state);
  }

  return ensemble_sampler(std::forward<ComputeLogProb>(compute_log_prob),
                          ensembles, max_iterations, gen,
                          std::forward<CallbackFunc>(callback));
}

template <typename ModelType, typename FeatureType,
          typename CallbackFunc = NullCallback>
std::vector<EnsembleSamplerState> ensemble_sampler(
    const ModelType &model, const RegressionDataset<FeatureType> &dataset,
    std::size_t n_walkers, std::size_t max_iterations,
    std::default_random_engine &gen, CallbackFunc &&callback = NullCallback()) {

  std::normal_distribution<double> jitter(0., 0.1);
  const auto initial_params =
      initial_params_from_jitter(model.get_params(), jitter, gen, n_walkers);

  auto compute_ll = [&](const std::vector<double> &param_values) {
    ModelType m(model);
    m.set_tunable_params_values(param_values);
    return m.log_likelihood(dataset);
  };

  return ensemble_sampler(compute_ll, initial_params, max_iterations, gen,
                          std::forward<CallbackFunc>(callback));
}

} // namespace albatross

#endif /* INCLUDE_ALBATROSS_SRC_SAMPLERS_ENSEMBLE_HPP_ */
