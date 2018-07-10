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

#include "core/parameter_handling_mixin.h"
#include <gtest/gtest.h>

#include "test_utils.h"

namespace albatross {

/*
 * Simply makes sure that a BaseModel that should be able to
 * make perfect predictions compiles and runs as expected.
 */
TEST(test_parameter_handler, test_get_set) {
  auto p = TestParameterHandler();
  auto params = p.get_params();
  for (auto &pair : params) {
    pair.second.value += 1.;
  }
  // Make sure modifying the returned map doesn't modify the original;
  auto unmodified_params = p.get_params();
  EXPECT_EQ(unmodified_params["A"], 1.);
  EXPECT_EQ(unmodified_params["B"], 2.);
  // Then make sure after we set the new params they stick.
  p.set_params(params);
  auto modified_params = p.get_params();
  EXPECT_EQ(modified_params["A"], 2.);
  EXPECT_EQ(modified_params["B"], 3.);
};

/*
 * Here we test to make sure that the parameters are in the same
 * order regardless of how they're created.  This is the case
 * when using std::map as a store, but if we were to change backends
 * we'd want to make sure this property held.
 */
TEST(test_parameter_handler, test_is_ordered) {
  const ParameterStore ordered = {{"1", 1.}, {"2", 2.}, {"3", 3.}};
  const ParameterStore unordered = {{"2", 2.}, {"1", 1.}, {"3", 3.}};

  // march through each store one by one and make sure the keys are the same
  typedef ParameterStore::const_iterator iter_t;
  for (std::pair<iter_t, iter_t> p(ordered.begin(), unordered.begin());
       p.first != ordered.end(); ++p.first, ++p.second) {
    const auto ordered_pair = *p.first;
    const auto unordered_pair = *p.second;
    EXPECT_EQ(ordered_pair.first, unordered_pair.first);
  }
  EXPECT_EQ(ordered, unordered);
}

/*
 * Test the helper functions that let you get and set parameters from
 * a vector of values.
 */
TEST(test_parameter_handler, test_get_set_from_vector) {
  const ParameterStore expected = {{"1", 4.}, {"2", 5.}, {"3", 6.}};
  const std::vector<ParameterValue> expected_param_vector = {4., 5., 6.};

  const ParameterStore original = {{"2", 2.}, {"1", 1.}, {"3", 3.}};
  const std::vector<ParameterValue> original_param_vector = {1., 2., 3.};
  MockParameterHandler original_handler(original);

  // Make sure we start with the parameter vector we'd expect, even though
  // it was initialized out of order.
  expect_parameter_vector_equal(original_param_vector,
                                original_handler.get_params_as_vector());

  // Now set the parameters using a new vector and make sure they stick
  original_handler.set_params_from_vector(expected_param_vector);
  expect_parameter_vector_equal(expected_param_vector,
                                original_handler.get_params_as_vector());
}

TEST(test_parameter_handler, test_prior_log_likelihood) {
  auto p = TestParameterHandler();

  auto params = p.get_params();

  double expected;
  {
    ParameterPrior gaussian_prior = std::make_shared<GaussianPrior>(3., 5.);
    ParameterPrior uninformative_prior = std::make_shared<UninformativePrior>();
    expected = gaussian_prior->log_pdf(p.get_params().at("A").value) +
               uninformative_prior->log_pdf(p.get_params().at("B").value);
    p.set_prior("A", gaussian_prior);
    p.set_prior("B", uninformative_prior);
  }
  EXPECT_DOUBLE_EQ(expected, p.prior_log_likelihood());
};

TEST(test_parameter_handler, test_set_prior) {
  auto p = TestParameterHandler();

  const auto orig_params = p.get_params();
  const auto orig_param_vector = p.get_params_as_vector();

  for (const auto &pair : orig_params) {
    ParameterPrior gaussian_prior =
        std::make_shared<GaussianPrior>(pair.second.value + 1., 1.);
    p.set_prior(pair.first, gaussian_prior);
  }

  const auto params_with_priors = p.get_params();
  // The actual parameters now have priors, so they shouldn't be the
  // same.
  EXPECT_NE(orig_params, params_with_priors);
  // But the parameter values shouldn't have changed.
  expect_parameter_vector_equal(orig_param_vector, p.get_params_as_vector());

  // We should also be able to change the parameter values without
  // modifying the prior.
  for (const auto &pair : orig_params) {
    p.set_param(pair.first, pair.second.value + 3.14159);
    EXPECT_EQ(p.get_params().at(pair.first).prior,
              params_with_priors.at(pair.first).prior);
  }
};

} // namespace albatross
