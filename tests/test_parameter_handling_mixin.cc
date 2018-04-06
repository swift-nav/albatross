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

#include <gtest/gtest.h>
#include "core/parameter_handling_mixin.h"

namespace albatross {

class MockParameterHandler : public ParameterHandlingMixin {
 public:
  MockParameterHandler(const ParameterStore &params)
      : ParameterHandlingMixin(params){};
};

class TestParameterHandler : public ParameterHandlingMixin {
 public:
  TestParameterHandler() : ParameterHandlingMixin() {
    params_ = {{"A", 1.}, {"B", 2.}};
  };
};

/*
 * Simply makes sure that a BaseModel that should be able to
 * make perfect predictions compiles and runs as expected.
 */
TEST(test_parameter_handler, test_get_set) {
  auto p = TestParameterHandler();
  auto params = p.get_params();
  for (auto &pair : params) {
    pair.second += 1.;
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

void expect_params_equal(const ParameterStore &x, const ParameterStore &y) {
  // Make sure all pairs in x are in y.
  for (const auto &x_pair : x) {
    const auto y_value = y.at(x_pair.first);
    EXPECT_DOUBLE_EQ(x_pair.second, y_value);
  }
  // And all pairs in y are in x.
  for (const auto &y_pair : y) {
    const auto x_value = x.at(y_pair.first);
    EXPECT_DOUBLE_EQ(y_pair.second, x_value);
  }
}

void expect_parameter_vector_equal(const std::vector<ParameterValue> &x,
                                   const std::vector<ParameterValue> &y) {
  for (std::size_t i = 0; i < x.size(); i++) {
    EXPECT_DOUBLE_EQ(x[i], y[i]);
  }
  EXPECT_EQ(x.size(), y.size());
}

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
       p.first != ordered.end();
       ++p.first, ++p.second)
  {
    const auto ordered_pair = *p.first;
    const auto unordered_pair = *p.second;
    EXPECT_EQ(ordered_pair.first, unordered_pair.first);
  }

  expect_params_equal(ordered, unordered);
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

/*
 * Tests to make sure we can serialize from one parameter handler to
 * another.
 */
TEST(test_parameter_handler, test_to_string) {
  const ParameterStore original = {{"2", 2.}, {"1", 1.}, {"3", 3.}};
  const std::vector<ParameterValue> original_param_vector = {1., 2., 3.};
  MockParameterHandler original_handler(original);

  // can print as string
  std::ostringstream os;
  os << original_handler.to_string() << std::endl;
}

///*
// * Tests to make sure we can serialize from one parameter handler to
// * another.
// */
//TEST(test_parameter_handler, test_to_string) {
//  const ParameterStore original = {{"2", 2.}, {"1", 1.}, {"3", 3.}};
//  const std::vector<ParameterValue> original_param_vector = {1., 2., 3.};
//  MockParameterHandler original_handler(original);
//  // Make another handler that starts with different parameters
//  MockParameterHandler new_handler({{"2", 4.}, {"1", 5.}, {"3", 6.}});
//
//  // can deserialize
//  new_handler.from_string(original_handler.to_string());
//
//  // deserialized has the same paremters
//  expect_params_equal(original, new_handler.get_params());
//
//  // And the same parameter vector (ie, order was preserved)
//  expect_parameter_vector_equal(original_param_vector,
//                                new_handler.get_params_as_vector());
//}
}
