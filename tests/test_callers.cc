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

#include <albatross/CovarianceFunctions>

#include <gtest/gtest.h>

#include "test_covariance_utils.h"

namespace albatross {

template <typename T> using Identity = T;

template <typename Caller, template <typename T> class XWrapper = Identity,
          template <typename T> class YWrapper = Identity>
inline void expect_direct_calls_true() {

  EXPECT_TRUE(
      bool(caller_has_valid_call<Caller, HasMultipleMean, XWrapper<X>>::value));
  EXPECT_TRUE(
      bool(caller_has_valid_call<Caller, HasMultipleMean, XWrapper<Y>>::value));
  EXPECT_TRUE(
      bool(caller_has_valid_call<Caller, HasMultipleMean, XWrapper<W>>::value));

  EXPECT_TRUE(bool(caller_has_valid_call<Caller, HasMultiple, XWrapper<X>,
                                         YWrapper<X>>::value));
  EXPECT_TRUE(bool(caller_has_valid_call<Caller, HasMultiple, XWrapper<X>,
                                         YWrapper<Y>>::value));
  EXPECT_TRUE(bool(caller_has_valid_call<Caller, HasMultiple, XWrapper<Y>,
                                         YWrapper<Y>>::value));
  EXPECT_TRUE(bool(caller_has_valid_call<Caller, HasMultiple, XWrapper<W>,
                                         YWrapper<W>>::value));
}

template <typename Caller, template <typename T> class XWrapper = Identity,
          template <typename T> class YWrapper = Identity>
inline void expect_symmetric_calls_true() {
  expect_direct_calls_true<Caller, XWrapper, YWrapper>();
  EXPECT_TRUE(bool(caller_has_valid_call<Caller, HasMultiple, XWrapper<Y>,
                                         YWrapper<X>>::value));
}

template <typename Caller, template <typename T> class XWrapper = Identity,
          template <typename T> class YWrapper = Identity>
inline void expect_all_calls_false() {

  EXPECT_FALSE(
      bool(caller_has_valid_call<Caller, HasMultipleMean, XWrapper<Z>>::value));

  EXPECT_FALSE(
      bool(caller_has_valid_call<Caller, HasMultipleMean, XWrapper<V>>::value));

  EXPECT_FALSE(bool(caller_has_valid_call<Caller, HasMultiple, XWrapper<X>,
                                          YWrapper<W>>::value));
  EXPECT_FALSE(bool(caller_has_valid_call<Caller, HasMultiple, XWrapper<Y>,
                                          YWrapper<W>>::value));
  EXPECT_FALSE(bool(caller_has_valid_call<Caller, HasMultiple, XWrapper<W>,
                                          YWrapper<Y>>::value));
  EXPECT_FALSE(bool(caller_has_valid_call<Caller, HasMultiple, XWrapper<Z>,
                                          YWrapper<Z>>::value));
  EXPECT_FALSE(bool(caller_has_valid_call<Caller, HasMultiple, XWrapper<X>,
                                          YWrapper<Z>>::value));
  EXPECT_FALSE(bool(caller_has_valid_call<Caller, HasMultiple, XWrapper<Y>,
                                          YWrapper<Z>>::value));
  EXPECT_FALSE(bool(caller_has_valid_call<Caller, HasMultiple, XWrapper<W>,
                                          YWrapper<Z>>::value));
}

TEST(test_callers, test_direct_caller) {
  expect_direct_calls_true<internal::DirectCaller>();
  expect_all_calls_false<internal::DirectCaller>();
  EXPECT_FALSE(bool(
      caller_has_valid_call<internal::DirectCaller, HasMultiple, Y, X>::value));
}

TEST(test_callers, test_symmetric_caller) {
  using Symmetric = internal::SymmetricCaller<internal::DirectCaller>;

  expect_symmetric_calls_true<Symmetric>();
  expect_all_calls_false<Symmetric>();
}

TEST(test_callers, test_measurement_caller) {
  using MeasCaller = internal::MeasurementForwarder<
      internal::SymmetricCaller<internal::DirectCaller>>;

  expect_symmetric_calls_true<MeasCaller, Identity, Identity>();
  expect_symmetric_calls_true<MeasCaller, Measurement, Identity>();
  expect_symmetric_calls_true<MeasCaller, Identity, Measurement>();
  expect_symmetric_calls_true<MeasCaller, Measurement, Measurement>();

  expect_all_calls_false<MeasCaller, Identity, Identity>();
  expect_all_calls_false<MeasCaller, Measurement, Identity>();
  expect_all_calls_false<MeasCaller, Identity, Measurement>();
  expect_all_calls_false<MeasCaller, Measurement, Measurement>();
}

TEST(test_callers, test_linear_combination_caller) {
  using Caller = internal::LinearCombinationCaller<
      internal::SymmetricCaller<internal::DirectCaller>>;

  HasMultiple cov_func;

  LinearCombination<X> two_xs({X(), X()});
  X x;
  Y y;

  const auto one_x = Caller::call(cov_func, x, x);
  const auto two_x = Caller::call(cov_func, x, two_xs);
  EXPECT_EQ(two_x, 2 * one_x);
  const auto four_x = Caller::call(cov_func, two_xs, two_xs);
  EXPECT_EQ(four_x, 4 * one_x);

  const auto one_xy = Caller::call(cov_func, y, x);
  const auto two_xy = Caller::call(cov_func, y, two_xs);
  EXPECT_EQ(two_xy, 2 * one_xy);
}

template <typename T, typename VariantType> struct VariantOrRaw {

  template <typename C,
            typename std::enable_if<!is_in_variant<C, VariantType>::value,
                                    int>::type = 0>
  static T test(C *);

  template <typename> static VariantType test(...);

public:
  typedef decltype(test<T>(0)) type;
};

template <typename T>
using VariantXY = typename VariantOrRaw<T, variant<X, Y>>::type;

TEST(test_callers, test_variant_caller_XY) {
  using VariantCaller = internal::VariantForwarder<
      internal::SymmetricCaller<internal::DirectCaller>>;

  expect_symmetric_calls_true<VariantCaller>();
  expect_symmetric_calls_true<VariantCaller, VariantXY, Identity>();
  expect_symmetric_calls_true<VariantCaller, Identity, VariantXY>();
  expect_symmetric_calls_true<VariantCaller, VariantXY, VariantXY>();

  expect_all_calls_false<VariantCaller>();
}

template <typename T>
using VariantXW = typename VariantOrRaw<T, variant<X, W>>::type;

TEST(test_callers, test_variant_caller_XW) {
  using VariantCaller = internal::VariantForwarder<
      internal::SymmetricCaller<internal::DirectCaller>>;

  expect_symmetric_calls_true<VariantCaller>();
  expect_symmetric_calls_true<VariantCaller, VariantXW, Identity>();
  expect_symmetric_calls_true<VariantCaller, Identity, VariantXW>();
  expect_symmetric_calls_true<VariantCaller, VariantXW, VariantXW>();

  expect_all_calls_false<VariantCaller>();
}

template <typename T>
using VariantXYW = typename VariantOrRaw<T, variant<X, Y, W>>::type;

TEST(test_callers, test_variant_caller_XYW) {
  using VariantCaller = internal::VariantForwarder<
      internal::SymmetricCaller<internal::DirectCaller>>;

  expect_symmetric_calls_true<VariantCaller>();
  expect_symmetric_calls_true<VariantCaller, VariantXYW, Identity>();
  expect_symmetric_calls_true<VariantCaller, Identity, VariantXYW>();
  expect_symmetric_calls_true<VariantCaller, VariantXYW, VariantXYW>();
}

/*
 * This Makes sure you can call with variants of different types.
 */
TEST(test_callers, test_variant_caller_XYW_XY) {
  using VariantCaller = internal::VariantForwarder<
      internal::SymmetricCaller<internal::DirectCaller>>;

  expect_symmetric_calls_true<VariantCaller, VariantXYW, VariantXY>();
  expect_symmetric_calls_true<VariantCaller, VariantXY, VariantXYW>();
}

template <typename T>
using VariantWithMeasurement =
    typename VariantOrRaw<T, variant<Measurement<X>, Y>>::type;

TEST(test_callers, test_variant_caller_with_measurement) {
  expect_symmetric_calls_true<DefaultCaller>();
  expect_symmetric_calls_true<DefaultCaller, VariantWithMeasurement,
                              Identity>();
  expect_symmetric_calls_true<DefaultCaller, Identity,
                              VariantWithMeasurement>();
  expect_symmetric_calls_true<DefaultCaller, VariantWithMeasurement,
                              VariantWithMeasurement>();

  expect_all_calls_false<DefaultCaller>();
}

} // namespace albatross
