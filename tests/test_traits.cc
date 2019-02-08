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

#include "core/traits.h"
#include <gtest/gtest.h>

namespace albatross {

struct X {};
struct Y {};
struct Z {};

class HasPublicCallOperator {
public:
  double operator()(const X &, const Y &) const { return 1.; };
};

class HasProtectedCallOperator {
protected:
  double operator()(const X &, const Y &) const { return 1.; };
};

class HasPrivateCallOperator {
  double operator()(const X &, const Y &) const { return 1.; };
};

class HasNoCallOperator {};

TEST(test_traits, test_has_call_operator) {
  EXPECT_TRUE(bool(has_call_operator<HasPublicCallOperator, X, Y>::value));
  EXPECT_FALSE(bool(has_call_operator<HasPrivateCallOperator, X, Y>::value));
  EXPECT_FALSE(bool(has_call_operator<HasProtectedCallOperator, X, Y>::value));
  EXPECT_FALSE(bool(has_call_operator<HasNoCallOperator, X, Y>::value));
}

class HasPublicCallImpl {
public:
  double call_impl_(const X &, const Y &) const { return 1.; };
};

class HasProtectedCallImpl {
protected:
  double call_impl_(const X &, const Y &) const { return 1.; };
};

class HasPrivateCallImpl {
  double call_impl_(const X &, const Y &) const { return 1.; };
};

class HasNoCallImpl {};

TEST(test_traits, test_has_any_call_impl_) {
  EXPECT_TRUE(bool(has_any_call_impl<HasPublicCallImpl>::value));
  EXPECT_TRUE(bool(has_any_call_impl<HasProtectedCallImpl>::value));
  EXPECT_TRUE(bool(has_any_call_impl<HasPrivateCallImpl>::value));
  EXPECT_FALSE(bool(has_any_call_impl<HasNoCallImpl>::value));
}

class HasMultiplePublicCallImpl {
public:
  double call_impl_(const X &, const Y &) const { return 1.; };

  double call_impl_(const X &, const X &) const { return 1.; };

  double call_impl_(const Y &, const Y &) const { return 1.; };
};

TEST(test_traits, test_has_defined_call_impl_) {
  EXPECT_TRUE(bool(has_defined_call_impl<HasPublicCallImpl, X, Y>::value));
  EXPECT_FALSE(bool(has_defined_call_impl<HasPublicCallImpl, Y, X>::value));
  EXPECT_TRUE(
      bool(has_defined_call_impl<HasMultiplePublicCallImpl, X, X>::value));
  EXPECT_TRUE(
      bool(has_defined_call_impl<HasMultiplePublicCallImpl, Y, Y>::value));
  EXPECT_FALSE(
      bool(has_defined_call_impl<HasMultiplePublicCallImpl, X, Z>::value));
}

class ValidInOutSerializer {
public:
  template <typename Archive> void serialize(Archive &){};
};

class ValidSaveLoadSerializer {
public:
  template <typename Archive> void save(Archive &) const {};

  template <typename Archive> void load(Archive &){};
};

class ValidInSerializer {
public:
  template <typename Archive> void load(Archive &){};
};

class ValidOutSerializer {
public:
  template <typename Archive> void save(Archive &) const {};
};

class InValidInOutSerializer {};

TEST(test_traits, test_valid_in_out_serializer) {
  EXPECT_TRUE(bool(valid_in_out_serializer<ValidInOutSerializer, X>::value));
  EXPECT_TRUE(bool(valid_in_out_serializer<ValidSaveLoadSerializer, X>::value));
  EXPECT_FALSE(bool(valid_in_out_serializer<ValidInSerializer, X>::value));
  EXPECT_FALSE(bool(valid_in_out_serializer<ValidOutSerializer, X>::value));
  EXPECT_FALSE(bool(valid_in_out_serializer<InValidInOutSerializer, X>::value));
}

TEST(test_traits, test_valid_input_serializer) {
  EXPECT_TRUE(bool(valid_input_serializer<ValidInOutSerializer, X>::value));
  EXPECT_TRUE(bool(valid_input_serializer<ValidSaveLoadSerializer, X>::value));
  EXPECT_TRUE(bool(valid_input_serializer<ValidInSerializer, X>::value));
  EXPECT_FALSE(bool(valid_input_serializer<ValidOutSerializer, X>::value));
  EXPECT_FALSE(bool(valid_input_serializer<InValidInOutSerializer, X>::value));
}

TEST(test_traits, test_valid_output_serializer) {
  EXPECT_TRUE(bool(valid_output_serializer<ValidInOutSerializer, X>::value));
  EXPECT_TRUE(bool(valid_output_serializer<ValidSaveLoadSerializer, X>::value));
  EXPECT_FALSE(bool(valid_output_serializer<ValidInSerializer, X>::value));
  EXPECT_TRUE(bool(valid_output_serializer<ValidOutSerializer, X>::value));
  EXPECT_FALSE(bool(valid_output_serializer<InValidInOutSerializer, X>::value));
}

class Complete {};

class Incomplete;

TEST(test_traits, test_is_complete) {
  EXPECT_TRUE(bool(is_complete<Complete>::value));
  EXPECT_FALSE(bool(is_complete<Incomplete>::value));
}

class HasName {
public:
  std::string name_;
};

class HasNoName {};

TEST(test_traits, test_has_name) {
  EXPECT_TRUE(bool(has_name_<HasName>::value));
  EXPECT_FALSE(bool(has_name_<HasNoName>::value));
}

} // namespace albatross
