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

#include <albatross/cereal/traits.h>
#include <gtest/gtest.h>

namespace albatross {

struct NullArchive {};

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

TEST(test_traits_cereal, test_valid_in_out_serializer) {
  EXPECT_TRUE(
      bool(valid_in_out_serializer<ValidInOutSerializer, NullArchive>::value));
  EXPECT_TRUE(bool(
      valid_in_out_serializer<ValidSaveLoadSerializer, NullArchive>::value));
  EXPECT_FALSE(
      bool(valid_in_out_serializer<ValidInSerializer, NullArchive>::value));
  EXPECT_FALSE(
      bool(valid_in_out_serializer<ValidOutSerializer, NullArchive>::value));
  EXPECT_FALSE(bool(
      valid_in_out_serializer<InValidInOutSerializer, NullArchive>::value));
}

TEST(test_traits_cereal, test_valid_input_serializer) {
  EXPECT_TRUE(
      bool(valid_input_serializer<ValidInOutSerializer, NullArchive>::value));
  EXPECT_TRUE(bool(
      valid_input_serializer<ValidSaveLoadSerializer, NullArchive>::value));
  EXPECT_TRUE(
      bool(valid_input_serializer<ValidInSerializer, NullArchive>::value));
  EXPECT_FALSE(
      bool(valid_input_serializer<ValidOutSerializer, NullArchive>::value));
  EXPECT_FALSE(
      bool(valid_input_serializer<InValidInOutSerializer, NullArchive>::value));
}

TEST(test_traits_cereal, test_valid_output_serializer) {
  EXPECT_TRUE(
      bool(valid_output_serializer<ValidInOutSerializer, NullArchive>::value));
  EXPECT_TRUE(bool(
      valid_output_serializer<ValidSaveLoadSerializer, NullArchive>::value));
  EXPECT_FALSE(
      bool(valid_output_serializer<ValidInSerializer, NullArchive>::value));
  EXPECT_TRUE(
      bool(valid_output_serializer<ValidOutSerializer, NullArchive>::value));
  EXPECT_FALSE(bool(
      valid_output_serializer<InValidInOutSerializer, NullArchive>::value));
}

} // namespace albatross
