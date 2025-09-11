/*
 * Copyright (C) 2023 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include <algorithm>
#include <random>

#include <albatross/Common>

#include <gtest/gtest.h>

static constexpr std::size_t kMaxInputs = 1000;
static constexpr std::size_t kNumIterations = 100;
static constexpr std::size_t kSeed = 22;
static constexpr int zstd_min_level = 0;
static constexpr int zstd_max_level = 20;

TEST(Compress, StringRoundtrips) {
  std::default_random_engine generator(kSeed);
  std::uniform_int_distribution<std::size_t> size_distribution(0, kMaxInputs);
  std::uniform_int_distribution<int8_t> distribution('A', 'z');
  for (std::size_t i = 0; i < kNumIterations; ++i) {
    std::string inputs(size_distribution(generator), '\0');
    std::generate(inputs.begin(), inputs.end(), [&generator, &distribution]() {
      return distribution(generator);
    });
    const auto compressed = albatross::zstd::compress(inputs);
    const auto outputs = albatross::zstd::decompress(compressed);
    EXPECT_EQ(inputs, outputs);
    std::string maybe_outputs;
    const auto result =
        albatross::zstd::maybe_decompress(compressed, &maybe_outputs);
    EXPECT_TRUE(result);
    EXPECT_EQ(inputs, maybe_outputs);
  }
}

TEST(Compress, StringRoundtripsAlternateCompressionLevels) {
  std::default_random_engine generator(kSeed);
  std::uniform_int_distribution<std::size_t> size_distribution(0, kMaxInputs);
  std::uniform_int_distribution<int8_t> distribution('A', 'z');
  for (std::size_t i = 0; i < kNumIterations; ++i) {
    for (int level = zstd_min_level; level <= zstd_max_level; ++level) {
      std::string inputs(size_distribution(generator), '\0');
      std::generate(
          inputs.begin(), inputs.end(),
          [&generator, &distribution]() { return distribution(generator); });
      const auto compressed = albatross::zstd::compress(inputs, level);
      const auto outputs = albatross::zstd::decompress(compressed);
      EXPECT_EQ(inputs, outputs);
      std::string maybe_outputs;
      const auto result =
          albatross::zstd::maybe_decompress(compressed, &maybe_outputs);
      EXPECT_TRUE(result);
      EXPECT_EQ(inputs, maybe_outputs);
    }
  }
}

TEST(Compress, DecompressEmpty) {
  std::string inputs;
  ASSERT_DEATH({ const auto result = albatross::zstd::decompress(inputs); },
               "error determining");
}

TEST(Compress, DecompressInvalidZstd) {
  std::string inputs = "albatross";
  ASSERT_DEATH({ const auto result = albatross::zstd::decompress(inputs); },
               "error determining");
}

TEST(Compress, MaybeDecompressEmpty) {
  std::string inputs;
  std::string outputs;
  const auto result = albatross::zstd::maybe_decompress(inputs, &outputs);
  EXPECT_FALSE(result);
}

TEST(Compress, MaybeDecompressInvalidZstd) {
  std::string inputs = "albatross";
  std::string outputs;
  const auto result = albatross::zstd::maybe_decompress(inputs, &outputs);
  EXPECT_FALSE(result);
}

template <typename Scalar>
class Array : public ::testing::Test {
 public:
  std::vector<Scalar> roundtrip(const std::vector<Scalar> &input) {
    const auto compressed =
        albatross::zstd::compress(input.data(), input.size());
    std::vector<Scalar> output(input.size());
    albatross::zstd::decompress(compressed, output.data(), output.size());
    return output;
  }

  std::vector<Scalar> roundtrip_wrong_size(const std::vector<Scalar> &input) {
    const auto compressed =
        albatross::zstd::compress(input.data(), input.size());
    std::vector<Scalar> output(input.size() / 2);
    albatross::zstd::decompress(compressed, output.data(), output.size());
    return output;
  }
};

template <typename Scalar>
class IntegerArray : public Array<Scalar> {};

TYPED_TEST_SUITE_P(IntegerArray);

TYPED_TEST_P(IntegerArray, IntegerRoundtrips) {
  std::default_random_engine generator(kSeed);
  std::uniform_int_distribution<std::size_t> size_distribution(0, kMaxInputs);
  std::uniform_int_distribution<TypeParam> distribution(
      std::numeric_limits<TypeParam>::min(),
      std::numeric_limits<TypeParam>::max());
  for (std::size_t i = 0; i < kNumIterations; ++i) {
    std::vector<TypeParam> inputs(size_distribution(generator));
    std::generate(inputs.begin(), inputs.end(), [&generator, &distribution]() {
      return distribution(generator);
    });
    const auto outputs = this->roundtrip(inputs);
    EXPECT_EQ(inputs, outputs);
  }
}

TYPED_TEST_P(IntegerArray, IntegerWrongSizeAsserts) {
  std::default_random_engine generator(kSeed);
  std::uniform_int_distribution<std::size_t> size_distribution(3, kMaxInputs);
  std::uniform_int_distribution<TypeParam> distribution(
      std::numeric_limits<TypeParam>::min(),
      std::numeric_limits<TypeParam>::max());
  for (std::size_t i = 0; i < kNumIterations; ++i) {
    std::vector<TypeParam> inputs(size_distribution(generator));
    std::generate(inputs.begin(), inputs.end(), [&generator, &distribution]() {
      return distribution(generator);
    });
    ASSERT_DEATH({ const auto outputs = this->roundtrip_wrong_size(inputs); },
                 "zstd expected decompressed size");
  }
}

REGISTER_TYPED_TEST_SUITE_P(IntegerArray, IntegerRoundtrips,
                            IntegerWrongSizeAsserts);

using IntegralTypes =
    ::testing::Types<std::uint8_t, std::int8_t, std::uint16_t, std::int16_t,
                     std::uint32_t, std::int32_t, std::uint64_t, std::int64_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(Compress, IntegerArray, IntegralTypes);

template <typename Scalar>
class FloatingArray : public Array<Scalar> {};

TYPED_TEST_SUITE_P(FloatingArray);

TYPED_TEST_P(FloatingArray, FloatingRoundtrips) {
  std::default_random_engine generator(kSeed);
  std::uniform_int_distribution<std::size_t> size_distribution(0, kMaxInputs);
  std::uniform_real_distribution<TypeParam> distribution(
      std::numeric_limits<TypeParam>::min(),
      std::numeric_limits<TypeParam>::max());
  for (std::size_t i = 0; i < kNumIterations; ++i) {
    std::vector<TypeParam> inputs(size_distribution(generator));
    std::generate(inputs.begin(), inputs.end(), [&generator, &distribution]() {
      return distribution(generator);
    });
    const auto outputs = this->roundtrip(inputs);
    EXPECT_EQ(inputs, outputs);
  }
}

TYPED_TEST_P(FloatingArray, FloatingWrongSizeAsserts) {
  std::default_random_engine generator(kSeed);
  std::uniform_int_distribution<std::size_t> size_distribution(3, kMaxInputs);
  std::uniform_real_distribution<TypeParam> distribution(
      std::numeric_limits<TypeParam>::min(),
      std::numeric_limits<TypeParam>::max());
  for (std::size_t i = 0; i < kNumIterations; ++i) {
    std::vector<TypeParam> inputs(size_distribution(generator));
    std::generate(inputs.begin(), inputs.end(), [&generator, &distribution]() {
      return distribution(generator);
    });
    ASSERT_DEATH({ const auto outputs = this->roundtrip_wrong_size(inputs); },
                 "expected decompressed size");
  }
}

REGISTER_TYPED_TEST_SUITE_P(FloatingArray, FloatingRoundtrips,
                            FloatingWrongSizeAsserts);

using FloatingTypes = ::testing::Types<float, double>;
INSTANTIATE_TYPED_TEST_SUITE_P(Compress, FloatingArray, FloatingTypes);
