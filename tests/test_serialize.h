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

namespace albatross {

template <typename X> struct SerializableType {
  using RepresentationType = X;
  virtual RepresentationType create() const {
    RepresentationType obj;
    return obj;
  }
  virtual bool are_equal(const X &lhs, const X &rhs) const {
    return lhs == rhs;
  };
};

template <typename Serializable> struct SerializeTest : public ::testing::Test {
  typedef typename Serializable::RepresentationType Representation;
};

TYPED_TEST_CASE_P(SerializeTest);

template <typename InputArchiveType, typename OutputArchiveType,
          typename SerializableTestType>
void expect_roundtrip_serializable() {
  SerializableTestType model_and_rep;
  using X = typename SerializableTestType::RepresentationType;
  const X original = model_and_rep.create();

  // Serialize it
  std::ostringstream os;
  {
    OutputArchiveType oarchive(os);
    oarchive(original);
  }
  // Deserialize it.
  std::istringstream is(os.str());
  X deserialized;
  {
    InputArchiveType iarchive(is);
    iarchive(deserialized);
  }
  // Make sure the original and deserialized representations are
  // equivalent.
  EXPECT_TRUE(model_and_rep.are_equal(original, deserialized));
  // Reserialize the deserialized object
  std::ostringstream os_again;
  {
    OutputArchiveType oarchive(os_again);
    oarchive(deserialized);
  }
  // And make sure the serialized strings are the same,
  EXPECT_EQ(os_again.str(), os.str());
}

TYPED_TEST_P(SerializeTest, test_roundtrip_serialize_json) {
  expect_roundtrip_serializable<cereal::JSONInputArchive,
                                cereal::JSONOutputArchive, TypeParam>();
}

TYPED_TEST_P(SerializeTest, test_roundtrip_serialize_binary) {
  expect_roundtrip_serializable<cereal::BinaryInputArchive,
                                cereal::BinaryOutputArchive, TypeParam>();
}
}
