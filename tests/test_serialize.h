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
#include <cereal/archives/json.hpp>
#include <cereal/types/polymorphic.hpp>

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

TYPED_TEST_P(SerializeTest, test_roundtrip_serialize_json) {
  TypeParam model_and_rep;
  using X = typename TypeParam::RepresentationType;
  const X original = model_and_rep.create();

  // Serialize it
  std::ostringstream os;
  {
    cereal::JSONOutputArchive oarchive(os);
    oarchive(original);
  }
  // Deserialize it.
  std::istringstream is(os.str());
  X deserialized;
  {
    cereal::JSONInputArchive iarchive(is);
    iarchive(deserialized);
  }
  // Make sure the original and deserialized representations are
  // equivalent.
  EXPECT_TRUE(model_and_rep.are_equal(original, deserialized));
  // Reserialize the deserialized object
  std::ostringstream os_again;
  {
    cereal::JSONOutputArchive oarchive(os_again);
    oarchive(deserialized);
  }
  // And make sure the serialized strings are the same,
  EXPECT_EQ(os_again.str(), os.str());
}

TYPED_TEST_P(SerializeTest, test_roundtrip_serialize_binary) {
  TypeParam model_and_rep;
  using X = typename TypeParam::RepresentationType;
  const X original = model_and_rep.create();

  // Serialize it
  std::ostringstream os;
  {
    cereal::BinaryOutputArchive oarchive(os);
    oarchive(original);
  }
  // Deserialize it.
  std::istringstream is(os.str());
  X deserialized;
  {
    cereal::BinaryInputArchive iarchive(is);
    iarchive(deserialized);
  }
  // Make sure the original and deserialized representations are
  // equivalent.
  EXPECT_TRUE(model_and_rep.are_equal(original, deserialized));
  // Reserialize the deserialized object
  std::ostringstream os_again;
  {
    cereal::BinaryOutputArchive oarchive(os_again);
    oarchive(deserialized);
  }
  // And make sure the serialized strings are the same,
  EXPECT_EQ(os_again.str(), os.str());
}
}
