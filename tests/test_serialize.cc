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

#include "core/model.h"
#include "core/serialize.h"
#include <gtest/gtest.h>

#include "test_utils.h"

#include "models/gp.h"
#include <cereal/archives/json.hpp>
#include <cereal/types/polymorphic.hpp>

CEREAL_REGISTER_TYPE_WITH_NAME(albatross::MockModel, "mock_model_name");

namespace albatross {

using SqrExp = SquaredExponential<ScalarDistance>;
using Noise = IndependentNoise<double>;
using SqrExpAndNoise = SumOfCovarianceTerms<SqrExp, Noise>;
using CovFunc = CovarianceFunction<SqrExpAndNoise>;
using SquaredExponentialGaussianProcess =
    GaussianProcessRegression<double, CovFunc>;

/*
 * Make sure we can serialize a model and recover the parameters.
 */
TEST(test_serialize, test_serialize_model_roundtrip) {
  MockModel m(log(2.));
  MockModel roundtrip;
  std::ostringstream oss;
  {
    cereal::JSONOutputArchive archive(oss);
    archive(m);
  }
  std::istringstream iss(oss.str());
  {
    cereal::JSONInputArchive archive(iss);
    archive(roundtrip);
  }
  EXPECT_EQ(roundtrip, m);
};

/*
 * Tests to make sure we can serialize from one parameter handler to
 * another.
 */
TEST(test_serialize, test_serialize_parameter_store_roundtrip) {
  const ParameterStore original = {{"2", 2.}, {"1", 1.}, {"3", 3.}};
  MockParameterHandler original_handler(original);
  std::ostringstream oss;
  {
    cereal::JSONOutputArchive archive(oss);
    archive(original_handler);
  }
  // Make another handler that starts with different parameters
  MockParameterHandler new_handler({{"2", 4.}, {"1", 5.}, {"3", 6.}});
  std::istringstream iss(oss.str());
  {
    cereal::JSONInputArchive archive(iss);
    archive(new_handler);
  }
  // deserialized has the same paremters
  expect_params_equal(original, new_handler.get_params());
}

/*
 * In what follows we set up a series of test cases in which vary how
 * a model we want to serialize is created and represented.  For example
 * we may want to make sure that a model which is fit to some data
 * and passed around in a pointer can be serialized and deserialized.
 * Ie, something like,
 *
 *   std::unique_ptr<RegressionModel<MockPredictor>> m;
 *   m.fit(dataset)
 *   std::unique_ptr<RegressionModel<MockPredictor>> actual = roundtrip(m);
 *   EXPECT_EQ(actual, m)
 *
 * same for a model that hasn't been fit first, etc ...
 *
 * To do so we create an abstract test case using the ModelRepresentation class,
 * create Specific variants of it, then run the same tests on all
 * the variants using TYPED_TEST.
 */

template <typename X> class ModelRepresentation {
public:
  typedef X RepresentationType;
  virtual RepresentationType create() const = 0;
  virtual bool are_equal(const X &lhs, const X &rhs) const {
    return lhs == rhs;
  };
};

using SerializableMockPointer =
    std::unique_ptr<SerializableRegressionModel<MockPredictor, MockFit>>;
using RegressionMockPointer = std::unique_ptr<RegressionModel<MockPredictor>>;
using SerializableLeastSquaresPointer =
    std::unique_ptr<SerializableRegressionModel<double, LeastSquaresFit>>;
using DoubleRegressionModelPointer = std::unique_ptr<RegressionModel<double>>;

class UnfitSerializableModel
    : public ModelRepresentation<SerializableMockPointer> {
public:
  RepresentationType create() const override {
    return std::make_unique<MockModel>(log(2.));
  }

  bool are_equal(const RepresentationType &lhs,
                 const RepresentationType &rhs) const override {
    return *lhs == *rhs;
  };
};

class FitSerializableModel
    : public ModelRepresentation<SerializableMockPointer> {
public:
  RepresentationType create() const override {
    auto dataset = mock_training_data();
    auto model_ptr = std::make_unique<MockModel>(log(2.));
    model_ptr->fit(dataset);
    return model_ptr;
  }

  bool are_equal(const RepresentationType &lhs,
                 const RepresentationType &rhs) const override {
    return *lhs == *rhs && lhs->get_fit() == rhs->get_fit();
  };
};

class FitDirectModel : public ModelRepresentation<MockModel> {
public:
  RepresentationType create() const override {
    auto dataset = mock_training_data();
    auto model = MockModel(log(2.));
    model.fit(dataset);
    return model;
  }

  bool are_equal(const RepresentationType &lhs,
                 const RepresentationType &rhs) const override {
    return lhs == rhs && lhs.get_fit() == rhs.get_fit();
  };
};

class UnfitDirectModel : public ModelRepresentation<MockModel> {
public:
  RepresentationType create() const override { return MockModel(log(2.)); }
  bool are_equal(const RepresentationType &lhs,
                 const RepresentationType &rhs) const override {
    return lhs == rhs && lhs.get_fit() == rhs.get_fit();
  };
};

class UnfitRegressionModel : public ModelRepresentation<RegressionMockPointer> {
public:
  RepresentationType create() const override {
    return std::make_unique<MockModel>(log(2.));
  }
  bool are_equal(const RepresentationType &lhs,
                 const RepresentationType &rhs) const override {
    return *lhs == *rhs;
  };
};

class FitLinearRegressionModel : public ModelRepresentation<LinearRegression> {
public:
  RepresentationType create() const override {
    auto model = LinearRegression();
    auto dataset = make_toy_linear_data();
    model.fit(dataset);
    return model;
  }

  bool are_equal(const RepresentationType &lhs,
                 const RepresentationType &rhs) const override {
    return lhs.get_fit() == rhs.get_fit();
  };
};

class FitLinearSerializablePointer
    : public ModelRepresentation<SerializableLeastSquaresPointer> {
public:
  RepresentationType create() const override {
    auto model = std::make_unique<LinearRegression>();
    auto dataset = make_toy_linear_data();
    model->fit(dataset);
    return model;
  }

  bool are_equal(const RepresentationType &lhs,
                 const RepresentationType &rhs) const override {
    return lhs->get_fit() == rhs->get_fit();
  };
};

class UnfitGaussianProcess
    : public ModelRepresentation<
          std::unique_ptr<SquaredExponentialGaussianProcess>> {
public:
  RepresentationType create() const override {
    auto gp =
        std::make_unique<SquaredExponentialGaussianProcess>("custom_name");
    gp->set_param("length_scale", log(2.));
    return gp;
  }

  bool are_equal(const RepresentationType &lhs,
                 const RepresentationType &rhs) const override {
    return *lhs == *rhs;
  };
};

class FitGaussianProcess
    : public ModelRepresentation<
          std::unique_ptr<SquaredExponentialGaussianProcess>> {
public:
  RepresentationType create() const override {

    auto dataset = make_toy_linear_data();
    auto gp =
        std::make_unique<SquaredExponentialGaussianProcess>("custom_name");
    gp->set_param("length_scale", log(2.));
    gp->fit(dataset);
    return gp;
  }

  bool are_equal(const RepresentationType &lhs,
                 const RepresentationType &rhs) const override {
    return *lhs == *rhs && lhs->get_fit() == rhs->get_fit();
  };
};

template <typename ModelRepresentationType>
class PolymorphicSerializeTest : public ::testing::Test {
public:
  typedef typename ModelRepresentationType::RepresentationType Representation;
};

typedef ::testing::Types<UnfitSerializableModel, UnfitRegressionModel,
                         FitSerializableModel, FitDirectModel, UnfitDirectModel,
                         FitLinearRegressionModel, FitLinearSerializablePointer,
                         UnfitGaussianProcess, FitGaussianProcess>
    ModelsAndRepresentations;
TYPED_TEST_CASE(PolymorphicSerializeTest, ModelsAndRepresentations);

TYPED_TEST(PolymorphicSerializeTest, test_roundtrip_serialize) {
  TypeParam model_and_rep;
  typename TestFixture::Representation original = model_and_rep.create();
  // Serialize it
  std::ostringstream os;
  {
    cereal::JSONOutputArchive oarchive(os);
    oarchive(original);
  }
  // Deserialize it.
  std::istringstream is(os.str());
  typename TestFixture::Representation deserialized;
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
} // namespace albatross
