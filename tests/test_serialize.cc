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
#include <functional>
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

using SerializableMockPointer =
    std::unique_ptr<SerializableRegressionModel<MockPredictor, MockFit>>;
using RegressionMockPointer = std::unique_ptr<RegressionModel<MockPredictor>>;
using SerializableLeastSquaresPointer =
    std::unique_ptr<SerializableRegressionModel<double, LeastSquaresFit>>;
using DoubleRegressionModelPointer = std::unique_ptr<RegressionModel<double>>;

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
 * To do so we create an abstract test case using the SerializableType
 * and create Specific variants of it, then run the same tests on all
 * the variants using TYPED_TEST.
 */

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

struct EmptyEigenVectorXd : public SerializableType<Eigen::VectorXd> {
  Eigen::VectorXd create() const override {
    Eigen::VectorXd x;
    return x;
  }
};

struct EigenVectorXd : public SerializableType<Eigen::VectorXd> {
  Eigen::VectorXd create() const override {
    Eigen::VectorXd x(2);
    x << 1., 2.;
    return x;
  }
};

struct EmptyEigenMatrixXd : public SerializableType<Eigen::MatrixXd> {
  Eigen::MatrixXd create() const override {
    Eigen::MatrixXd x;
    return x;
  }
};

struct EigenMatrixXd : public SerializableType<Eigen::MatrixXd> {
  Eigen::MatrixXd create() const override {
    Eigen::MatrixXd x(2, 3);
    x << 1., 2., 3., 4., 5., 6.;
    return x;
  }
};

struct LDLT
    : public SerializableType<Eigen::SerializableLDLT<Eigen::MatrixXd>> {
  Eigen::Index n = 3;

  RepresentationType create() const override {
    auto part = Eigen::MatrixXd::Random(n, n);
    auto cov = part * part.transpose();
    auto ldlt = cov.ldlt();
    auto information = Eigen::VectorXd::Ones(n);

    // Make sure our two LDLT objects behave the same.
    RepresentationType serializable_ldlt(ldlt);
    EXPECT_EQ(ldlt.solve(information), serializable_ldlt.solve(information));

    return serializable_ldlt;
  }

  bool are_equal(const RepresentationType &lhs,
                 const RepresentationType &rhs) const override {
    auto information = Eigen::VectorXd::Ones(n);
    return (lhs == rhs && lhs.solve(information) == rhs.solve(information));
  };
};

struct ParameterStoreType : public SerializableType<ParameterStore> {

  RepresentationType create() const override {
    ParameterStore original = {{"2", 2.}, {"1", 1.}, {"3", 3.}};
    return original;
  }

  bool are_equal(const RepresentationType &lhs,
                 const RepresentationType &rhs) const override {
    expect_params_equal(lhs, rhs);
    return true;
  };
};

class UnfitSerializableModel
    : public SerializableType<SerializableMockPointer> {
public:
  RepresentationType create() const override {
    return std::make_unique<MockModel>(log(2.));
  }

  bool are_equal(const RepresentationType &lhs,
                 const RepresentationType &rhs) const override {
    return *lhs == *rhs;
  };
};

class FitSerializableModel : public SerializableType<SerializableMockPointer> {
public:
  RepresentationType create() const override {
    auto dataset = mock_training_data();
    auto model_ptr = std::make_unique<MockModel>(log(2.));
    model_ptr->fit(dataset);
    return std::move(model_ptr);
  }

  bool are_equal(const RepresentationType &lhs,
                 const RepresentationType &rhs) const override {
    return *lhs == *rhs && lhs->get_fit() == rhs->get_fit();
  };
};

class FitDirectModel : public SerializableType<MockModel> {
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

class UnfitDirectModel : public SerializableType<MockModel> {
public:
  RepresentationType create() const override { return MockModel(log(2.)); }
  bool are_equal(const RepresentationType &lhs,
                 const RepresentationType &rhs) const override {
    return lhs == rhs && lhs.get_fit() == rhs.get_fit();
  };
};

class UnfitRegressionModel : public SerializableType<RegressionMockPointer> {
public:
  RepresentationType create() const override {
    return std::make_unique<MockModel>(log(2.));
  }
  bool are_equal(const RepresentationType &lhs,
                 const RepresentationType &rhs) const override {
    return *lhs == *rhs;
  };
};

class FitLinearRegressionModel : public SerializableType<LinearRegression> {
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
    : public SerializableType<SerializableLeastSquaresPointer> {
public:
  RepresentationType create() const override {
    auto model = std::make_unique<LinearRegression>();
    auto dataset = make_toy_linear_data();
    model->fit(dataset);
    return std::move(model);
  }

  bool are_equal(const RepresentationType &lhs,
                 const RepresentationType &rhs) const override {
    return lhs->get_fit() == rhs->get_fit();
  };
};

class UnfitGaussianProcess
    : public SerializableType<
          std::unique_ptr<SquaredExponentialGaussianProcess>> {
public:
  RepresentationType create() const override {
    auto gp =
        std::make_unique<SquaredExponentialGaussianProcess>("custom_name");
    gp->set_param("length_scale", log(2.));
    return std::move(gp);
  }

  bool are_equal(const RepresentationType &lhs,
                 const RepresentationType &rhs) const override {
    return *lhs == *rhs;
  };
};

class FitGaussianProcess
    : public SerializableType<
          std::unique_ptr<SquaredExponentialGaussianProcess>> {
public:
  RepresentationType create() const override {

    auto dataset = make_toy_linear_data();
    auto gp =
        std::make_unique<SquaredExponentialGaussianProcess>("custom_name");
    gp->set_param("length_scale", log(2.));
    gp->fit(dataset);
    return std::move(gp);
  }

  bool are_equal(const RepresentationType &lhs,
                 const RepresentationType &rhs) const override {
    return *lhs == *rhs && lhs->get_fit() == rhs->get_fit();
  };
};

template <typename X>
X expect_roundtrip_serializable(
    const X &original,
    const std::function<bool(const X &, const X &)> &compare) {
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
  EXPECT_TRUE(compare(original, deserialized));
  // Reserialize the deserialized object
  std::ostringstream os_again;
  {
    cereal::JSONOutputArchive oarchive(os_again);
    oarchive(deserialized);
  }
  // And make sure the serialized strings are the same,
  EXPECT_EQ(os_again.str(), os.str());
  return deserialized;
}

template <typename X> X expect_roundtrip_serializable(const X &original) {
  std::function<bool(const X &, const X &)> equality =
      [](const X &lhs, const X &rhs) { return lhs == rhs; };
  return expect_roundtrip_serializable(original, equality);
}

template <typename Serializable>
struct PolymorphicSerializeTest : public ::testing::Test {
  typedef typename Serializable::RepresentationType Representation;
};

typedef ::testing::Types<
    LDLT, SerializableType<Eigen::Vector3d>, SerializableType<Eigen::Matrix3d>,
    SerializableType<Eigen::Matrix2i>, EmptyEigenVectorXd, EigenVectorXd,
    EmptyEigenMatrixXd, EigenMatrixXd, ParameterStoreType,
    SerializableType<MockModel>, UnfitSerializableModel, FitSerializableModel,
    FitDirectModel, UnfitDirectModel, UnfitRegressionModel,
    FitLinearRegressionModel, FitLinearSerializablePointer,
    UnfitGaussianProcess, FitGaussianProcess>
    ToTest;

TYPED_TEST_CASE(PolymorphicSerializeTest, ToTest);

TYPED_TEST(PolymorphicSerializeTest, test_roundtrip_serialize) {
  TypeParam model_and_rep;
  using X = typename TypeParam::RepresentationType;
  const X original = model_and_rep.create();
  std::function<bool(const X &, const X &)> compare =
      std::bind(&TypeParam::are_equal, model_and_rep, std::placeholders::_1,
                std::placeholders::_2);
  expect_roundtrip_serializable(original, compare);
}
} // namespace albatross
