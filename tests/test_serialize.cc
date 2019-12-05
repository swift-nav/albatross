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

#include <albatross/Common>
#include <albatross/GP>
#include <albatross/Ransac>

#include <albatross/serialize/Common>
#include <albatross/serialize/GP>
#include <albatross/serialize/LeastSquares>
#include <albatross/serialize/Ransac>
#include <albatross/utils/RandomUtils>

#include <gtest/gtest.h>

#include "test_models.h"
#include "test_serialize.h"
#include "test_utils.h"

namespace cereal {

using albatross::Fit;
using albatross::MockFeature;
using albatross::MockModel;

template <typename Archive>
inline void serialize(Archive &archive, MockFeature &f) {
  archive(f.value);
}

template <typename Archive>
inline void serialize(Archive &archive, Fit<MockModel> &f) {
  archive(f.train_data);
}

} // namespace cereal

namespace albatross {

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

struct EigenMatrix3d : public SerializableType<Eigen::Matrix3d> {
  Eigen::Matrix3d create() const override {
    Eigen::Matrix3d x;
    x << 1., 2., 3., 4., 5., 6., 7., 8., 9.;
    return x;
  }
};

struct FullJointDistribution : public SerializableType<JointDistribution> {
  JointDistribution create() const override {
    Eigen::MatrixXd cov(3, 3);
    cov << 1., 2., 3., 4., 5., 6., 7, 8, 9;
    Eigen::VectorXd mean = Eigen::VectorXd::Ones(3);
    return JointDistribution(mean, cov);
  }
};

struct FullMarginalDistribution
    : public SerializableType<MarginalDistribution> {
  MarginalDistribution create() const override {
    Eigen::VectorXd diag = Eigen::VectorXd::Ones(3);
    Eigen::VectorXd mean = Eigen::VectorXd::Ones(3);
    return MarginalDistribution(mean, diag.asDiagonal());
  }
};

struct LDLT : public SerializableType<Eigen::SerializableLDLT> {
  Eigen::Index n = 3;

  RepresentationType create() const override {
    // Without forcing this to a Eigen::MatrixXd
    // type every time `part` is accessed it'll have new random values!
    // so part.tranpose() will not be the transpose of `part` but the
    // transpose of some new random matrix.  Seems like a bug to me.
    const Eigen::MatrixXd part = Eigen::MatrixXd::Random(n, n);
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

struct ExplainedCovarianceRepresentation
    : public SerializableType<ExplainedCovariance> {
  Eigen::Index n = 3;

  RepresentationType create() const override {
    // Without forcing this to a Eigen::MatrixXd
    // type every time `part` is accessed it'll have new random values!
    // so part.tranpose() will not be the transpose of `part` but the
    // transpose of some new random matrix.  Seems like a bug to me.
    const Eigen::MatrixXd part = Eigen::MatrixXd::Random(n, n);
    auto cov = part * part.transpose();

    ExplainedCovariance representation(cov, cov);
    return representation;
  }

  bool are_equal(const RepresentationType &lhs,
                 const RepresentationType &rhs) const override {
    auto information = Eigen::VectorXd::Ones(n);
    return (lhs == rhs && lhs.solve(information) == rhs.solve(information));
  };
};

struct ParameterStoreType : public SerializableType<ParameterStore> {

  RepresentationType create() const override {
    ParameterStore original = {{"1", {1., UninformativePrior()}},
                               {"2", {2., FixedPrior()}},
                               {"3", {3., NonNegativePrior()}},
                               {"4", {4., PositivePrior()}},
                               {"5", {5., UniformPrior()}},
                               {"6", {6., LogScaleUniformPrior()}},
                               {"7", {7., GaussianPrior()}},
                               {"8", {8., LogNormalPrior()}},
                               {"9", 9.}};
    return original;
  }

  bool are_equal(const RepresentationType &lhs,
                 const RepresentationType &rhs) const override {
    return lhs == rhs;
  };
};

struct Dataset : public SerializableType<RegressionDataset<MockFeature>> {

  RepresentationType create() const override {

    std::vector<MockFeature> features = {{1}, {3}, {-2}};
    Eigen::VectorXd targets(3);
    targets << 5., 3., 9.;

    RegressionDataset<MockFeature> dataset(features, targets);

    return dataset;
  }

  bool are_equal(const RepresentationType &lhs,
                 const RepresentationType &rhs) const override {
    return lhs == rhs;
  };
};

struct DatasetWithMetadata
    : public SerializableType<RegressionDataset<MockFeature>> {

  RepresentationType create() const override {

    std::vector<MockFeature> features = {{1}, {3}, {-2}};
    Eigen::VectorXd targets(3);
    targets << 5., 3., 9.;

    RegressionDataset<MockFeature> dataset(features, targets);
    dataset.metadata["time"] = "1";

    return dataset;
  }

  bool are_equal(const RepresentationType &lhs,
                 const RepresentationType &rhs) const override {
    return lhs == rhs;
  };
};

struct VariantAsInt : public SerializableType<variant<int, double>> {

  RepresentationType create() const override {

    variant<int, double> output;
    int foo = 1;
    output = foo;
    return output;
  }
};

struct VariantAsDouble : public SerializableType<variant<int, double>> {

  RepresentationType create() const override {

    variant<int, double> output;
    double foo = 1.;
    output = foo;
    return output;
  }
};

struct BlockSymmetricMatrix
    : public SerializableType<BlockSymmetric<Eigen::SerializableLDLT>> {

  RepresentationType create() const override {

    std::default_random_engine gen(2012);
    const auto X = random_covariance_matrix(5, gen);

    const Eigen::MatrixXd A = X.topLeftCorner(3, 3);
    const Eigen::MatrixXd B = X.topRightCorner(3, 2);
    const Eigen::MatrixXd C = X.bottomRightCorner(2, 2);

    // Test when constructing from the actual blocks.
    return BlockSymmetric<Eigen::SerializableLDLT>(A.ldlt(), B, C);
  }
};

REGISTER_TYPED_TEST_CASE_P(SerializeTest, test_roundtrip_serialize_json,
                           test_roundtrip_serialize_binary);

typedef ::testing::Types<LDLT, ExplainedCovarianceRepresentation, EigenMatrix3d,
                         SerializableType<Eigen::Matrix2i>, EmptyEigenVectorXd,
                         EigenVectorXd, EmptyEigenMatrixXd, EigenMatrixXd,
                         FullJointDistribution, FullMarginalDistribution,
                         ParameterStoreType, Dataset, DatasetWithMetadata,
                         SerializableType<MockModel>, VariantAsInt,
                         VariantAsDouble, BlockSymmetricMatrix>
    ToTest;

INSTANTIATE_TYPED_TEST_CASE_P(Albatross, SerializeTest, ToTest);

/*
 * Make sure all the example models serialize.
 */

template <typename DatasetType, int = 0> struct feature_type {
  typedef void type;
};

template <typename FeatureType>
struct feature_type<RegressionDataset<FeatureType>> {
  typedef FeatureType type;
};

template <typename T> class model_types {
  template <typename C,
            typename ModelType = decltype(std::declval<const T>().get_model())>
  static ModelType test_model(C *);
  template <typename> static void test_model(...);

  template <typename C, typename DatasetType =
                            decltype(std::declval<const T>().get_dataset())>
  static typename feature_type<DatasetType>::type test_feature_type(C *);
  template <typename> static void test_feature_type(...);

public:
  typedef decltype(test_model<T>(0)) model_type;
  typedef decltype(test_feature_type<T>(0)) feature;
  typedef typename fit_model_type<decltype(test_model<T>(0)),
                                  decltype(test_feature_type<T>(0))>::type
      fit_model_type;
};

template <typename ModelTestCase>
struct SerializableModelType
    : public SerializableType<typename model_types<ModelTestCase>::model_type> {

  using ModelType = typename model_types<ModelTestCase>::model_type;

  ModelType create() const {
    ModelTestCase test_case;
    auto model = test_case.get_model();
    double offset = 1.1;
    for (const auto &param : model.get_params()) {
      model.set_param_value(param.first, param.second.value + offset);
      offset += 1.11;
    }
    return model;
  }

  virtual bool are_equal(const ModelType &lhs, const ModelType &rhs) const {
    return lhs == rhs;
  };
};

template <typename ModelTestCase>
struct SerializableFitModelType
    : public SerializableType<
          typename model_types<ModelTestCase>::fit_model_type> {

  using FitModelType = typename model_types<ModelTestCase>::fit_model_type;

  FitModelType create() const {
    ModelTestCase test_case;
    auto model = test_case.get_model();
    auto dataset = test_case.get_dataset();
    double offset = 1.1;
    for (const auto &param : model.get_params()) {
      model.set_param_value(param.first, param.second.value + offset);
      offset += 1.11;
    }
    return model.fit(dataset);
  }

  virtual bool are_equal(const FitModelType &lhs,
                         const FitModelType &rhs) const {
    return lhs == rhs;
  };
};

TYPED_TEST_P(RegressionModelTester, test_model_serializes) {
  expect_roundtrip_serializable<cereal::JSONInputArchive,
                                cereal::JSONOutputArchive,
                                SerializableModelType<TypeParam>>();
}

TYPED_TEST_P(RegressionModelTester, test_fit_model_serializes) {
  expect_roundtrip_serializable<cereal::JSONInputArchive,
                                cereal::JSONOutputArchive,
                                SerializableFitModelType<TypeParam>>();
}

REGISTER_TYPED_TEST_CASE_P(RegressionModelTester, test_model_serializes,
                           test_fit_model_serializes);

INSTANTIATE_TYPED_TEST_CASE_P(test_serialize, RegressionModelTester,
                              ExampleModels);

} // namespace albatross
