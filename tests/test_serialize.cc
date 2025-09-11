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

#include <albatross/SparseGP>
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
                               {"9", Parameter(9.)}};
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

struct LinearCombo : public SerializableType<LinearCombination<double>> {
  LinearCombination<double> create() const override {
    std::vector<double> values = {1., 2., 5.};
    Eigen::VectorXd coefs(3);
    coefs << 10., 20., 50.;
    return LinearCombination<double>(values, coefs);
  }
};

REGISTER_TYPED_TEST_SUITE_P(SerializeTest, test_roundtrip_serialize_json,
                            test_roundtrip_serialize_binary,
                            test_roundtrip_serialize_portable_binary);

typedef ::testing::Types<LDLT, ExplainedCovarianceRepresentation, EigenMatrix3d,
                         SerializableType<Eigen::Matrix2i>, EmptyEigenVectorXd,
                         EigenVectorXd, EmptyEigenMatrixXd, EigenMatrixXd,
                         FullJointDistribution, FullMarginalDistribution,
                         ParameterStoreType, Dataset, DatasetWithMetadata,
                         SerializableType<MockModel>, VariantAsInt,
                         VariantAsDouble, BlockSymmetricMatrix, LinearCombo>
    ToTest;

INSTANTIATE_TYPED_TEST_SUITE_P(Albatross, SerializeTest, ToTest);

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

TYPED_TEST_P(RegressionModelTester, test_model_serializes_binary) {
  expect_roundtrip_serializable<cereal::BinaryInputArchive,
                                cereal::BinaryOutputArchive,
                                SerializableModelType<TypeParam>>();
}

TYPED_TEST_P(RegressionModelTester, test_fit_model_serializes_binary) {
  expect_roundtrip_serializable<cereal::BinaryInputArchive,
                                cereal::BinaryOutputArchive,
                                SerializableFitModelType<TypeParam>>();
}

TYPED_TEST_P(RegressionModelTester, test_model_serializes_portable_binary) {
  expect_roundtrip_serializable<cereal::PortableBinaryInputArchive,
                                cereal::PortableBinaryOutputArchive,
                                SerializableModelType<TypeParam>>();
}

TYPED_TEST_P(RegressionModelTester, test_fit_model_serializes_portable_binary) {
  expect_roundtrip_serializable<cereal::PortableBinaryInputArchive,
                                cereal::PortableBinaryOutputArchive,
                                SerializableFitModelType<TypeParam>>();
}

REGISTER_TYPED_TEST_SUITE_P(RegressionModelTester, test_model_serializes,
                            test_fit_model_serializes,
                            test_model_serializes_binary,
                            test_fit_model_serializes_binary,
                            test_model_serializes_portable_binary,
                            test_fit_model_serializes_portable_binary);

INSTANTIATE_TYPED_TEST_SUITE_P(test_serialize, RegressionModelTester,
                               ExampleModels);

TEST(test_serialize, test_variant_version_0) {
  int one = 1;
  variant<int, double> foo = one;

  // Serialize it using version 0
  std::ostringstream os;
  {
    cereal::JSONOutputArchive oarchive(os);
    save(oarchive, foo, 0);
  }

  // Deserialize it.
  std::istringstream is(os.str());
  variant<int, double> deserialized;
  {
    cereal::JSONInputArchive iarchive(is);
    load(iarchive, deserialized, 0);
  }
  // Make sure the original and deserialized representations are
  // equivalent.
  EXPECT_TRUE(foo == deserialized);
}

TEST(test_serialize, test_gp_serialize_version) {
  MakeGaussianProcess test_case;

  EXPECT_TRUE(bool(cereal::detail::has_serialization_version<
                   decltype(test_case.get_model())>::value));

  const std::uint32_t expected_version =
      test_case.get_model().serialization_version;
  const std::uint32_t actual_version =
      cereal::detail::Version<decltype(test_case.get_model())>::version;

  EXPECT_EQ(actual_version, expected_version);
}

} // namespace albatross

namespace other {

TEST(test_serialize, test_dataset_streamable) {
  albatross::MarginalDistribution targets(Eigen::VectorXd(1));
  RegressionDataset<int> dataset({1}, targets);
  std::ostringstream oss;
  oss << dataset;
}

TEST(test_serialize, test_marginal_streamable) {
  albatross::MarginalDistribution dist(Eigen::VectorXd(1));
  std::ostringstream oss;
  oss << dist;
}

TEST(test_serialize, test_joint_streamable) {
  albatross::JointDistribution dist(Eigen::VectorXd(1), Eigen::MatrixXd(1, 1));
  std::ostringstream oss;
  oss << dist;
}

TEST(test_serialize, test_ThreadPool_threads) {
  for (std::size_t n_threads = 1; n_threads < 33; ++n_threads) {
    auto pool = albatross::make_shared_thread_pool(n_threads);
    std::ostringstream os;
    {
      cereal::JSONOutputArchive oarchive(os);
      save(oarchive, pool);
    }

    std::istringstream is(os.str());
    std::shared_ptr<ThreadPool> pool_out = albatross::serial_thread_pool;
    {
      cereal::JSONInputArchive iarchive(is);
      load(iarchive, pool_out);
    }
    if (n_threads < 2) {
      EXPECT_EQ(pool, albatross::serial_thread_pool);
    } else {
      EXPECT_NE(pool_out, albatross::serial_thread_pool);
      EXPECT_EQ(pool->thread_count(), pool_out->thread_count());
    }
  }
}

TEST(test_serialize, test_ThreadPool_nullptr) {
  std::shared_ptr<ThreadPool> pool = albatross::serial_thread_pool;
  std::ostringstream os;
  {
    cereal::JSONOutputArchive oarchive(os);
    save(oarchive, pool);
  }

  std::istringstream is(os.str());
  std::shared_ptr<ThreadPool> pool_out = albatross::serial_thread_pool;
  {
    cereal::JSONInputArchive iarchive(is);
    load(iarchive, pool_out);
  }
  EXPECT_EQ(pool_out, albatross::serial_thread_pool);
}

using SparseMatrix = Eigen::SparseMatrix<double>;
using SPQR = Eigen::SerializableSPQR<SparseMatrix>;

TEST(test_serialize, serialize_spqr_simple) {
  Eigen::MatrixXd Adense(3, 3);
  // clang-format off
  Adense <<
    1., 4, 2,
    2, 2, 9.1,
    4, 8, 111.2;
  // clang-format on
  Eigen::VectorXd b(3);
  // clang-format off
  b <<
    2,
    4,
    99;
  // clang-format on
  SPQR spqr;
  spqr.setSPQROrdering(SPQR_ORDERING_COLAMD);
  spqr.compute(Adense.sparseView());
  EXPECT_EQ(spqr.info(), Eigen::Success);
  const auto x = spqr.solve(b);
  EXPECT_EQ(spqr.info(), Eigen::Success);
  std::ostringstream os;
  {
    cereal::JSONOutputArchive oarchive(os);
    spqr.save(oarchive, 0);
  }
  SPQR spqr_out;
  {
    std::istringstream is(os.str());
    cereal::JSONInputArchive iarchive(is);
    spqr_out.load(iarchive, 0);
  }
  const auto x_out = spqr_out.solve(b);
  EXPECT_EQ(x, x_out);
}

TEST(test_serialize, serialize_spqr_random) {
  constexpr Eigen::Index kMaxSize = 10000;
  constexpr Eigen::Index kMeanSize = 50;
  constexpr double kMeanFill = 0.2;
  constexpr std::size_t kNumIters = 1000;
  std::seed_seq seed{22};
  std::default_random_engine gen{seed};
  const auto gen_size = [&gen, &kMaxSize]() {
    std::poisson_distribution<Eigen::Index> size_dist{kMeanSize};
    return std::max(Eigen::Index{1}, std::min(kMaxSize, size_dist(gen)));
  };
  const auto gen_fill = [&gen]() {
    std::gamma_distribution<double> fill_dist{2, kMeanFill / 2};
    return std::min(1.0, fill_dist(gen));
  };
  // Allocate this outside the loop to make sure we don't care what's
  // in the incoming object when we deserialize.
  SPQR spqr_out;
  for (std::size_t iter = 0; iter < kNumIters; ++iter) {
    const auto rows = gen_size();
    const auto cols = gen_size();
    const auto fill = gen_fill();
    const SparseMatrix A =
        albatross::random_sparse_matrix<double>(rows, cols, fill, gen);
    const Eigen::VectorXd B = Eigen::VectorXd::Random(rows);

    SPQR spqr;
    spqr.setSPQROrdering(SPQR_ORDERING_COLAMD);
    spqr.cholmodCommon()->SPQR_nthreads = 2;
    spqr.compute(A);
    EXPECT_EQ(spqr.info(), Eigen::Success);
    const auto x = spqr.solve(B);
    EXPECT_EQ(spqr.info(), Eigen::Success);
    std::ostringstream os;
    {
      cereal::JSONOutputArchive oarchive(os);
      spqr.save(oarchive, 0);
    }
    {
      std::istringstream is(os.str());
      cereal::JSONInputArchive iarchive(is);
      spqr_out.load(iarchive, 0);
    }
    const auto x_out = spqr_out.solve(B);
    EXPECT_EQ(x, x_out);
  }
}

} // namespace other
