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

#include <albatross/Core>
#include <albatross/serialize/Core>

#include <albatross/src/utils/csv_utils.hpp>
#include <csv.h>
#include <gtest/gtest.h>

namespace albatross {

struct SubFeature {
  double one = 1.;
  int two = 2;

  template <typename Archive> void serialize(Archive &archive) {
    archive(cereal::make_nvp("one", one));
    archive(cereal::make_nvp("two", two));
  }
};

struct TestFeature {
  double foo;
  int bar;
  SubFeature feature;
  bool has_other;
  long *other;
  variant<double, SubFeature> double_or_feature;

  TestFeature()
      : foo(1.), bar(2), feature(), has_other(false), other(nullptr),
        double_or_feature(1.){};

  TestFeature(double foo_, int bar_, const SubFeature &feature_, long *other_,
              const variant<double, SubFeature> &double_or_feature_)
      : foo(foo_), bar(bar_), feature(feature_), has_other(false),
        other(other_), double_or_feature(double_or_feature_) {
    has_other = other_ != nullptr;
  };

  template <typename Archive> void save(Archive &archive) const {
    archive(CEREAL_NVP(foo));
    archive(CEREAL_NVP(bar));
    archive(CEREAL_NVP(feature));
    archive(CEREAL_NVP(has_other));
    if (has_other) {
      archive(cereal::make_nvp("other", *other));
    }
    archive(CEREAL_NVP(double_or_feature));
  }

  template <typename Archive> void load(Archive &archive) {
    archive(CEREAL_NVP(foo));
    archive(CEREAL_NVP(bar));
    archive(CEREAL_NVP(feature));
    archive(CEREAL_NVP(has_other));
    if (has_other) {
      archive(cereal::make_nvp("other", *other));
    }
    archive(CEREAL_NVP(double_or_feature));
  }
};

/*
 * This does nothing more than read the CSV, but would fail if
 * the CSV were missing columns or had unparsable data.  It does NOT
 * try to reassemble the object we wrote to file.
 */
void read_test_csv(std::istream &stream) {
  io::CSVReader<16> reader("garbage", stream);
  reader.read_header(io::ignore_no_column, "bar",
                     "double_or_feature.cereal_class_version",
                     "double_or_feature.data.one", "double_or_feature.data.two",
                     "double_or_feature.data", "double_or_feature.which",
                     "double_or_feature.which_typeid", "feature.one",
                     "feature.two", "foo", "has_other", "other", "prediction",
                     "prediction_variance", "target", "target_variance");

  bool more_to_parse = true;
  while (more_to_parse) {
    double prediction;
    std::string prediction_variance_str;
    double target;
    std::string target_variance_str;
    TestFeature f;
    int version, which;
    std::string double_or_feature;
    std::string double_or_feature_one;
    std::string double_or_feature_two;
    std::string double_or_feature_which_typeid;
    std::string has_other;
    std::string other;
    more_to_parse = reader.read_row(
        f.bar, version, double_or_feature_one, double_or_feature_two,
        double_or_feature, which, double_or_feature_which_typeid, f.feature.one,
        f.feature.two, f.foo, has_other, other, prediction,
        prediction_variance_str, target, target_variance_str);
  }
}

static long test_integer = 5;

std::vector<TestFeature> test_features() {
  SubFeature sub = {4.4, 6};
  TestFeature one(1.2, 2, {1.3, 3}, nullptr, 3.);
  TestFeature two(2.2, 3, {2.3, 4}, &test_integer, 3.);
  TestFeature three(3.2, 4, {3.3, 5}, nullptr, sub);

  std::vector<TestFeature> features = {one, two, three};
  return features;
}

TEST(test_csv_utils, test_writes) {
  std::vector<TestFeature> features = test_features();
  Eigen::VectorXd targets(features.size());
  targets << 1., 2., 3.;

  MarginalDistribution predictions(targets);

  RegressionDataset<TestFeature> dataset(features, targets);

  std::ostringstream oss;
  write_to_csv(oss, dataset, predictions);

  std::istringstream iss(oss.str());

  read_test_csv(iss);
}

TEST(test_csv_utils, test_writes_without_predictions) {
  std::vector<TestFeature> features = test_features();
  Eigen::VectorXd targets(3);
  targets << 1., 2., 3.;

  RegressionDataset<TestFeature> dataset(features, targets);

  std::ostringstream oss;
  write_to_csv(oss, dataset);

  std::istringstream iss(oss.str());

  read_test_csv(iss);
}

/*
 * This does nothing more than read the CSV, but would fail if
 * the CSV were missing columns or had unparsable data.
 */
void read_test_csv_with_metadata(std::istream &stream) {
  io::CSVReader<17> reader("garbage", stream);
  reader.read_header(
      io::ignore_no_column, "bar", "double_or_feature.cereal_class_version",
      "double_or_feature.data.one", "double_or_feature.data.two",
      "double_or_feature.data", "double_or_feature.which",
      "double_or_feature.which_typeid", "feature.one", "feature.two", "foo",
      "has_other", "other", "prediction", "prediction_variance", "target",
      "target_variance", "time");

  bool more_to_parse = true;
  while (more_to_parse) {
    double prediction;
    std::string prediction_variance_str;
    double target;
    std::string time;
    std::string target_variance_str;
    TestFeature f;
    int version, which;
    std::string double_or_feature;
    std::string double_or_feature_one;
    std::string double_or_feature_two;
    std::string double_or_feature_which_typeid;
    std::string has_other;
    std::string other;
    more_to_parse = reader.read_row(
        f.bar, version, double_or_feature_one, double_or_feature_two,
        double_or_feature, which, double_or_feature_which_typeid, f.feature.one,
        f.feature.two, f.foo, has_other, other, prediction,
        prediction_variance_str, target, target_variance_str, time);
  }
}

TEST(test_csv_utils, test_writes_metadata) {

  std::vector<TestFeature> features = test_features();
  Eigen::VectorXd targets(3);
  targets << 1., 2., 3.;

  MarginalDistribution prediction(targets);

  RegressionDataset<TestFeature> first(features, targets);
  first.metadata["time"] = "1";

  RegressionDataset<TestFeature> second(features, targets);
  second.metadata["time"] = "2";

  std::vector<decltype(first)> datasets = {first, second};
  std::vector<decltype(prediction)> predictions = {prediction, prediction};

  std::ostringstream oss;
  write_to_csv(oss, datasets, predictions);

  std::istringstream iss(oss.str());
  read_test_csv_with_metadata(iss);
}

struct CustomFeature {
  double one = 1.;
  int two = 2;

  template <typename Archive> void serialize(Archive &archive) {
    archive(cereal::make_nvp("one", one));
    archive(cereal::make_nvp("two", two));
  }
};

auto to_map(const CustomFeature &feature) {
  std::map<std::string, std::string> output;
  output["one"] = std::to_string(feature.one);
  output["two"] = std::to_string(feature.two);
  output["three"] = std::to_string(feature.one) + std::to_string(feature.two);
  return output;
};

/*
 * This does nothing more than read the CSV, but would fail if
 * the CSV were missing columns or had unparsable data.
 */
void read_test_csv_with_custom_to_map(std::istream &stream) {
  io::CSVReader<7> reader("garbage", stream);
  reader.read_header(io::ignore_no_column, "one", "two", "three", "prediction",
                     "prediction_variance", "target", "target_variance");

  bool more_to_parse = true;
  while (more_to_parse) {
    double prediction;
    std::string prediction_variance_str;
    double target;
    std::string three;
    std::string target_variance_str;
    CustomFeature f;
    more_to_parse =
        reader.read_row(f.one, f.two, three, prediction,
                        prediction_variance_str, target, target_variance_str);
  }
}

TEST(test_csv_utils, test_custom_writes) {

  CustomFeature one = {1.2, 2};
  CustomFeature two = {2.2, 3};
  std::vector<CustomFeature> features = {one, two};
  Eigen::Vector2d targets;
  targets << 1., 2.;

  MarginalDistribution predictions(targets);

  RegressionDataset<CustomFeature> dataset(features, targets);

  std::ostringstream oss;
  write_to_csv(oss, dataset, predictions);

  std::istringstream iss(oss.str());
  read_test_csv_with_custom_to_map(iss);
}

TEST(test_csv_utils, test_writes_eigen) {

  Eigen::MatrixXd x = Eigen::MatrixXd::Random(3, 4);

  std::ostringstream oss;
  write_to_csv(oss, x);

  EXPECT_GT(oss.str().size(), 0);
}

} // namespace albatross
