/*
 * Copyright (C) 2019 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include <albatross/CovarianceFunctions>
#include <albatross/Dataset>
#include <albatross/Indexing>
#include <gtest/gtest.h>

namespace albatross {

TEST(test_dataset, test_construct_and_subset) {
  std::vector<int> features = {3, 7, 1};
  const auto targets = Eigen::VectorXd::Random(3);

  RegressionDataset<int> dataset(features, targets);

  EXPECT_EQ(dataset.size(), features.size());
  EXPECT_EQ(dataset.size(), cast::to_size(targets.size()));

  std::vector<std::size_t> indices = {0, 2};
  const auto subset_dataset = dataset.subset(indices);
  EXPECT_EQ(subset_dataset.size(), indices.size());

  auto is_3_or_1 = [](const int &x) { return x == 3 || x == 1; };

  const auto filtered_dataset = filter(dataset, is_3_or_1);
  EXPECT_EQ(subset_dataset, filtered_dataset);
}

template <typename T>
Eigen::VectorXd random_targets_for(const std::vector<T> &features) {
  return Eigen::VectorXd::Random(cast::to_index(features.size()));
}

template <typename T>
RegressionDataset<T> random_dataset_for(const std::vector<T> &features) {
  return {features, random_targets_for(features)};
}

TEST(test_dataset, test_deduplicate) {
  const auto dataset = random_dataset_for(std::vector<int>{0, 1, 1, 2});
  const auto dedupped = deduplicate(dataset);

  const std::vector<std::size_t> expected_inds = {0, 2, 3};

  EXPECT_EQ(dedupped, dataset.subset(expected_inds));
  EXPECT_EQ(dedupped, deduplicate(dedupped));
}

TEST(test_dataset, test_align_datasets_a_in_b) {
  auto dataset_a = random_dataset_for(std::vector<int>{0, 1, 2});
  auto dataset_b = random_dataset_for(std::vector<int>{2, 3, 0, 1});

  EXPECT_NE(dataset_a.features, dataset_b.features);
  align_datasets(&dataset_a, &dataset_b);
  EXPECT_EQ(dataset_a.size(), 3);
  EXPECT_EQ(dataset_a.features, dataset_b.features);
}

TEST(test_dataset, test_align_datasets_a_in_b_custom_compare) {
  auto dataset_a = random_dataset_for(std::vector<int>{0, 1, 2});
  auto dataset_b = random_dataset_for(std::vector<int>{2, 3, 0, 1});

  EXPECT_NE(dataset_a.features, dataset_b.features);

// GCC 6 gets confused by this line, I think because `align_datasets`
// is marked `inline`
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
  auto custom_compare = [](const auto &x, const auto &y) { return x == y; };
#pragma GCC diagnostic pop

  align_datasets(&dataset_a, &dataset_b, custom_compare);
  EXPECT_EQ(dataset_a.size(), 3);
  EXPECT_EQ(dataset_a.features, dataset_b.features);
}

TEST(test_dataset, test_align_datasets_a_in_b_unordered) {
  auto dataset_a = random_dataset_for(std::vector<int>{0, 2, 1});
  auto dataset_b = random_dataset_for(std::vector<int>{2, 3, 0, 1});

  EXPECT_NE(dataset_a.features, dataset_b.features);
  align_datasets(&dataset_a, &dataset_b);
  EXPECT_EQ(dataset_a.size(), 3);
  EXPECT_EQ(dataset_a.features, dataset_b.features);
}

TEST(test_dataset, test_align_datasets_a_not_in_b) {
  auto dataset_a = random_dataset_for(std::vector<int>{0, 1, 2, 3});
  auto dataset_b = random_dataset_for(std::vector<int>{2, 4, 0});

  EXPECT_NE(dataset_a.features, dataset_b.features);
  align_datasets(&dataset_a, &dataset_b);
  EXPECT_EQ(dataset_a.size(), 2);
  EXPECT_EQ(dataset_a.features, dataset_b.features);
}

TEST(test_dataset, test_align_datasets_no_intersect) {
  auto dataset_a = random_dataset_for(std::vector<int>{0, 1, 2});
  auto dataset_b = random_dataset_for(std::vector<int>{3, 4, 5});

  align_datasets(&dataset_a, &dataset_b);
  EXPECT_EQ(dataset_a.size(), 0);
  EXPECT_EQ(dataset_b.size(), 0);
}

void expect_split_recombine(const RegressionDataset<int> &dataset) {
  std::vector<std::size_t> first_indices = {0, 1};
  const auto first = subset(dataset, first_indices);
  EXPECT_EQ(first.size(), first_indices.size());

  std::vector<std::size_t> second_indices = {2};
  const auto second = subset(dataset, second_indices);
  EXPECT_EQ(second.size(), second_indices.size());

  const auto reconstructed = concatenate_datasets(first, second);

  EXPECT_EQ(dataset, reconstructed);
}

TEST(test_dataset, test_concatenate_same_type) {
  std::vector<int> features = {3, 7, 1};
  const auto mean_only_targets = random_targets_for(features);
  RegressionDataset<int> mean_only_dataset(features, mean_only_targets);

  expect_split_recombine(mean_only_dataset);

  Eigen::VectorXd variance = Eigen::VectorXd::Ones(mean_only_targets.size());
  MarginalDistribution targets(mean_only_targets, variance.asDiagonal());
  RegressionDataset<int> dataset(features, targets);

  expect_split_recombine(dataset);
}

TEST(test_dataset, test_concatenate_different_type) {
  std::vector<int> int_features = {3, 7, 1};
  RegressionDataset<int> int_dataset(int_features,
                                     random_targets_for(int_features));

  std::vector<double> double_features = {3., 7., 1.};
  RegressionDataset<double> double_dataset(double_features,
                                           random_targets_for(double_features));

  const auto reconstructed = concatenate_datasets(int_dataset, double_dataset);

  EXPECT_TRUE(
      bool(std::is_same<typename decltype(reconstructed.features)::value_type,
                        variant<int, double>>::value));

  for (std::size_t i = 0; i < reconstructed.features.size(); ++i) {
    if (i < int_features.size()) {
      EXPECT_TRUE(reconstructed.features[i].is<int>());
      EXPECT_FALSE(reconstructed.features[i].is<double>());
      int actual = reconstructed.features[i].get<int>();
      EXPECT_EQ(actual, int_features[i]);
    } else {
      EXPECT_TRUE(reconstructed.features[i].is<double>());
      EXPECT_FALSE(reconstructed.features[i].is<int>());
      double actual = reconstructed.features[i].get<double>();
      EXPECT_EQ(actual, double_features[i - int_features.size()]);
    }
  }
}

TEST(test_dataset, test_streamable_features) {
  auto dataset = random_dataset_for(std::vector<int>{3, 7, 1});

  std::ostringstream oss;
  oss << dataset << std::endl;
}

struct NotStreamable {};

TEST(test_dataset, test_not_streamable_features) {
  auto dataset = random_dataset_for(std::vector<NotStreamable>{
      NotStreamable(), NotStreamable(), NotStreamable()});

  std::ostringstream oss;
  oss << dataset << std::endl;
}

struct EmptyTransform {
  auto get_matrix(Eigen::Index n) {
    Eigen::SparseMatrix<double> matrix(0, n);
    return matrix;
  }
};

struct EmptyRow {
  auto get_matrix(Eigen::Index n) {
    Eigen::SparseMatrix<double, Eigen::ColMajor> matrix(n, n);
    matrix.setIdentity();
    matrix.coeffRef(0, 0) = 0.;
    matrix.prune(0.);
    return matrix;
  }
};

struct SparseIdentity {
  auto get_matrix(Eigen::Index n) {
    Eigen::SparseMatrix<double, Eigen::ColMajor> matrix(n, n);
    matrix.setIdentity();
    return matrix;
  }
};

struct SparseShortIdentity {
  auto get_matrix(Eigen::Index n) {
    Eigen::SparseMatrix<double, Eigen::ColMajor> matrix =
        Eigen::MatrixXd::Identity(n - 1, n).sparseView();
    return matrix;
  }
};

struct SparseShortIdentityRowMajor {
  auto get_matrix(Eigen::Index n) {
    Eigen::SparseMatrix<double, Eigen::RowMajor> matrix =
        Eigen::MatrixXd::Identity(n - 1, n).sparseView();
    return matrix;
  }
};

struct SparseRandomTall {
  auto get_matrix(Eigen::Index n) {
    Eigen::SparseMatrix<double> matrix =
        Eigen::MatrixXd::Random(n + 1, n).sparseView();
    return matrix;
  }
};

struct SparseRandomRowMajor {
  auto get_matrix(Eigen::Index n) {
    Eigen::SparseMatrix<double, Eigen::RowMajor> matrix =
        Eigen::MatrixXd::Random(n, n).sparseView();
    return matrix;
  }
};

template <typename CaseType>
class DatasetOperatorTester : public ::testing::Test {
public:
  CaseType test_case;
};

typedef ::testing::Types<EmptyTransform, EmptyRow, SparseIdentity,
                         SparseShortIdentity, SparseShortIdentityRowMajor,
                         SparseRandomTall, SparseRandomRowMajor>
    DatasetOperatorTestCases;

TYPED_TEST_SUITE_P(DatasetOperatorTester);

template <typename X> bool all_values_are_unique(const std::vector<X> &xs) {
  return std::set<X>(xs.begin(), xs.end()).size() == xs.size();
}

template <typename X> struct expected_transformed_type {
  typedef albatross::LinearCombination<X> type;
};

// A transformation of a linear combination should preserve the
// original linear combination type.
template <typename X>
struct expected_transformed_type<albatross::LinearCombination<X>> {
  typedef albatross::LinearCombination<X> type;
};

TYPED_TEST_P(DatasetOperatorTester, test_output_type_ints) {
  const std::vector<int> features = {3, 7, 1};

  const auto matrix =
      this->test_case.get_matrix(cast::to_index(features.size()));
  const auto linear_combos = matrix * features;

  using OriginalType = typename decltype(features)::value_type;
  using ActualType = typename decltype(linear_combos)::value_type;
  using ExpectedType = typename expected_transformed_type<OriginalType>::type;
  bool is_linear_combo = std::is_same<ActualType, ExpectedType>::value;

  EXPECT_TRUE(is_linear_combo);
}

TYPED_TEST_P(DatasetOperatorTester, test_output_size_combos) {
  std::vector<albatross::LinearCombination<int>> features;

  auto add_combo = [&](const std::vector<int> &values,
                       const Eigen::VectorXd &coefs) {
    features.emplace_back(values, coefs);
  };

  add_combo({1, 3}, Eigen::VectorXd::Random(2));
  add_combo({7, 5}, Eigen::VectorXd::Random(2));
  add_combo({3, 1, 4}, Eigen::VectorXd::Random(3));
  add_combo({9}, Eigen::VectorXd::Random(1));

  const auto matrix =
      this->test_case.get_matrix(cast::to_index(features.size()));
  const auto linear_combos = matrix * features;

  const Eigen::MatrixXd dense = matrix;

  EXPECT_EQ(cast::to_index(linear_combos.size()), matrix.rows());
  for (Eigen::Index row = 0; row < dense.rows(); ++row) {
    const auto srow = cast::to_size(row);
    std::size_t expected_size_of_output_combo = 0;
    for (Eigen::Index col = 0; col < dense.cols(); ++col) {
      if (dense(row, col) != 0.) {
        const auto scol = cast::to_size(col);
        expected_size_of_output_combo += features[scol].values.size();
      }
    }
    EXPECT_EQ(expected_size_of_output_combo, linear_combos[srow].values.size());
  }
}

TYPED_TEST_P(DatasetOperatorTester, test_equivalent_cov_combos) {
  std::vector<albatross::LinearCombination<int>> features;

  auto add_combo = [&](const std::vector<int> &values,
                       const Eigen::VectorXd &coefs) {
    features.emplace_back(values, coefs);
  };

  add_combo({1, 3}, Eigen::VectorXd::Random(2));
  add_combo({7, 5}, Eigen::VectorXd::Random(2));
  add_combo({3, 1, 4}, Eigen::VectorXd::Random(3));
  add_combo({9}, Eigen::VectorXd::Random(1));

  const auto matrix =
      this->test_case.get_matrix(cast::to_index(features.size()));
  const auto linear_combos = matrix * features;

  Exponential<EuclideanDistance> cov;

  // Computing a covariance matrix using the original features then
  // transforming the result should be the same as transforming the
  // features, then computing the covariance.
  const Eigen::MatrixXd full_cov_matrix = cov(features);
  const Eigen::MatrixXd transformed_cov_matrix =
      (matrix * full_cov_matrix) * matrix.transpose();
  const Eigen::MatrixXd combo_cov_matrix = cov(linear_combos);

  EXPECT_LT((transformed_cov_matrix - combo_cov_matrix).norm(), 1e-8);
}

TYPED_TEST_P(DatasetOperatorTester, test_output_type_combos) {
  std::vector<albatross::LinearCombination<int>> features;

  auto add_combo = [&](const std::vector<int> &values,
                       const Eigen::VectorXd &coefs) {
    features.emplace_back(values, coefs);
  };

  add_combo({1, 3}, Eigen::VectorXd::Random(2));
  add_combo({7, 5}, Eigen::VectorXd::Random(2));
  add_combo({3, 1, 4}, Eigen::VectorXd::Random(3));
  add_combo({9}, Eigen::VectorXd::Random(1));

  const auto matrix =
      this->test_case.get_matrix(cast::to_index(features.size()));
  const auto linear_combos = matrix * features;

  using OriginalType = typename decltype(features)::value_type;
  using ActualType = typename decltype(linear_combos)::value_type;
  using ExpectedType = typename expected_transformed_type<OriginalType>::type;
  bool is_linear_combo = std::is_same<ActualType, ExpectedType>::value;
  EXPECT_TRUE(is_linear_combo);
}

TYPED_TEST_P(DatasetOperatorTester, test_inferred_transformation) {
  // Note these have to be unique for the tests to work.
  const std::vector<int> features = {3, 7, 1};
  EXPECT_TRUE(all_values_are_unique(features));

  const auto matrix =
      this->test_case.get_matrix(cast::to_index(features.size()));
  const auto linear_combos = matrix * features;

  using MatrixType = typename std::remove_const<decltype(matrix)>::type;
  MatrixType inferred_matrix(matrix.rows(), matrix.cols());
  // Build the implied matrix from the linear combination objects
  for (std::size_t i = 0; i < linear_combos.size(); ++i) {
    const auto &lin_combo = linear_combos[i];
    for (std::size_t j = 0; j < lin_combo.values.size(); ++j) {
      const Eigen::Index index =
          std::find(features.begin(), features.end(), lin_combo.values[j]) -
          features.begin();
      inferred_matrix.coeffRef(cast::to_index(i), index) =
          lin_combo.coefficients[cast::to_index(j)];
    }
  }

  Eigen::MatrixXd dense_actual(matrix);
  Eigen::MatrixXd dense_inferred(inferred_matrix);
  EXPECT_LT((dense_actual - dense_inferred).norm(), 1e-4);
}

TYPED_TEST_P(DatasetOperatorTester, test_multiply_dataset) {
  const std::vector<int> features = {3, 7, 1};
  RegressionDataset<int> dataset(features, Eigen::VectorXd::Ones(3));

  const auto sparse_matrix =
      this->test_case.get_matrix(cast::to_index(features.size()));
  const auto transformed_dataset = sparse_matrix * dataset;

  Eigen::MatrixXd dense_matrix(sparse_matrix);
  const auto dense_transformed_dataset = dense_matrix * dataset;

  EXPECT_EQ(transformed_dataset, dense_transformed_dataset);
}

TYPED_TEST_P(DatasetOperatorTester, test_sparse_multiply_same_as_dense) {
  const std::vector<int> features = {3, 7, 1};

  const auto sparse_matrix =
      this->test_case.get_matrix(cast::to_index(features.size()));
  const auto sparse_combos = sparse_matrix * features;

  Eigen::MatrixXd dense_matrix(sparse_matrix);
  const auto dense_combos = dense_matrix * features;

  EXPECT_EQ(sparse_combos, dense_combos);
}

REGISTER_TYPED_TEST_SUITE_P(DatasetOperatorTester, test_output_type_ints,
                            test_output_size_combos, test_output_type_combos,
                            test_equivalent_cov_combos,
                            test_inferred_transformation,
                            test_sparse_multiply_same_as_dense,
                            test_multiply_dataset);

INSTANTIATE_TYPED_TEST_SUITE_P(test_core_dataset, DatasetOperatorTester,
                               DatasetOperatorTestCases);

} // namespace albatross
