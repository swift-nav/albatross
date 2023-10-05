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

#ifndef ALBATROSS_RANDOM_UTILS_H
#define ALBATROSS_RANDOM_UTILS_H

namespace albatross {
/*
 * Samples integers between low and high (inclusive) with replacement.
 */
inline std::vector<std::size_t>
randint_without_replacement(std::size_t n, std::size_t low, std::size_t high,
                            std::default_random_engine &gen) {
  std::size_t n_choices = high - low + 1;
  if (n > n_choices) {
    std::cout << "ERROR: n (" << n << ") is larger than n_choices ("
              << n_choices << ")" << std::endl;
    ALBATROSS_ASSERT(false);
  }

  if (n == n_choices) {
    std::vector<std::size_t> all_inds(n);
    std::iota(all_inds.begin(), all_inds.end(), low);
    return all_inds;
  }

  if (n > n_choices / 2 + 1) {
    // Since we're trying to randomly sample more than half of the
    // points it'll be faster to randomly sample which points we
    // should throw out than which ones we should keep.
    const auto to_throw_out =
        randint_without_replacement(n_choices - n, 0, n_choices - 1, gen);
    auto to_keep = indices_complement(to_throw_out, n_choices);

    if (low != 0) {
      for (auto &el : to_keep) {
        el += low;
      }
    }
    return to_keep;
  }

  std::uniform_int_distribution<std::size_t> dist(low, high);
  std::set<std::size_t> samples;
  while (samples.size() < n) {
    samples.insert(dist(gen));
  }
  return std::vector<std::size_t>(samples.begin(), samples.end());
}

template <typename X>
inline std::vector<X>
random_without_replacement(const std::vector<X> &xs, std::size_t n,
                           std::default_random_engine &gen) {
  std::vector<X> random_sample;
  for (const auto &i : randint_without_replacement(n, 0, xs.size() - 1, gen)) {
    random_sample.emplace_back(xs[i]);
  }
  return random_sample;
}

template <typename _Scalar, int _Rows, int _Cols, typename DistributionType,
          typename RandomNumberGenerator>
void random_fill(Eigen::Matrix<_Scalar, _Rows, _Cols> &matrix,
                 DistributionType &dist, RandomNumberGenerator &rng) {

  auto random_sample = [&]() { return dist(rng); };

  matrix =
      Eigen::MatrixXd::NullaryExpr(matrix.rows(), matrix.cols(), random_sample);
}

template <typename _Scalar, int _Rows, int _Cols,
          typename RandomNumberGenerator>
void gaussian_fill(Eigen::Matrix<_Scalar, _Rows, _Cols> &matrix, double mean,
                   double sd, RandomNumberGenerator &rng) {
  std::normal_distribution<_Scalar> dist(mean, sd);
  random_fill(matrix, dist, rng);
}

template <typename _Scalar, int _Rows, int _Cols,
          typename RandomNumberGenerator>
void gaussian_fill(Eigen::Matrix<_Scalar, _Rows, _Cols> &matrix,
                   RandomNumberGenerator &rng) {
  gaussian_fill(matrix, 0., 1., rng);
}

template <typename _Scalar, int _Rows, int _Cols>
void gaussian_fill(Eigen::Matrix<_Scalar, _Rows, _Cols> &matrix) {
  std::default_random_engine rng;
  gaussian_fill(matrix, 0., 1., rng);
}

template <typename Distribution, typename RandomNumberGenerator>
inline Eigen::MatrixXd
random_covariance_matrix(Eigen::Index k, Distribution &eigen_value_distribution,
                         RandomNumberGenerator &gen) {

  Eigen::MatrixXd Q(k, k);
  gaussian_fill(Q, gen);
  Q = Q.colPivHouseholderQr().matrixQ();

  Eigen::VectorXd diag(k);

  random_fill(diag, eigen_value_distribution, gen);

  return Q * diag.asDiagonal() * Q.transpose();
}

template <typename RandomNumberGenerator>
inline Eigen::MatrixXd random_covariance_matrix(Eigen::Index k,
                                                RandomNumberGenerator &gen) {
  std::gamma_distribution<double> distribution(1.0, 1.0);
  return random_covariance_matrix(k, distribution, gen);
}

template <typename RandomNumberGenerator>
inline Eigen::VectorXd random_multivariate_normal(const Eigen::VectorXd &mean,
                                                  const Eigen::MatrixXd &cov,
                                                  RandomNumberGenerator &gen) {
  std::normal_distribution<double> dist;
  ALBATROSS_ASSERT(mean.size() == cov.rows());
  ALBATROSS_ASSERT(cov.rows() == cov.cols());

  Eigen::VectorXd sample(mean.size());
  gaussian_fill(sample, gen);

  sample = cov.llt().matrixL() * sample;

  sample += mean;

  return sample;
}

template <typename RandomNumberGenerator>
inline Eigen::VectorXd random_multivariate_normal(const Eigen::MatrixXd &cov,
                                                  RandomNumberGenerator &gen) {
  return random_multivariate_normal(Eigen::VectorXd::Zero(cov.rows()), cov,
                                    gen);
}

template <typename T>
Eigen::SparseMatrix<T>
random_sparse_matrix(Eigen::Index rows, Eigen::Index cols, double fill,
                     std::default_random_engine &gen, T min_element = -1e6,
                     T max_element = 1e6) {
  std::uniform_real_distribution<T> valdis(min_element, max_element);
  std::uniform_int_distribution<Eigen::Index> rowdis(0, rows - 1);
  std::uniform_int_distribution<Eigen::Index> coldis(0, cols - 1);

  std::vector<Eigen::Triplet<T>> tripletList;
  const auto nnz = static_cast<std::size_t>(static_cast<double>(rows) *
                                            (static_cast<double>(cols) * fill));
  std::set<Eigen::Index> nnz_pos;
  for (std::size_t i = 0; i < nnz; ++i) {
    Eigen::Index r = rowdis(gen);
    Eigen::Index c = coldis(gen);
    Eigen::Index pos = r * cols + c;
    while (nnz_pos.find(pos) != nnz_pos.end()) {
      r = rowdis(gen);
      c = coldis(gen);
      pos = r * cols + c;
    }

    nnz_pos.insert(pos);
    tripletList.emplace_back(r, c, valdis(gen));
  }

  Eigen::SparseMatrix<T> mat(rows, cols);
  mat.setFromTriplets(tripletList.begin(),
                      tripletList.end()); // create the matrix
  return mat;
}

} // namespace albatross

#endif
