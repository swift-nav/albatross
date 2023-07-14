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

#ifndef ALBATROSS_CORE_DISTRIBUTION_H
#define ALBATROSS_CORE_DISTRIBUTION_H

inline bool operator==(const albatross::DiagonalMatrixXd &x,
                       const albatross::DiagonalMatrixXd &y) {
  return (x.diagonal() == y.diagonal());
}

namespace albatross {

constexpr double cDefaultApproximatelyEqualEpsilon = 1e-3;

template <typename Derived> struct DistributionBase {

private:
  // Declaring these private makes it impossible to accidentally do things like:
  //     class A : public CovarianceFunction<B> {}
  // or
  //     using A = CovarianceFunction<B>;
  //
  // which if unchecked can lead to some very strange behavior.
  DistributionBase(){};
  friend Derived;

public:
  DistributionBase(const Eigen::VectorXd &mean_) : mean(mean_){};

  std::size_t size() const {
    // If the covariance is defined it must have the same number
    // of rows and columns which should be the same size as the mean.
    derived().assert_valid();
    return cast::to_size(mean.size());
  }

  double get_diagonal(Eigen::Index i) const {
    return derived().get_diagonal(i);
  }

  Eigen::VectorXd mean;
  std::map<std::string, std::string> metadata;

  Derived &derived() { return *static_cast<Derived *>(this); }

  const Derived &derived() const { return *static_cast<const Derived *>(this); }
};

template <typename T>
struct is_distribution : public std::is_base_of<DistributionBase<T>, T> {};

struct MarginalDistribution : public DistributionBase<MarginalDistribution> {

  using Base = DistributionBase<MarginalDistribution>;
  using CovarianceType = DiagonalMatrixXd;

  MarginalDistribution(){};

  MarginalDistribution(const Eigen::VectorXd &mean_)
      : Base(mean_), covariance(mean_.size()) {
    covariance.diagonal().fill(0.);
    assert_valid();
  };

  MarginalDistribution(const Eigen::VectorXd &mean_,
                       const DiagonalMatrixXd &covariance_)
      : Base(mean_), covariance(covariance_) {
    assert_valid();
  };

  template <typename DiagonalDerived>
  MarginalDistribution(const Eigen::VectorXd &mean_,
                       const Eigen::DiagonalBase<DiagonalDerived> &covariance_)
      : Base(mean_), covariance(covariance_) {
    assert_valid();
  }

  MarginalDistribution(const Eigen::VectorXd &mean_,
                       const Eigen::VectorXd &variance_)
      : Base(mean_), covariance(variance_.asDiagonal()) {
    assert_valid();
  };

  void assert_valid() const {
    ALBATROSS_ASSERT(mean.size() == covariance.rows());
    ALBATROSS_ASSERT(mean.size() == covariance.cols());
  }

  double get_diagonal(Eigen::Index i) const {
    ALBATROSS_ASSERT(i >= 0 && i < covariance.rows());
    return covariance.diagonal()[i];
  }

  bool approximately_equal(
      const MarginalDistribution &other,
      const double epsilon = cDefaultApproximatelyEqualEpsilon) const {
    const bool mean_approx_equal = mean.isApprox(other.mean, epsilon);
    const bool cov_approx_equal =
        covariance.diagonal().isApprox(other.covariance.diagonal(), epsilon);
    return mean_approx_equal && cov_approx_equal;
  }

  bool operator==(const MarginalDistribution &other) const {
    return (mean == other.mean) && (covariance == other.covariance);
  }

  MarginalDistribution operator+(const MarginalDistribution &other) const {
    return MarginalDistribution(
        mean + other.mean, covariance.diagonal() + other.covariance.diagonal());
  }

  MarginalDistribution operator-(const MarginalDistribution &other) const {
    return MarginalDistribution(
        mean - other.mean, covariance.diagonal() + other.covariance.diagonal());
  }

  MarginalDistribution operator*(double scale) const {
    return MarginalDistribution(scale * mean, scale * scale * covariance);
  }

  template <typename SizeType>
  MarginalDistribution subset(const std::vector<SizeType> &indices) const {
    return MarginalDistribution(
        albatross::subset(mean, indices),
        albatross::subset(covariance.diagonal(), indices));
  }

  template <typename SizeType>
  void set_subset(const MarginalDistribution &from,
                  const std::vector<SizeType> &indices) {
    albatross::set_subset(from.mean, indices, &mean);
    albatross::set_subset(from.covariance.diagonal(), indices,
                          &covariance.diagonal());
  }

  CovarianceType covariance;
};

struct JointDistribution : public DistributionBase<JointDistribution> {

  using Base = DistributionBase<JointDistribution>;
  using CovarianceType = Eigen::MatrixXd;

  JointDistribution(){};

  JointDistribution(double mean_, double variance_) {
    mean.resize(1);
    mean << mean_;
    covariance.resize(1, 1);
    covariance << variance_;
  }

  JointDistribution(const Eigen::VectorXd &mean_,
                    const Eigen::MatrixXd &covariance_)
      : Base(mean_), covariance(covariance_) {
    ALBATROSS_ASSERT(mean_.size() == covariance_.rows());
  };

  void assert_valid() const {
    ALBATROSS_ASSERT(mean.size() == covariance.rows());
    ALBATROSS_ASSERT(mean.size() == covariance.cols());
  }

  double get_diagonal(Eigen::Index i) const {
    ALBATROSS_ASSERT(i >= 0 && i < covariance.rows());
    return covariance.diagonal()[i];
  }

  bool approximately_equal(
      const JointDistribution &other,
      const double epsilon = cDefaultApproximatelyEqualEpsilon) const {
    const bool mean_approx_equal = mean.isApprox(other.mean, epsilon);
    const bool cov_approx_equal =
        covariance.isApprox(other.covariance, epsilon);
    return mean_approx_equal && cov_approx_equal;
  }

  bool operator==(const JointDistribution &other) const {
    return (mean == other.mean) && (covariance == other.covariance);
  }

  JointDistribution operator*(double scale) const {
    return JointDistribution(scale * mean, scale * scale * covariance);
  }

  JointDistribution operator+(const JointDistribution &other) const {
    return JointDistribution(mean + other.mean, covariance + other.covariance);
  }

  JointDistribution operator+(const MarginalDistribution &other) const {
    Eigen::MatrixXd new_covariance = covariance;
    new_covariance += other.covariance;
    return JointDistribution(mean + other.mean, new_covariance);
  }

  JointDistribution operator-(const JointDistribution &other) const {
    Eigen::MatrixXd new_covariance = covariance;
    new_covariance += other.covariance;
    return JointDistribution(mean - other.mean, new_covariance);
  }

  JointDistribution operator-(const MarginalDistribution &other) const {
    Eigen::MatrixXd new_covariance = covariance;
    new_covariance += other.covariance;
    return JointDistribution(mean - other.mean, new_covariance);
  }

  template <typename SizeType>
  JointDistribution subset(const std::vector<SizeType> &indices) const {
    return JointDistribution(albatross::subset(mean, indices),
                             symmetric_subset(covariance, indices));
  }

  MarginalDistribution marginal() const {
    return MarginalDistribution(mean, covariance.diagonal());
  }

  CovarianceType covariance;
};

template <typename SizeType, typename DistributionType>
inline DistributionType subset(const DistributionBase<DistributionType> &dist,
                               const std::vector<SizeType> &indices) {
  return dist.derived().subset(indices);
}

template <typename SizeType, typename DistributionType>
inline void set_subset(const DistributionBase<DistributionType> &from,
                       const std::vector<SizeType> &indices,
                       DistributionBase<DistributionType> *to) {
  to->derived().set_subset(from, indices);
}

inline MarginalDistribution
concatenate_marginals(const MarginalDistribution &x,
                      const MarginalDistribution &y) {
  return MarginalDistribution(
      concatenate(x.mean, y.mean),
      concatenate(x.covariance.diagonal(), y.covariance.diagonal()));
}

inline MarginalDistribution
concatenate_marginals(const std::vector<MarginalDistribution> &dists) {
  if (dists.size() == 0) {
    return MarginalDistribution();
  }

  Eigen::Index size = 0;
  for (const auto &dist : dists) {
    size += dist.mean.size();
  }

  Eigen::VectorXd mean(size);
  Eigen::VectorXd variance(size);
  Eigen::Index i = 0;
  for (const auto &dist : dists) {
    mean.middleRows(i, dist.mean.size()) = dist.mean;
    variance.middleRows(i, dist.mean.size()) = dist.covariance.diagonal();
    i += dist.mean.size();
  }

  return MarginalDistribution(mean, variance);
}

inline std::ostream &operator<<(std::ostream &os,
                                const MarginalDistribution &marginal) {
  for (Eigen::Index i = 0; i < cast::to_index(marginal.size()); ++i) {
    os << i << "    " << marginal.mean[i] << "   +/- "
       << std::sqrt(marginal.get_diagonal(i)) << std::endl;
  }
  return os;
}

inline std::ostream &operator<<(std::ostream &os,
                                const JointDistribution &joint) {
  Eigen::MatrixXd combined(joint.covariance.rows(),
                           joint.covariance.cols() + 1);
  combined.rightCols(joint.covariance.cols()) = joint.covariance;
  combined.col(0) = joint.mean;
  std::cout << combined;
  return os;
}

} // namespace albatross
#endif
