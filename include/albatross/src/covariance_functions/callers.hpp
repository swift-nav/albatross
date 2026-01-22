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

#ifndef ALBATROSS_COVARIANCE_FUNCTIONS_CALLERS_HPP_
#define ALBATROSS_COVARIANCE_FUNCTIONS_CALLERS_HPP_

namespace albatross {

/*
 * Implementing a CovarianceFunction requires defining a method with
 * signature:
 *
 *     double _call_impl(const X &x, const Y &y) const
 *
 * There are often different sets of arguments for which the covariance
 * function should be equivalent.  For example, we know that covariance
 * funcitons are symmetric, so a call to cov(x, y) will be equivalent
 * to cov(y, x).
 *
 * This and some additional desired behavior, the ability to distinguish
 * between a measurement and measurement noise free feature for example,
 * led to an intermediary step which use Callers.  These callers
 * can be strung together to avoid repeated trait inspection.
 */

/*
 * Cross covariance between two vectors of (possibly) different types.
 */
template <typename CovFuncCaller, typename X, typename Y>
inline Eigen::MatrixXd compute_covariance_matrix(CovFuncCaller caller,
                                                 const std::vector<X> &xs,
                                                 const std::vector<Y> &ys) {
  static_assert(is_invocable<CovFuncCaller, X, Y>::value,
                "caller does not support the required arguments");
  static_assert(is_invocable_with_result<CovFuncCaller, double, X, Y>::value,
                "caller does not return a double");
  Eigen::Index m = cast::to_index(xs.size());
  Eigen::Index n = cast::to_index(ys.size());
  Eigen::MatrixXd C(m, n);

  Eigen::Index i, j;
  std::size_t si, sj;
  for (i = 0; i < m; i++) {
    si = cast::to_size(i);
    for (j = 0; j < n; j++) {
      sj = cast::to_size(j);
      C(i, j) = caller(xs[si], ys[sj]);
    }
  }
  return C;
}

/*
 * Multithreaded cross covariance between two vectors of (possibly)
 * different types.
 */
template <typename CovFuncCaller, typename X, typename Y>
inline Eigen::MatrixXd
compute_covariance_matrix(CovFuncCaller caller, const std::vector<X> &xs,
                          const std::vector<Y> &ys, ThreadPool *pool) {
  static_assert(is_invocable<CovFuncCaller, X, Y>::value,
                "caller does not support the required arguments");
  static_assert(is_invocable_with_result<CovFuncCaller, double, X, Y>::value,
                "caller does not return a double");
  if (detail::should_serial_apply(pool)) {
    return compute_covariance_matrix(caller, xs, ys);
  }

  // Here we use `ceil()` because it should be faster to have one
  // block slightly smaller than slightly larger.
  const auto block_size = static_cast<Eigen::Index>(
      ceil(cast::to_double(ys.size()) / cast::to_double(pool->thread_count())));
  const auto num_rows = cast::to_index(xs.size());
  const auto num_cols = cast::to_index(ys.size());
  Eigen::MatrixXd output(num_rows, num_cols);

  const auto apply_block = [&](const Eigen::Index block_index) {
    const auto start = block_index * block_size;
    const auto end = std::min(num_cols, (block_index + 1) * block_size);
    for (Eigen::Index col = start; col < end; ++col) {
      const auto yidx = cast::to_size(col);
      for (Eigen::Index row = 0; row < num_rows; ++row) {
        const auto xidx = cast::to_size(row);
        output(row, col) = caller(xs[xidx], ys[yidx]);
      }
    }
  };
  std::vector<Eigen::Index> block_indices(pool->thread_count());
  std::iota(block_indices.begin(), block_indices.end(), 0);

  apply(block_indices, apply_block, pool);
  return output;
}

/*
 * Cross covariance between all elements of a vector.
 */
template <typename CovFuncCaller, typename X>
inline Eigen::MatrixXd compute_covariance_matrix(CovFuncCaller caller,
                                                 const std::vector<X> &xs) {
  static_assert(is_invocable<CovFuncCaller, X, X>::value,
                "caller does not support the required arguments");
  static_assert(is_invocable_with_result<CovFuncCaller, double, X, X>::value,
                "caller does not return a double");

  Eigen::Index n = cast::to_index(xs.size());
  Eigen::MatrixXd C(n, n);

  Eigen::Index i, j;
  std::size_t si, sj;
  for (i = 0; i < n; i++) {
    si = cast::to_size(i);
    for (j = 0; j <= i; j++) {
      sj = cast::to_size(j);
      C(i, j) = caller(xs[si], xs[sj]);
      C(j, i) = C(i, j);
    }
  }
  return C;
}

/*
 * Cross covariance between all elements of a vector.
 */
template <typename CovFuncCaller, typename X>
inline Eigen::MatrixXd compute_covariance_matrix(CovFuncCaller caller,
                                                 const std::vector<X> &xs,
                                                 ThreadPool *pool) {
  static_assert(is_invocable<CovFuncCaller, X, X>::value,
                "caller does not support the required arguments");
  static_assert(is_invocable_with_result<CovFuncCaller, double, X, X>::value,
                "caller does not return a double");
  if (detail::should_serial_apply(pool)) {
    return compute_covariance_matrix(caller, xs);
  }

  const auto size = cast::to_index(xs.size());
  Eigen::MatrixXd output(size, size);

  const auto apply_block = [&](const auto indices) {
    for (Eigen::Index col = indices.first; col < indices.second; ++col) {
      const auto vcol = cast::to_size(col);
      for (Eigen::Index row = 0; row <= col; ++row) {
        const auto vrow = cast::to_size(row);
        output(row, col) = caller(xs[vrow], xs[vcol]);
      }
    }
  };

  const auto block_count = cast::to_index(pool->thread_count());
  const auto blocks = detail::partition_triangular(size, block_count);
  apply(blocks, apply_block, pool);

  // Copy upper triangle to lower.
  output.triangularView<Eigen::Lower>() = output.transpose();
  return output;
}

/*
 * Mean of all elements of a vector.
 */
template <typename MeanFuncCaller, typename X>
inline Eigen::VectorXd compute_mean_vector(MeanFuncCaller caller,
                                           const std::vector<X> &xs) {
  static_assert(is_invocable<MeanFuncCaller, X>::value,
                "caller does not support the required arguments");
  static_assert(is_invocable_with_result<MeanFuncCaller, double, X>::value,
                "caller does not return a double");

  Eigen::Index n = cast::to_index(xs.size());
  Eigen::VectorXd m(n);

  Eigen::Index i;
  std::size_t si;
  for (i = 0; i < n; i++) {
    si = cast::to_size(i);
    m[i] = caller(xs[si]);
  }
  return m;
}

/*
 * Helper function to validate that all variants in a vector hold the same type.
 */
template <typename... Ts>
inline void
assert_homogeneous_variant_vector(const std::vector<variant<Ts...>> &variants) {
  if (variants.empty())
    return;

  const std::size_t expected_index = variants[0].which();
  for (std::size_t i = 1; i < variants.size(); ++i) {
    ALBATROSS_ASSERT(
        variants[i].which() == expected_index &&
        "All variants in a batch operation must hold the same type");
  }
}

/*
 * Helper function to unwrap a vector of Measurement<X> to a vector of X.
 */
template <typename X>
inline std::vector<X>
unwrap_measurements(const std::vector<Measurement<X>> &measurements) {
  std::vector<X> values;
  values.reserve(measurements.size());
  for (const auto &m : measurements) {
    values.push_back(m.value);
  }
  return values;
}

namespace internal {

/*
 * This Caller just directly call the underlying CovFunc.
 */
struct DirectCaller {
  // Covariance Functions - Pointwise
  template <typename CovFunc, typename X, typename Y,
            typename std::enable_if<has_valid_call_impl<CovFunc, X, Y>::value,
                                    int>::type = 0>
  static double call(const CovFunc &cov_func, const X &x, const Y &y) {
    return cov_func._call_impl(x, y);
  }

  // Covariance Functions - Batch (base case: pointwise loop)
  // Only enabled if pointwise _call_impl exists (batch-only covariances won't
  // reach here)
  template <typename CovFunc, typename X, typename Y,
            typename std::enable_if<has_valid_call_impl<CovFunc, X, Y>::value,
                                    int>::type = 0>
  static Eigen::MatrixXd
  call_vector(const CovFunc &cov_func, const std::vector<X> &xs,
              const std::vector<Y> &ys, ThreadPool *pool = nullptr) {
    auto caller = [&](const auto &x, const auto &y) {
      return cov_func._call_impl(x, y);
    };
    return compute_covariance_matrix(caller, xs, ys, pool);
  }

  // NOTE: If a covariance defines ONLY _call_impl_vector (no _call_impl),
  // then BatchCaller will catch it before we reach DirectCaller, so this
  // base case is never instantiated for batch-only covariances.

  // Symmetric batch (base case: pointwise loop using symmetry)
  template <typename CovFunc, typename X,
            typename std::enable_if<has_valid_call_impl<CovFunc, X, X>::value,
                                    int>::type = 0>
  static Eigen::MatrixXd call_vector(const CovFunc &cov_func,
                                     const std::vector<X> &xs,
                                     ThreadPool *pool = nullptr) {
    auto caller = [&](const auto &x, const auto &y) {
      return cov_func._call_impl(x, y);
    };
    return compute_covariance_matrix(caller, xs, pool);
  }

  // Diagonal batch (base case: pointwise loop)
  template <typename CovFunc, typename X,
            typename std::enable_if<has_valid_call_impl<CovFunc, X, X>::value,
                                    int>::type = 0>
  static Eigen::VectorXd
  call_vector_diagonal(const CovFunc &cov_func, const std::vector<X> &xs,
                       [[maybe_unused]] ThreadPool *pool = nullptr) {
    const Eigen::Index n = cast::to_index(xs.size());
    Eigen::VectorXd diag(n);
    for (Eigen::Index i = 0; i < n; ++i) {
      const auto si = cast::to_size(i);
      diag[i] = cov_func._call_impl(xs[si], xs[si]);
    }
    return diag;
  }

  // Mean Functions
  template <typename MeanFunc, typename X,
            typename std::enable_if<has_valid_call_impl<MeanFunc, X>::value,
                                    int>::type = 0>
  static double call(const MeanFunc &mean_func, const X &x) {
    return mean_func._call_impl(x);
  }
};

/*
 * This Caller turns any CovFunc defined for argument types X, Y into
 * one valid for Y, X as well.
 */
template <typename SubCaller> struct SymmetricCaller {
  // Covariance Functions

  // CovFunc has a direct call implementation for X and Y
  template <
      typename CovFunc, typename X, typename Y,
      typename std::enable_if<
          has_valid_cov_caller<CovFunc, SubCaller, X, Y>::value, int>::type = 0>
  static double call(const CovFunc &cov_func, const X &x, const Y &y) {
    return SubCaller::call(cov_func, x, y);
  }

  // CovFunc has a call for Y and X but not X and Y
  template <typename CovFunc, typename X, typename Y,
            typename std::enable_if<
                (has_valid_cov_caller<CovFunc, SubCaller, Y, X>::value &&
                 !has_valid_cov_caller<CovFunc, SubCaller, X, Y>::value),
                int>::type = 0>
  static double call(const CovFunc &cov_func, const X &x, const Y &y) {
    return SubCaller::call(cov_func, y, x);
  }

  // Batch Covariance Functions - cross-covariance passthrough
  template <typename CovFunc, typename X, typename Y>
  static Eigen::MatrixXd
  call_vector(const CovFunc &cov_func, const std::vector<X> &xs,
              const std::vector<Y> &ys, ThreadPool *pool = nullptr) {
    return SubCaller::call_vector(cov_func, xs, ys, pool);
  }

  // Symmetric passthrough
  template <typename CovFunc, typename X>
  static Eigen::MatrixXd call_vector(const CovFunc &cov_func,
                                     const std::vector<X> &xs,
                                     ThreadPool *pool = nullptr) {
    return SubCaller::call_vector(cov_func, xs, pool);
  }

  // Diagonal passthrough
  template <typename CovFunc, typename X>
  static Eigen::VectorXd call_vector_diagonal(const CovFunc &cov_func,
                                              const std::vector<X> &xs,
                                              ThreadPool *pool = nullptr) {
    return SubCaller::call_vector_diagonal(cov_func, xs, pool);
  }

  // Mean Functions
  template <
      typename MeanFunc, typename X,
      typename std::enable_if<
          has_valid_mean_caller<MeanFunc, SubCaller, X>::value, int>::type = 0>
  static double call(const MeanFunc &mean_func, const X &x) {
    return SubCaller::call(mean_func, x);
  }
};

/*
 * This Caller maps any call with type Measurement<X> to one which
 * just uses the underlying type (X) UNLESS a call is actually defined
 * for Measurement<X> in which case that is used.  This makes it possible
 * to define a covariance function which behaves differently when presented
 * with training data (measurements) versus test data where you may
 * actually be interested in the underlying process.
 */
template <typename SubCaller> struct MeasurementForwarder {
  // Covariance Functions
  template <
      typename CovFunc, typename X, typename Y,
      typename std::enable_if<
          has_valid_cov_caller<CovFunc, SubCaller, X, Y>::value, int>::type = 0>
  static double call(const CovFunc &cov_func, const X &x, const Y &y) {
    return SubCaller::call(cov_func, x, y);
  }

  template <typename CovFunc, typename X, typename Y,
            typename std::enable_if<
                (has_valid_cov_caller<CovFunc, SubCaller, X, Y>::value &&
                 !has_valid_cov_caller<CovFunc, SubCaller, Measurement<X>,
                                       Measurement<Y>>::value),
                int>::type = 0>
  static double call(const CovFunc &cov_func, const Measurement<X> &x,
                     const Measurement<Y> &y) {
    return SubCaller::call(cov_func, x.value, y.value);
  }

  template <
      typename CovFunc, typename X, typename Y,
      typename std::enable_if<
          (has_valid_cov_caller<CovFunc, SubCaller, X, Y>::value &&
           !has_valid_cov_caller<CovFunc, SubCaller, Measurement<X>, Y>::value),
          int>::type = 0>
  static double call(const CovFunc &cov_func, const Measurement<X> &x,
                     const Y &y) {
    return SubCaller::call(cov_func, x.value, y);
  }

  template <
      typename CovFunc, typename X, typename Y,
      typename std::enable_if<
          (has_valid_cov_caller<CovFunc, SubCaller, X, Y>::value &&
           !has_valid_cov_caller<CovFunc, SubCaller, X, Measurement<Y>>::value),
          int>::type = 0>
  static double call(const CovFunc &cov_func, const X &x,
                     const Measurement<Y> &y) {
    return SubCaller::call(cov_func, x, y.value);
  }

  // Batch Covariance Functions

  // Passthrough when no Measurement types
  template <typename CovFunc, typename X, typename Y,
            typename std::enable_if<!is_measurement<X>::value &&
                                        !is_measurement<Y>::value,
                                    int>::type = 0>
  static Eigen::MatrixXd
  call_vector(const CovFunc &cov_func, const std::vector<X> &xs,
              const std::vector<Y> &ys, ThreadPool *pool = nullptr) {
    return SubCaller::call_vector(cov_func, xs, ys, pool);
  }

  // Unwrap Measurement types using pointwise loop (matches old behavior
  // exactly) We use pointwise here rather than unwrap+delegate because it
  // ensures exact numerical equivalence with the original code, avoiding test
  // failures.
  template <
      typename CovFunc, typename X, typename Y,
      typename std::enable_if<
          is_measurement<X>::value || is_measurement<Y>::value, int>::type = 0>
  static Eigen::MatrixXd
  call_vector(const CovFunc &cov_func, const std::vector<X> &xs,
              const std::vector<Y> &ys, ThreadPool *pool = nullptr) {
    auto caller = [&](const auto &x, const auto &y) {
      return MeasurementForwarder::call(cov_func, x, y);
    };
    return compute_covariance_matrix(caller, xs, ys, pool);
  }

  // Symmetric: passthrough for non-Measurements
  template <typename CovFunc, typename X,
            typename std::enable_if<!is_measurement<X>::value, int>::type = 0>
  static Eigen::MatrixXd call_vector(const CovFunc &cov_func,
                                     const std::vector<X> &xs,
                                     ThreadPool *pool = nullptr) {
    return SubCaller::call_vector(cov_func, xs, pool);
  }

  // Symmetric: pointwise for Measurements
  template <typename CovFunc, typename X,
            typename std::enable_if<is_measurement<X>::value, int>::type = 0>
  static Eigen::MatrixXd call_vector(const CovFunc &cov_func,
                                     const std::vector<X> &xs,
                                     ThreadPool *pool = nullptr) {
    auto caller = [&](const auto &x, const auto &y) {
      return MeasurementForwarder::call(cov_func, x, y);
    };
    return compute_covariance_matrix(caller, xs, pool);
  }

  // Diagonal: passthrough for non-Measurements
  template <typename CovFunc, typename X,
            typename std::enable_if<!is_measurement<X>::value, int>::type = 0>
  static Eigen::VectorXd call_vector_diagonal(const CovFunc &cov_func,
                                              const std::vector<X> &xs,
                                              ThreadPool *pool = nullptr) {
    return SubCaller::call_vector_diagonal(cov_func, xs, pool);
  }

  // Diagonal: pointwise for Measurements
  template <typename CovFunc, typename X,
            typename std::enable_if<is_measurement<X>::value, int>::type = 0>
  static Eigen::VectorXd
  call_vector_diagonal(const CovFunc &cov_func, const std::vector<X> &xs,
                       [[maybe_unused]] ThreadPool *pool = nullptr) {
    const Eigen::Index n = cast::to_index(xs.size());
    Eigen::VectorXd diag(n);
    for (Eigen::Index i = 0; i < n; ++i) {
      const auto si = cast::to_size(i);
      diag[i] = MeasurementForwarder::call(cov_func, xs[si], xs[si]);
    }
    return diag;
  }

  // Mean Functions
  template <
      typename MeanFunc, typename X,
      typename std::enable_if<
          has_valid_mean_caller<MeanFunc, SubCaller, X>::value, int>::type = 0>
  static double call(const MeanFunc &mean_func, const X &x) {
    return SubCaller::call(mean_func, x);
  }

  template <typename MeanFunc, typename X,
            typename std::enable_if<
                has_valid_mean_caller<MeanFunc, SubCaller, X>::value &&
                    !has_valid_mean_caller<MeanFunc, SubCaller,
                                           Measurement<X>>::value,
                int>::type = 0>
  static double call(const MeanFunc &mean_func, const Measurement<X> &x) {
    return SubCaller::call(mean_func, x.value);
  }
};

template <typename SubCaller> struct LinearCombinationCaller {
  // Covariance Functions
  template <
      typename CovFunc, typename X, typename Y,
      typename std::enable_if<
          has_valid_cov_caller<CovFunc, SubCaller, X, Y>::value, int>::type = 0>
  static double call(const CovFunc &cov_func, const X &x, const Y &y) {
    return SubCaller::call(cov_func, x, y);
  }

  template <
      typename CovFunc, typename X, typename Y,
      typename std::enable_if<
          has_valid_cov_caller<CovFunc, SubCaller, X, Y>::value, int>::type = 0>
  static double call(const CovFunc &cov_func, const LinearCombination<X> &xs,
                     const LinearCombination<Y> &ys) {
    auto sub_caller = [&](const auto &x, const auto &y) {
      return SubCaller::call(cov_func, x, y);
    };

    const auto mat =
        compute_covariance_matrix(sub_caller, xs.values, ys.values);
    return xs.coefficients.dot(mat * ys.coefficients);
  }

  template <
      typename CovFunc, typename X, typename Y,
      typename std::enable_if<
          has_valid_cov_caller<CovFunc, SubCaller, X, Y>::value, int>::type = 0>
  static double call(const CovFunc &cov_func, const X &x,
                     const LinearCombination<Y> &ys) {
    double sum = 0.;
    for (std::size_t i = 0; i < ys.values.size(); ++i) {
      sum += ys.coefficients[cast::to_index(i)] *
             SubCaller::call(cov_func, x, ys.values[i]);
    }
    return sum;
  }

  template <
      typename CovFunc, typename X, typename Y,
      typename std::enable_if<
          has_valid_cov_caller<CovFunc, SubCaller, X, Y>::value, int>::type = 0>
  static double call(const CovFunc &cov_func, const LinearCombination<X> &xs,
                     const Y &y) {
    double sum = 0.;
    for (std::size_t i = 0; i < xs.values.size(); ++i) {
      sum += xs.coefficients[cast::to_index(i)] *
             SubCaller::call(cov_func, xs.values[i], y);
    }
    return sum;
  }

  // Batch Covariance Functions

  // Passthrough when no LinearCombination types
  template <typename CovFunc, typename X, typename Y,
            typename std::enable_if<!is_linear_combination<X>::value &&
                                        !is_linear_combination<Y>::value,
                                    int>::type = 0>
  static Eigen::MatrixXd
  call_vector(const CovFunc &cov_func, const std::vector<X> &xs,
              const std::vector<Y> &ys, ThreadPool *pool = nullptr) {
    return SubCaller::call_vector(cov_func, xs, ys, pool);
  }

  // Handle LinearCombination types by building matrix pointwise (like old code)
  template <typename CovFunc, typename X, typename Y,
            typename std::enable_if<is_linear_combination<X>::value ||
                                        is_linear_combination<Y>::value,
                                    int>::type = 0>
  static Eigen::MatrixXd
  call_vector(const CovFunc &cov_func, const std::vector<X> &xs,
              const std::vector<Y> &ys, ThreadPool *pool = nullptr) {
    auto caller = [&](const auto &x, const auto &y) {
      return LinearCombinationCaller::call(cov_func, x, y);
    };
    return compute_covariance_matrix(caller, xs, ys, pool);
  }

  // Symmetric: passthrough for non-LinearCombinations
  template <
      typename CovFunc, typename X,
      typename std::enable_if<!is_linear_combination<X>::value, int>::type = 0>
  static Eigen::MatrixXd call_vector(const CovFunc &cov_func,
                                     const std::vector<X> &xs,
                                     ThreadPool *pool = nullptr) {
    return SubCaller::call_vector(cov_func, xs, pool);
  }

  // Symmetric: pointwise for LinearCombinations
  template <
      typename CovFunc, typename X,
      typename std::enable_if<is_linear_combination<X>::value, int>::type = 0>
  static Eigen::MatrixXd call_vector(const CovFunc &cov_func,
                                     const std::vector<X> &xs,
                                     ThreadPool *pool = nullptr) {
    auto caller = [&](const auto &x, const auto &y) {
      return LinearCombinationCaller::call(cov_func, x, y);
    };
    return compute_covariance_matrix(caller, xs, pool);
  }

  // Diagonal: passthrough for non-LinearCombinations
  template <
      typename CovFunc, typename X,
      typename std::enable_if<!is_linear_combination<X>::value, int>::type = 0>
  static Eigen::VectorXd call_vector_diagonal(const CovFunc &cov_func,
                                              const std::vector<X> &xs,
                                              ThreadPool *pool = nullptr) {
    return SubCaller::call_vector_diagonal(cov_func, xs, pool);
  }

  // Diagonal: pointwise for LinearCombinations
  template <
      typename CovFunc, typename X,
      typename std::enable_if<is_linear_combination<X>::value, int>::type = 0>
  static Eigen::VectorXd
  call_vector_diagonal(const CovFunc &cov_func, const std::vector<X> &xs,
                       [[maybe_unused]] ThreadPool *pool = nullptr) {
    const Eigen::Index n = cast::to_index(xs.size());
    Eigen::VectorXd diag(n);
    for (Eigen::Index i = 0; i < n; ++i) {
      const auto si = cast::to_size(i);
      diag[i] = LinearCombinationCaller::call(cov_func, xs[si], xs[si]);
    }
    return diag;
  }

  // Mean Functions
  template <
      typename MeanFunc, typename X,
      typename std::enable_if<
          has_valid_mean_caller<MeanFunc, SubCaller, X>::value, int>::type = 0>
  static double call(const MeanFunc &mean_func, const X &x) {
    return SubCaller::call(mean_func, x);
  }

  template <
      typename MeanFunc, typename X,
      typename std::enable_if<
          has_valid_mean_caller<MeanFunc, SubCaller, X>::value, int>::type = 0>
  static double call(const MeanFunc &mean_func,
                     const LinearCombination<X> &xs) {
    double sum = 0.;
    for (std::size_t i = 0; i < xs.values.size(); ++i) {
      sum += xs.coefficients[cast::to_index(i)] *
             SubCaller::call(mean_func, xs.values[i]);
    }
    return sum;
  }
};

/*
 * VariantForwarder
 *
 * The variant forwarder allows covariance functions to be called with
 * variant types, without explicitly handling them.  For example, a
 * covariance function which has methods:
 *
 *   double _call_impl(const X&, const X&) const;
 *
 *   double _call_impl(const X&, const Y&) const;
 *
 *   double _call_impl(const Y&, const Y&) const;
 *
 * would then also support the following pairs of arguments:
 *
 *   variant<X, Y>, X
 *   X, variant<X, Y>
 *   variant<X, Y>, Y
 *   Y, variant<X, Y>
 *
 */
template <typename SubCaller> struct VariantForwarder {
  // Covariance Functions - Pointwise

  // directly forward on the case where both types aren't variants
  template <typename CovFunc, typename X, typename Y,
            typename std::enable_if<
                !is_variant<X>::value && !is_variant<Y>::value &&
                    has_valid_cov_caller<CovFunc, SubCaller, X, Y>::value,
                int>::type = 0>
  static double call(const CovFunc &cov_func, const X &x, const Y &y) {
    return SubCaller::call(cov_func, x, y);
  }

  // Covariance Functions - Batch

  // Passthrough when no variants
  template <typename CovFunc, typename X, typename Y,
            typename std::enable_if<
                !is_variant<X>::value && !is_variant<Y>::value, int>::type = 0>
  static Eigen::MatrixXd
  call_vector(const CovFunc &cov_func, const std::vector<X> &xs,
              const std::vector<Y> &ys, ThreadPool *pool = nullptr) {
    return SubCaller::call_vector(cov_func, xs, ys, pool);
  }

  // Variants - use pointwise with visitor (like old code)
  template <typename CovFunc, typename X, typename Y,
            typename std::enable_if<
                is_variant<X>::value || is_variant<Y>::value, int>::type = 0>
  static Eigen::MatrixXd
  call_vector(const CovFunc &cov_func, const std::vector<X> &xs,
              const std::vector<Y> &ys, ThreadPool *pool = nullptr) {
    auto caller = [&](const auto &x, const auto &y) {
      return VariantForwarder::call(cov_func, x, y);
    };
    return compute_covariance_matrix(caller, xs, ys, pool);
  }

  // Symmetric: passthrough for non-variants
  template <typename CovFunc, typename X,
            typename std::enable_if<!is_variant<X>::value, int>::type = 0>
  static Eigen::MatrixXd call_vector(const CovFunc &cov_func,
                                     const std::vector<X> &xs,
                                     ThreadPool *pool = nullptr) {
    return SubCaller::call_vector(cov_func, xs, pool);
  }

  // Symmetric: pointwise for variants
  template <typename CovFunc, typename X,
            typename std::enable_if<is_variant<X>::value, int>::type = 0>
  static Eigen::MatrixXd call_vector(const CovFunc &cov_func,
                                     const std::vector<X> &xs,
                                     ThreadPool *pool = nullptr) {
    auto caller = [&](const auto &x, const auto &y) {
      return VariantForwarder::call(cov_func, x, y);
    };
    return compute_covariance_matrix(caller, xs, pool);
  }

  // Diagonal: passthrough for non-variants
  template <typename CovFunc, typename X,
            typename std::enable_if<!is_variant<X>::value, int>::type = 0>
  static Eigen::VectorXd call_vector_diagonal(const CovFunc &cov_func,
                                              const std::vector<X> &xs,
                                              ThreadPool *pool = nullptr) {
    return SubCaller::call_vector_diagonal(cov_func, xs, pool);
  }

  // Diagonal: pointwise for variants
  template <typename CovFunc, typename X,
            typename std::enable_if<is_variant<X>::value, int>::type = 0>
  static Eigen::VectorXd
  call_vector_diagonal(const CovFunc &cov_func, const std::vector<X> &xs,
                       [[maybe_unused]] ThreadPool *pool = nullptr) {
    const Eigen::Index n = cast::to_index(xs.size());
    Eigen::VectorXd diag(n);
    for (Eigen::Index i = 0; i < n; ++i) {
      const auto si = cast::to_size(i);
      diag[i] = VariantForwarder::call(cov_func, xs[si], xs[si]);
    }
    return diag;
  }

  /*
   * This visitor helps deal with enabling and disabling the call operator
   * depending on whether pairs of types in variants are defined.
   */
  template <typename CovFunc, typename X> struct CallVisitor {
    CallVisitor(const CovFunc &cov_func, const X &x)
        : cov_func_(cov_func), x_(x) {}

    template <typename Y,
              typename std::enable_if<
                  has_valid_cov_caller<CovFunc, SubCaller, X, Y>::value,
                  int>::type = 0>
    double operator()(const Y &y) const {
      return SubCaller::call(cov_func_, x_, y);
    }

    template <typename Y,
              typename std::enable_if<
                  !has_valid_cov_caller<CovFunc, SubCaller, X, Y>::value,
                  int>::type = 0>
    double operator()(const Y &y ALBATROSS_UNUSED) const {
      return 0.;
    }

    const CovFunc &cov_func_;
    const X &x_;
  };

  // Deals with a type, A, and an arbitrary variant.
  template <typename CovFunc, typename A, typename... Ts,
            typename std::enable_if<
                !is_variant<A>::value &&
                    has_valid_variant_cov_caller<CovFunc, SubCaller, A,
                                                 variant<Ts...>>::value,
                int>::type = 0>
  static double call(const CovFunc &cov_func, const A &x,
                     const variant<Ts...> &y) {
    CallVisitor<CovFunc, A> visitor(cov_func, x);
    return apply_visitor(visitor, y);
  }

  // Deals with a type, A, and an arbitrary variant.
  template <typename CovFunc, typename A, typename... Ts,
            typename std::enable_if<
                !is_variant<A>::value &&
                    has_valid_variant_cov_caller<CovFunc, SubCaller, A,
                                                 variant<Ts...>>::value,
                int>::type = 0>
  static double call(const CovFunc &cov_func, const variant<Ts...> &x,
                     const A &y) {
    CallVisitor<CovFunc, A> visitor(cov_func, y);
    return apply_visitor(visitor, x);
  }

  // Deals with two variants.
  template <typename CovFunc, typename... Xs, typename... Ys,
            typename std::enable_if<
                has_valid_cov_caller<CovFunc, SubCaller, variant<Xs...>,
                                     variant<Ys...>>::value,
                int>::type = 0>
  static double call(const CovFunc &cov_func, const variant<Xs...> &x,
                     const variant<Ys...> &y) {
    return x.match(
        [&y, &cov_func](const auto &xx) { return call(cov_func, xx, y); });
  }

  // Mean Functions

  /*
   * This visitor helps deal with enabling and disabling the call operator
   * depending on whether pairs of types in variants are defined.
   */
  template <typename MeanFunc> struct MeanCallVisitor {
    MeanCallVisitor(const MeanFunc &mean_func) : mean_func_(mean_func) {}

    template <typename X,
              typename std::enable_if<
                  has_valid_mean_caller<MeanFunc, SubCaller, X>::value,
                  int>::type = 0>
    double operator()(const X &x) const {
      return SubCaller::call(mean_func_, x);
    }

    template <typename X,
              typename std::enable_if<
                  !has_valid_mean_caller<MeanFunc, SubCaller, X>::value,
                  int>::type = 0>
    double operator()(const X &x ALBATROSS_UNUSED) const {
      return 0.;
    }

    const MeanFunc &mean_func_;
  };

  template <typename MeanFunc, typename X,
            typename std::enable_if<
                !is_variant<X>::value &&
                    has_valid_mean_caller<MeanFunc, SubCaller, X>::value,
                int>::type = 0>
  static double call(const MeanFunc &mean_func, const X &x) {
    return SubCaller::call(mean_func, x);
  }

  template <typename MeanFunc, typename X,
            typename std::enable_if<is_variant<X>::value &&
                                        has_valid_variant_mean_caller<
                                            MeanFunc, SubCaller, X>::value,
                                    int>::type = 0>
  static double call(const MeanFunc &mean_func, const X &x) {
    MeanCallVisitor<MeanFunc> visitor(mean_func);
    return apply_visitor(visitor, x);
  }
};

/*
 * BatchCaller - attempts batch covariance via _call_impl_vector
 */
template <typename SubCaller> struct BatchCaller {
  // Batch method exists - use it directly
  template <
      typename CovFunc, typename X, typename Y,
      typename std::enable_if<has_valid_call_impl_vector<CovFunc, X, Y>::value,
                              int>::type = 0>
  static Eigen::MatrixXd
  call_vector(const CovFunc &cov_func, const std::vector<X> &xs,
              const std::vector<Y> &ys, ThreadPool *pool = nullptr) {
    return cov_func._call_impl_vector(xs, ys, pool);
  }

  // No batch - build matrix pointwise using SubCaller
  template <typename CovFunc, typename X, typename Y,
            typename std::enable_if<
                !has_valid_call_impl_vector<CovFunc, X, Y>::value &&
                    has_valid_cov_caller<CovFunc, SubCaller, X, Y>::value,
                int>::type = 0>
  static Eigen::MatrixXd
  call_vector(const CovFunc &cov_func, const std::vector<X> &xs,
              const std::vector<Y> &ys, ThreadPool *pool = nullptr) {
    auto caller = [&](const auto &x, const auto &y) {
      return SubCaller::call(cov_func, x, y);
    };
    return compute_covariance_matrix(caller, xs, ys, pool);
  }

  // Symmetric batch: use _call_impl_vector when available
  template <typename CovFunc, typename X,
            typename std::enable_if<
                has_valid_call_impl_vector_symmetric<CovFunc, X>::value,
                int>::type = 0>
  static Eigen::MatrixXd call_vector(const CovFunc &cov_func,
                                     const std::vector<X> &xs,
                                     ThreadPool *pool = nullptr) {
    return cov_func._call_impl_vector(xs, xs, pool);
  }

  // Symmetric no batch - pointwise loop
  template <typename CovFunc, typename X,
            typename std::enable_if<
                !has_valid_call_impl_vector_symmetric<CovFunc, X>::value &&
                    has_valid_cov_caller<CovFunc, SubCaller, X, X>::value,
                int>::type = 0>
  static Eigen::MatrixXd call_vector(const CovFunc &cov_func,
                                     const std::vector<X> &xs,
                                     ThreadPool *pool = nullptr) {
    auto caller = [&](const auto &x, const auto &y) {
      return SubCaller::call(cov_func, x, y);
    };
    return compute_covariance_matrix(caller, xs, pool);
  }

  // Primary: use batch when available (synthesize scalar from batch)
  // This ensures consistency between scalar and matrix calls.
  template <
      typename CovFunc, typename X, typename Y,
      typename std::enable_if<has_valid_call_impl_vector<CovFunc, X, Y>::value,
                              int>::type = 0>
  static double call(const CovFunc &cov_func, const X &x, const Y &y) {
    const std::vector<X> xs{x};
    const std::vector<Y> ys{y};
    return cov_func._call_impl_vector(xs, ys, nullptr)(0, 0);
  }

  // Fallback: use pointwise when batch is not available
  template <typename CovFunc, typename X, typename Y,
            typename std::enable_if<
                (!has_valid_call_impl_vector<CovFunc, X, Y>::value &&
                 has_valid_cov_caller<CovFunc, SubCaller, X, Y>::value),
                int>::type = 0>
  static double call(const CovFunc &cov_func, const X &x, const Y &y) {
    return SubCaller::call(cov_func, x, y);
  }

  // Mean function pass-through
  template <
      typename MeanFunc, typename X,
      typename std::enable_if<
          has_valid_mean_caller<MeanFunc, SubCaller, X>::value, int>::type = 0>
  static double call(const MeanFunc &mean_func, const X &x) {
    return SubCaller::call(mean_func, x);
  }

  // Diagonal batch: use _call_impl_vector_diagonal when available
  template <typename CovFunc, typename X,
            typename std::enable_if<
                has_valid_call_impl_vector_diagonal<CovFunc, X>::value,
                int>::type = 0>
  static Eigen::VectorXd call_vector_diagonal(const CovFunc &cov_func,
                                              const std::vector<X> &xs,
                                              ThreadPool *pool = nullptr) {
    return cov_func._call_impl_vector_diagonal(xs, pool);
  }

  // Diagonal no batch - extract diagonal pointwise (when SubCaller supports it)
  template <typename CovFunc, typename X,
            typename std::enable_if<
                !has_valid_call_impl_vector_diagonal<CovFunc, X>::value &&
                    has_valid_cov_caller<CovFunc, SubCaller, X, X>::value,
                int>::type = 0>
  static Eigen::VectorXd
  call_vector_diagonal(const CovFunc &cov_func, const std::vector<X> &xs,
                       [[maybe_unused]] ThreadPool *pool = nullptr) {
    const Eigen::Index n = cast::to_index(xs.size());
    Eigen::VectorXd diag(n);
    for (Eigen::Index i = 0; i < n; ++i) {
      const auto si = cast::to_size(i);
      diag[i] = SubCaller::call(cov_func, xs[si], xs[si]);
    }
    return diag;
  }

  // Diagonal for batch-only covariances - synthesize from _call_impl_vector
  template <typename CovFunc, typename X,
            typename std::enable_if<
                !has_valid_call_impl_vector_diagonal<CovFunc, X>::value &&
                    !has_valid_cov_caller<CovFunc, SubCaller, X, X>::value &&
                    has_valid_call_impl_vector<CovFunc, X, X>::value,
                int>::type = 0>
  static Eigen::VectorXd call_vector_diagonal(const CovFunc &cov_func,
                                              const std::vector<X> &xs,
                                              ThreadPool *pool = nullptr) {
    // Extract diagonal from full matrix
    // This is less efficient but correct for batch-only covariances
    return cov_func._call_impl_vector(xs, xs, pool).diagonal();
  }
};

} // namespace internal

/*
 * This defines the order of operations of the covariance function Callers.
 */
using DefaultCaller = internal::VariantForwarder<
    internal::MeasurementForwarder<internal::LinearCombinationCaller<
        internal::BatchCaller<internal::VariantForwarder<
            internal::SymmetricCaller<internal::DirectCaller>>>>>>;

template <typename Caller, typename CovFunc, typename... Args>
class caller_has_valid_call
    : public has_call_with_return_type<Caller, double,
                                       typename const_ref<CovFunc>::type,
                                       typename const_ref<Args>::type...> {};

template <typename CovFunc, typename... Args>
class has_valid_caller
    : public caller_has_valid_call<DefaultCaller, CovFunc, Args...> {};

/*
 * Check if EITHER pointwise OR batch is available for a type pair.
 * This allows operator() to be enabled for batch-only covariances.
 */
template <typename CovFunc, typename X, typename Y>
class has_valid_caller_or_batch {
public:
  static constexpr bool value =
      has_valid_caller<CovFunc, X, Y>::value ||
      has_valid_call_impl_vector<CovFunc, X, Y>::value;
};

// Symmetric version: check if pointwise OR symmetric batch is available
template <typename CovFunc, typename X>
class has_valid_caller_or_batch_symmetric {
public:
  static constexpr bool value =
      has_valid_caller<CovFunc, X, X>::value ||
      has_valid_call_impl_vector_symmetric<CovFunc, X>::value;
};

// Diagonal version: check if pointwise OR diagonal batch is available
template <typename CovFunc, typename X>
class has_valid_caller_or_batch_diagonal {
public:
  static constexpr bool value =
      has_valid_caller<CovFunc, X, X>::value ||
      has_valid_call_impl_vector_diagonal<CovFunc, X>::value;
};

/*
 * This defines a helper trait which indicates whether a call to a
 * function has an equivalent call. This is different from the valid
 * caller which might do more complicated operations such as unpacking
 * a variant, or integrating over linear combinations, here we just want
 * to know if there is a call method available for types such as
 * the Measurement<> wrapper for which we'd happily unwrap the type.
 *
 * For example if we have a function CovFunc for which we have defined:
 *
 *   double _call_impl(const X &x, const Y &y) const;
 *
 * We would expect
 *
 *   has_equivalent_caller<CovFunc, Measurement<X>, Y>::value == true
 *   has_equivalent_caller<CovFunc, LinearCombination<X>, Y>::value == false
 *   has_equivalent_caller<CovFunc, variant<X, Y>, Y>::value == false
 *
 * while has_valid_caller should evaluate true in both cases, but only the
 * first is equivalent, the others _could_ be equivalent but aren't
 * neccesarily.
 *
 * Note: BatchCaller is included to support batch-only covariances that
 * define _call_impl_vector but not _call_impl. BatchCaller can synthesize
 * pointwise results from the batch method.
 */
using EquivalentCaller = internal::MeasurementForwarder<
    internal::BatchCaller<internal::SymmetricCaller<internal::DirectCaller>>>;

template <typename CovFunc, typename... Args>
class has_equivalent_caller
    : public caller_has_valid_call<EquivalentCaller, CovFunc, Args...> {};

} // namespace albatross

#endif /* ALBATROSS_COVARIANCE_FUNCTIONS_CALLERS_HPP_ */
