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
 * Helper function to check if all variants in a vector hold the same type.
 * Returns true if homogeneous (including empty), false otherwise.
 */
template <typename... Ts>
inline bool
is_homogeneous_variant_vector(const std::vector<variant<Ts...>> &variants) {
  if (variants.empty())
    return true;

  const std::size_t expected_index = variants[0].which();
  for (std::size_t i = 1; i < variants.size(); ++i) {
    if (variants[i].which() != expected_index) {
      return false;
    }
  }
  return true;
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

  // Both sides Measurement: delegate to SubCaller if covariance has
  // _call_impl_vector for Measurement types directly. This includes
  // Sum/Product which define _call_impl_vector<Measurement<X>>.
  template <typename CovFunc, typename X, typename Y,
            typename std::enable_if<
                is_measurement<X>::value && is_measurement<Y>::value &&
                    has_valid_call_impl_vector<CovFunc, X, Y>::value,
                int>::type = 0>
  static Eigen::MatrixXd
  call_vector(const CovFunc &cov_func, const std::vector<X> &xs,
              const std::vector<Y> &ys, ThreadPool *pool = nullptr) {
    return SubCaller::call_vector(cov_func, xs, ys, pool);
  }

  // Both sides Measurement: unwrap if no _call_impl_vector for Measurement,
  // but the underlying type has batch support.
  template <typename CovFunc, typename X, typename Y,
            typename std::enable_if<
                is_measurement<X>::value && is_measurement<Y>::value &&
                    !has_valid_call_impl_vector<CovFunc, X, Y>::value &&
                    !has_valid_call_impl<CovFunc, X, Y>::value,
                int>::type = 0>
  static Eigen::MatrixXd
  call_vector(const CovFunc &cov_func, const std::vector<X> &xs,
              const std::vector<Y> &ys, ThreadPool *pool = nullptr) {
    auto unwrapped_xs = unwrap_measurements(xs);
    auto unwrapped_ys = unwrap_measurements(ys);
    return SubCaller::call_vector(cov_func, unwrapped_xs, unwrapped_ys, pool);
  }

  // Both sides Measurement: use pointwise if covariance function has explicit
  // _call_impl for Measurement types (e.g. IndependentNoise which needs to
  // distinguish Measurement self-covariance from cross-covariance) but no
  // _call_impl_vector for Measurement types.
  template <typename CovFunc, typename X, typename Y,
            typename std::enable_if<
                is_measurement<X>::value && is_measurement<Y>::value &&
                    !has_valid_call_impl_vector<CovFunc, X, Y>::value &&
                    has_valid_call_impl<CovFunc, X, Y>::value,
                int>::type = 0>
  static Eigen::MatrixXd
  call_vector(const CovFunc &cov_func, const std::vector<X> &xs,
              const std::vector<Y> &ys, ThreadPool *pool = nullptr) {
    // Covariance function has special handling for Measurement types,
    // so we must use pointwise to respect that behavior
    auto caller = [&](const auto &x, const auto &y) {
      return MeasurementForwarder::call(cov_func, x, y);
    };
    return compute_covariance_matrix(caller, xs, ys, pool);
  }

  // Left side Measurement: delegate to SubCaller if _call_impl_vector exists
  template <typename CovFunc, typename X, typename Y,
            typename std::enable_if<
                is_measurement<X>::value && !is_measurement<Y>::value &&
                    has_valid_call_impl_vector<CovFunc, X, Y>::value,
                int>::type = 0>
  static Eigen::MatrixXd
  call_vector(const CovFunc &cov_func, const std::vector<X> &xs,
              const std::vector<Y> &ys, ThreadPool *pool = nullptr) {
    return SubCaller::call_vector(cov_func, xs, ys, pool);
  }

  // Left side Measurement: unwrap left if no _call_impl_vector, no _call_impl
  template <typename CovFunc, typename X, typename Y,
            typename std::enable_if<
                is_measurement<X>::value && !is_measurement<Y>::value &&
                    !has_valid_call_impl_vector<CovFunc, X, Y>::value &&
                    !has_valid_call_impl<CovFunc, X, Y>::value,
                int>::type = 0>
  static Eigen::MatrixXd
  call_vector(const CovFunc &cov_func, const std::vector<X> &xs,
              const std::vector<Y> &ys, ThreadPool *pool = nullptr) {
    auto unwrapped_xs = unwrap_measurements(xs);
    return SubCaller::call_vector(cov_func, unwrapped_xs, ys, pool);
  }

  // Left side Measurement: pointwise if _call_impl but no _call_impl_vector
  template <typename CovFunc, typename X, typename Y,
            typename std::enable_if<
                is_measurement<X>::value && !is_measurement<Y>::value &&
                    !has_valid_call_impl_vector<CovFunc, X, Y>::value &&
                    has_valid_call_impl<CovFunc, X, Y>::value,
                int>::type = 0>
  static Eigen::MatrixXd
  call_vector(const CovFunc &cov_func, const std::vector<X> &xs,
              const std::vector<Y> &ys, ThreadPool *pool = nullptr) {
    auto caller = [&](const auto &x, const auto &y) {
      return MeasurementForwarder::call(cov_func, x, y);
    };
    return compute_covariance_matrix(caller, xs, ys, pool);
  }

  // Right side Measurement: delegate to SubCaller if _call_impl_vector exists
  template <typename CovFunc, typename X, typename Y,
            typename std::enable_if<
                !is_measurement<X>::value && is_measurement<Y>::value &&
                    has_valid_call_impl_vector<CovFunc, X, Y>::value,
                int>::type = 0>
  static Eigen::MatrixXd
  call_vector(const CovFunc &cov_func, const std::vector<X> &xs,
              const std::vector<Y> &ys, ThreadPool *pool = nullptr) {
    return SubCaller::call_vector(cov_func, xs, ys, pool);
  }

  // Right side Measurement: unwrap right if no _call_impl_vector, no _call_impl
  template <typename CovFunc, typename X, typename Y,
            typename std::enable_if<
                !is_measurement<X>::value && is_measurement<Y>::value &&
                    !has_valid_call_impl_vector<CovFunc, X, Y>::value &&
                    !has_valid_call_impl<CovFunc, X, Y>::value,
                int>::type = 0>
  static Eigen::MatrixXd
  call_vector(const CovFunc &cov_func, const std::vector<X> &xs,
              const std::vector<Y> &ys, ThreadPool *pool = nullptr) {
    auto unwrapped_ys = unwrap_measurements(ys);
    return SubCaller::call_vector(cov_func, xs, unwrapped_ys, pool);
  }

  // Right side Measurement: pointwise if _call_impl but no _call_impl_vector
  template <typename CovFunc, typename X, typename Y,
            typename std::enable_if<
                !is_measurement<X>::value && is_measurement<Y>::value &&
                    !has_valid_call_impl_vector<CovFunc, X, Y>::value &&
                    has_valid_call_impl<CovFunc, X, Y>::value,
                int>::type = 0>
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

  // Symmetric Measurement: delegate to SubCaller if _call_impl_vector exists for Measurement directly
  template <typename CovFunc, typename X,
            typename std::enable_if<
                is_measurement<X>::value &&
                    has_valid_call_impl_vector_symmetric<CovFunc, X>::value,
                int>::type = 0>
  static Eigen::MatrixXd call_vector(const CovFunc &cov_func,
                                     const std::vector<X> &xs,
                                     ThreadPool *pool = nullptr) {
    return SubCaller::call_vector(cov_func, xs, pool);
  }

  // Symmetric Measurement: unwrap and delegate if single-arg OR two-arg exists for inner type (no Measurement-specific _call_impl_vector)
  template <typename CovFunc, typename X,
            typename std::enable_if<
                is_measurement<X>::value &&
                    !has_valid_call_impl_vector_symmetric<CovFunc, X>::value &&
                    !has_valid_call_impl<CovFunc, X, X>::value &&
                    (has_valid_call_impl_vector_single_arg<CovFunc,
                         measurement_inner_t<X>>::value ||
                     has_valid_call_impl_vector_symmetric<CovFunc,
                         measurement_inner_t<X>>::value),
                int>::type = 0>
  static Eigen::MatrixXd call_vector(const CovFunc &cov_func,
                                     const std::vector<X> &xs,
                                     ThreadPool *pool = nullptr) {
    auto unwrapped_xs = unwrap_measurements(xs);
    return SubCaller::call_vector(cov_func, unwrapped_xs, pool);
  }

  // Symmetric Measurement: unwrap if no single-arg, no _call_impl_vector, no _call_impl
  // but inner type has pointwise support
  template <typename CovFunc, typename X,
            typename std::enable_if<
                is_measurement<X>::value &&
                    !has_valid_call_impl_vector_symmetric<CovFunc, X>::value &&
                    !has_valid_call_impl<CovFunc, X, X>::value &&
                    !has_valid_call_impl_vector_single_arg<CovFunc,
                         measurement_inner_t<X>>::value &&
                    !has_valid_call_impl_vector_symmetric<CovFunc,
                         measurement_inner_t<X>>::value,
                int>::type = 0>
  static Eigen::MatrixXd call_vector(const CovFunc &cov_func,
                                     const std::vector<X> &xs,
                                     ThreadPool *pool = nullptr) {
    auto unwrapped_xs = unwrap_measurements(xs);
    return SubCaller::call_vector(cov_func, unwrapped_xs, pool);
  }

  // Symmetric Measurement: pointwise if _call_impl exists for Measurement
  template <typename CovFunc, typename X,
            typename std::enable_if<
                is_measurement<X>::value &&
                    !has_valid_call_impl_vector_symmetric<CovFunc, X>::value &&
                    has_valid_call_impl<CovFunc, X, X>::value,
                int>::type = 0>
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

  // Diagonal Measurement: unwrap ONLY if no explicit _call_impl for Measurement types
  template <typename CovFunc, typename X,
            typename std::enable_if<
                is_measurement<X>::value &&
                    !has_valid_call_impl<CovFunc, X, X>::value,
                int>::type = 0>
  static Eigen::VectorXd call_vector_diagonal(const CovFunc &cov_func,
                                              const std::vector<X> &xs,
                                              ThreadPool *pool = nullptr) {
    auto unwrapped_xs = unwrap_measurements(xs);
    return SubCaller::call_vector_diagonal(cov_func, unwrapped_xs, pool);
  }

  // Diagonal Measurement: pointwise if explicit _call_impl exists for Measurement types
  template <typename CovFunc, typename X,
            typename std::enable_if<
                is_measurement<X>::value &&
                    has_valid_call_impl<CovFunc, X, X>::value,
                int>::type = 0>
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
 * Variant batch dispatch helpers
 *
 * These helpers enable batch covariance dispatch for homogeneous variant
 * vectors by:
 * 1. Asserting all elements hold the same variant arm at runtime
 * 2. Extracting the concrete vector using variant::get<>()
 * 3. Dispatching to SubCaller::call_vector with the concrete types
 */

namespace variant_batch_detail {

/*
 * SortedVariantData - result of sorting a variant vector by type
 *
 * Contains:
 * - sorted: The variant vector with elements grouped by type
 * - to_sorted: Permutation matrix where sorted[i] = original[perm[i]]
 * - block_starts: block_starts[type_idx] = first index of that type in sorted
 *                 block_starts[sizeof...(Ts)] = total size (sentinel)
 */
template <typename... Ts> struct SortedVariantData {
  std::vector<variant<Ts...>> sorted;
  Eigen::PermutationMatrix<Eigen::Dynamic> to_sorted;
  std::array<Eigen::Index, sizeof...(Ts) + 1> block_starts;
};

/*
 * sort_variants_by_type - Sort a variant vector so elements are grouped by type
 *
 * This is a key primitive for heterogeneous variant batch dispatch.
 * Elements are reordered so all elements of type Ts[0] come first,
 * then all elements of type Ts[1], etc.
 *
 * Returns a SortedVariantData containing:
 * - The sorted vector
 * - A permutation matrix for recovering original order
 * - Block boundaries for each type
 */
template <typename... Ts>
SortedVariantData<Ts...>
sort_variants_by_type(const std::vector<variant<Ts...>> &variants) {
  SortedVariantData<Ts...> result;
  const std::size_t n = variants.size();

  // Count elements per type
  std::array<std::size_t, sizeof...(Ts)> counts{};
  for (const auto &v : variants) {
    counts[v.which()]++;
  }

  // Compute block start positions (prefix sum)
  result.block_starts[0] = 0;
  for (std::size_t t = 0; t < sizeof...(Ts); ++t) {
    result.block_starts[t + 1] =
        result.block_starts[t] + cast::to_index(counts[t]);
  }

  // Build permutation and sorted vector in single pass
  result.sorted.resize(n);
  result.to_sorted.resize(cast::to_index(n));
  std::array<std::size_t, sizeof...(Ts)> next_pos{};
  for (std::size_t t = 0; t < sizeof...(Ts); ++t) {
    next_pos[t] = cast::to_size(result.block_starts[t]);
  }

  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t type_idx = variants[i].which();
    const std::size_t sorted_idx = next_pos[type_idx]++;
    result.sorted[sorted_idx] = variants[i];
    // Eigen convention: indices[orig_idx] = sorted_idx means
    // (P * original)[sorted_idx] = original[orig_idx]
    // So P transforms original order to sorted order.
    result.to_sorted.indices()[cast::to_index(i)] = cast::to_index(sorted_idx);
  }

  return result;
}

/*
 * extract_block - Extract concrete-typed block from sorted variant vector
 *
 * Given a SortedVariantData and a type index, extracts all elements of that
 * type as a vector of the concrete type.
 */
template <typename T, typename... Ts>
std::vector<T> extract_block(const SortedVariantData<Ts...> &sorted_data,
                             std::size_t type_index) {
  const Eigen::Index start = sorted_data.block_starts[type_index];
  const Eigen::Index end = sorted_data.block_starts[type_index + 1];
  std::vector<T> result;
  result.reserve(cast::to_size(end - start));
  for (Eigen::Index i = start; i < end; ++i) {
    result.push_back(sorted_data.sorted[cast::to_size(i)].template get<T>());
  }
  return result;
}

// Helper to extract all values of a specific type from a homogeneous variant
// vector. This is a simplified version of extract_from_variants that doesn't
// depend on variant_utils.hpp's include order.
template <typename OutputType, typename... VariantTypes>
std::vector<OutputType>
extract_homogeneous(const std::vector<variant<VariantTypes...>> &xs) {
  std::vector<OutputType> output;
  output.reserve(xs.size());
  for (const auto &x : xs) {
    output.push_back(x.template get<OutputType>());
  }
  return output;
}

// Cross-covariance: variant on left, concrete on right
template <typename CovFunc, typename SubCaller, typename... Ts, typename Y,
          std::size_t... Is>
Eigen::MatrixXd dispatch_variant_batch_left_impl(
    const CovFunc &cov_func, const std::vector<variant<Ts...>> &xs,
    const std::vector<Y> &ys, ThreadPool *pool, std::size_t runtime_index,
    std::index_sequence<Is...>) {

  Eigen::MatrixXd result;
  bool found = false;

  // Fold expression: match runtime index to compile-time type
  ((Is == runtime_index
        ? (found = true,
           result = [&]() {
             using ConcreteType = std::tuple_element_t<Is, std::tuple<Ts...>>;
             auto concrete_xs = extract_homogeneous<ConcreteType>(xs);
             // Check if this type pair is supported; if not, return zeros
             // (matches pointwise visitor behavior)
             if constexpr (has_valid_cov_caller<CovFunc, SubCaller, ConcreteType,
                                                Y>::value) {
               return SubCaller::call_vector(cov_func, concrete_xs, ys, pool);
             } else {
               return Eigen::MatrixXd::Zero(cast::to_index(xs.size()),
                                            cast::to_index(ys.size()));
             }
           }(),
           0)
        : 0),
   ...);

  ALBATROSS_ASSERT(found && "Invalid variant index in batch dispatch");
  return result;
}

template <typename CovFunc, typename SubCaller, typename... Ts, typename Y>
Eigen::MatrixXd
dispatch_variant_batch_left(const CovFunc &cov_func,
                            const std::vector<variant<Ts...>> &xs,
                            const std::vector<Y> &ys, ThreadPool *pool) {
  if (xs.empty()) {
    return Eigen::MatrixXd(0, cast::to_index(ys.size()));
  }

  assert_homogeneous_variant_vector(xs);
  const std::size_t runtime_index = xs[0].which();

  return dispatch_variant_batch_left_impl<CovFunc, SubCaller>(
      cov_func, xs, ys, pool, runtime_index, std::index_sequence_for<Ts...>{});
}

// Cross-covariance: concrete on left, variant on right
template <typename CovFunc, typename SubCaller, typename X, typename... Ts,
          std::size_t... Is>
Eigen::MatrixXd dispatch_variant_batch_right_impl(
    const CovFunc &cov_func, const std::vector<X> &xs,
    const std::vector<variant<Ts...>> &ys, ThreadPool *pool,
    std::size_t runtime_index, std::index_sequence<Is...>) {

  Eigen::MatrixXd result;
  bool found = false;

  ((Is == runtime_index
        ? (found = true,
           result = [&]() {
             using ConcreteType = std::tuple_element_t<Is, std::tuple<Ts...>>;
             auto concrete_ys = extract_homogeneous<ConcreteType>(ys);
             // Check if this type pair is supported; if not, return zeros
             // (matches pointwise visitor behavior)
             if constexpr (has_valid_cov_caller<CovFunc, SubCaller, X,
                                                ConcreteType>::value) {
               return SubCaller::call_vector(cov_func, xs, concrete_ys, pool);
             } else {
               return Eigen::MatrixXd::Zero(cast::to_index(xs.size()),
                                            cast::to_index(ys.size()));
             }
           }(),
           0)
        : 0),
   ...);

  ALBATROSS_ASSERT(found && "Invalid variant index in batch dispatch");
  return result;
}

template <typename CovFunc, typename SubCaller, typename X, typename... Ts>
Eigen::MatrixXd
dispatch_variant_batch_right(const CovFunc &cov_func, const std::vector<X> &xs,
                             const std::vector<variant<Ts...>> &ys,
                             ThreadPool *pool) {
  if (ys.empty()) {
    return Eigen::MatrixXd(cast::to_index(xs.size()), 0);
  }

  assert_homogeneous_variant_vector(ys);
  const std::size_t runtime_index = ys[0].which();

  return dispatch_variant_batch_right_impl<CovFunc, SubCaller>(
      cov_func, xs, ys, pool, runtime_index, std::index_sequence_for<Ts...>{});
}

// Cross-covariance: both sides are variants
// Inner dispatch over Y types, given a concrete X type
template <typename CovFunc, typename SubCaller, typename XType, typename... Ys,
          std::size_t... YIs>
Eigen::MatrixXd dispatch_variant_batch_both_y_impl(
    const CovFunc &cov_func, const std::vector<XType> &concrete_xs,
    const std::vector<variant<Ys...>> &ys, ThreadPool *pool,
    std::size_t y_index, std::index_sequence<YIs...>) {

  Eigen::MatrixXd result;
  bool found = false;

  ((YIs == y_index
        ? (found = true,
           result = [&]() {
             using YType = std::tuple_element_t<YIs, std::tuple<Ys...>>;
             auto concrete_ys = extract_homogeneous<YType>(ys);
             // Check if this type pair is supported; if not, return zeros
             if constexpr (has_valid_cov_caller<CovFunc, SubCaller, XType,
                                                YType>::value) {
               return SubCaller::call_vector(cov_func, concrete_xs, concrete_ys,
                                             pool);
             } else {
               return Eigen::MatrixXd::Zero(cast::to_index(concrete_xs.size()),
                                            cast::to_index(ys.size()));
             }
           }(),
           0)
        : 0),
   ...);

  ALBATROSS_ASSERT(found && "Invalid Y variant index in batch dispatch");
  return result;
}

// Outer dispatch over X types
template <typename CovFunc, typename SubCaller, typename... Xs, typename... Ys,
          std::size_t... XIs>
Eigen::MatrixXd dispatch_variant_batch_both_x_impl(
    const CovFunc &cov_func, const std::vector<variant<Xs...>> &xs,
    const std::vector<variant<Ys...>> &ys, ThreadPool *pool,
    std::size_t x_index, std::size_t y_index, std::index_sequence<XIs...>) {

  Eigen::MatrixXd result;
  bool found = false;

  ((XIs == x_index
        ? (found = true,
           result = [&]() {
             using XType = std::tuple_element_t<XIs, std::tuple<Xs...>>;
             auto concrete_xs = extract_homogeneous<XType>(xs);
             return dispatch_variant_batch_both_y_impl<CovFunc, SubCaller,
                                                       XType>(
                 cov_func, concrete_xs, ys, pool, y_index,
                 std::index_sequence_for<Ys...>{});
           }(),
           0)
        : 0),
   ...);

  ALBATROSS_ASSERT(found && "Invalid X variant index in batch dispatch");
  return result;
}

template <typename CovFunc, typename SubCaller, typename... Xs, typename... Ys>
Eigen::MatrixXd
dispatch_variant_batch_both(const CovFunc &cov_func,
                            const std::vector<variant<Xs...>> &xs,
                            const std::vector<variant<Ys...>> &ys,
                            ThreadPool *pool) {
  if (xs.empty()) {
    return Eigen::MatrixXd(0, cast::to_index(ys.size()));
  }
  if (ys.empty()) {
    return Eigen::MatrixXd(cast::to_index(xs.size()), 0);
  }

  assert_homogeneous_variant_vector(xs);
  assert_homogeneous_variant_vector(ys);
  const std::size_t x_index = xs[0].which();
  const std::size_t y_index = ys[0].which();

  return dispatch_variant_batch_both_x_impl<CovFunc, SubCaller>(
      cov_func, xs, ys, pool, x_index, y_index,
      std::index_sequence_for<Xs...>{});
}

// Symmetric case: single variant vector
template <typename CovFunc, typename SubCaller, typename... Ts,
          std::size_t... Is>
Eigen::MatrixXd dispatch_variant_batch_symmetric_impl(
    const CovFunc &cov_func, const std::vector<variant<Ts...>> &xs,
    ThreadPool *pool, std::size_t runtime_index, std::index_sequence<Is...>) {

  Eigen::MatrixXd result;
  bool found = false;

  ((Is == runtime_index
        ? (found = true,
           result = [&]() {
             using ConcreteType = std::tuple_element_t<Is, std::tuple<Ts...>>;
             auto concrete_xs = extract_homogeneous<ConcreteType>(xs);
             // Check if this type is supported; if not, return zeros
             // (matches pointwise visitor behavior)
             if constexpr (has_valid_cov_caller<CovFunc, SubCaller, ConcreteType,
                                                ConcreteType>::value) {
               return SubCaller::call_vector(cov_func, concrete_xs, pool);
             } else {
               return Eigen::MatrixXd::Zero(cast::to_index(xs.size()),
                                            cast::to_index(xs.size()));
             }
           }(),
           0)
        : 0),
   ...);

  ALBATROSS_ASSERT(found && "Invalid variant index in batch dispatch");
  return result;
}

template <typename CovFunc, typename SubCaller, typename... Ts>
Eigen::MatrixXd
dispatch_variant_batch_symmetric(const CovFunc &cov_func,
                                 const std::vector<variant<Ts...>> &xs,
                                 ThreadPool *pool) {
  if (xs.empty()) {
    return Eigen::MatrixXd(0, 0);
  }

  assert_homogeneous_variant_vector(xs);
  const std::size_t runtime_index = xs[0].which();

  return dispatch_variant_batch_symmetric_impl<CovFunc, SubCaller>(
      cov_func, xs, pool, runtime_index, std::index_sequence_for<Ts...>{});
}

// Diagonal case: single variant vector
template <typename CovFunc, typename SubCaller, typename... Ts,
          std::size_t... Is>
Eigen::VectorXd dispatch_variant_batch_diagonal_impl(
    const CovFunc &cov_func, const std::vector<variant<Ts...>> &xs,
    ThreadPool *pool, std::size_t runtime_index, std::index_sequence<Is...>) {

  Eigen::VectorXd result;
  bool found = false;

  ((Is == runtime_index
        ? (found = true,
           result = [&]() {
             using ConcreteType = std::tuple_element_t<Is, std::tuple<Ts...>>;
             auto concrete_xs = extract_homogeneous<ConcreteType>(xs);
             // Check if this type is supported; if not, return zeros
             // (matches pointwise visitor behavior)
             if constexpr (has_valid_cov_caller<CovFunc, SubCaller, ConcreteType,
                                                ConcreteType>::value) {
               return SubCaller::call_vector_diagonal(cov_func, concrete_xs,
                                                      pool);
             } else {
               return Eigen::VectorXd::Zero(cast::to_index(xs.size()));
             }
           }(),
           0)
        : 0),
   ...);

  ALBATROSS_ASSERT(found && "Invalid variant index in batch dispatch");
  return result;
}

template <typename CovFunc, typename SubCaller, typename... Ts>
Eigen::VectorXd
dispatch_variant_batch_diagonal(const CovFunc &cov_func,
                                const std::vector<variant<Ts...>> &xs,
                                ThreadPool *pool) {
  if (xs.empty()) {
    return Eigen::VectorXd(0);
  }

  assert_homogeneous_variant_vector(xs);
  const std::size_t runtime_index = xs[0].which();

  return dispatch_variant_batch_diagonal_impl<CovFunc, SubCaller>(
      cov_func, xs, pool, runtime_index, std::index_sequence_for<Ts...>{});
}

/*
 * ============================================================================
 * Heterogeneous Variant Batch Dispatch
 *
 * These functions handle batch covariance computation for variant vectors
 * containing a mix of different types (heterogeneous). The approach is:
 *
 * 1. Sort inputs by type, creating permutation vectors
 * 2. Compute covariance blocks for each type pair
 * 3. Unpermute the result using Eigen::PermutationMatrix
 *
 * This avoids O(n*m) visitor calls by computing contiguous blocks and using
 * efficient matrix permutation operations.
 * ============================================================================
 */

// Type wrapper to pass variant types as a single template parameter
template <typename... Ts> struct TypeList {};

/*
 * compute_sorted_covariance_cross - Compute cross-covariance in sorted order
 *
 * Given sorted variant data for xs and ys, computes the covariance matrix
 * where blocks of same-type elements are contiguous. The result matrix is
 * in "sorted order" and needs to be unpermuted to get the original order.
 */

// Helper struct to process one row block (X type XI) against all column blocks
template <typename CovFunc, typename SubCaller, std::size_t XI,
          typename TypeListT, std::size_t... YIs>
struct CrossRowBlockProcessor;

template <typename CovFunc, typename SubCaller, std::size_t XI, typename... Ts,
          std::size_t... YIs>
struct CrossRowBlockProcessor<CovFunc, SubCaller, XI, TypeList<Ts...>, YIs...> {
  static void process(const CovFunc &cov_func,
                      const SortedVariantData<Ts...> &x_sorted,
                      const SortedVariantData<Ts...> &y_sorted,
                      Eigen::MatrixXd &result, ThreadPool *pool) {
    using XType = std::tuple_element_t<XI, std::tuple<Ts...>>;
    const Eigen::Index x_start = x_sorted.block_starts[XI];
    const Eigen::Index x_end = x_sorted.block_starts[XI + 1];
    if (x_start == x_end)
      return; // Empty block

    auto x_block = extract_block<XType>(x_sorted, XI);

    // For each Y type (inner fold with fixed XI)
    (
        [&]() {
          using YType = std::tuple_element_t<YIs, std::tuple<Ts...>>;
          const Eigen::Index y_start = y_sorted.block_starts[YIs];
          const Eigen::Index y_end = y_sorted.block_starts[YIs + 1];
          if (y_start == y_end)
            return; // Empty block

          if constexpr (has_valid_cov_caller<CovFunc, SubCaller, XType,
                                             YType>::value) {
            // Use SubCaller to go through full caller chain
            auto y_block = extract_block<YType>(y_sorted, YIs);
            auto block_result =
                SubCaller::call_vector(cov_func, x_block, y_block, pool);
            // Place block into contiguous region of result
            result.block(x_start, y_start, x_end - x_start, y_end - y_start) =
                block_result;
          }
          // Unsupported pairs: leave as zero
        }(),
        ...);
  }
};

// Helper to call the processor with an index sequence
template <typename CovFunc, typename SubCaller, std::size_t XI, typename... Ts,
          std::size_t... YIs>
void compute_cross_row_block(const CovFunc &cov_func,
                             const SortedVariantData<Ts...> &x_sorted,
                             const SortedVariantData<Ts...> &y_sorted,
                             Eigen::MatrixXd &result, ThreadPool *pool,
                             std::index_sequence<YIs...>) {
  CrossRowBlockProcessor<CovFunc, SubCaller, XI, TypeList<Ts...>,
                         YIs...>::process(cov_func, x_sorted, y_sorted, result,
                                          pool);
}

template <typename CovFunc, typename SubCaller, typename... Ts,
          std::size_t... XIs>
Eigen::MatrixXd compute_sorted_covariance_cross_impl(
    const CovFunc &cov_func, const SortedVariantData<Ts...> &x_sorted,
    const SortedVariantData<Ts...> &y_sorted, ThreadPool *pool,
    std::index_sequence<XIs...>) {

  const Eigen::Index m = cast::to_index(x_sorted.sorted.size());
  const Eigen::Index n = cast::to_index(y_sorted.sorted.size());
  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(m, n);

  // For each X type, process all Y types
  (compute_cross_row_block<CovFunc, SubCaller, XIs>(
       cov_func, x_sorted, y_sorted, result, pool,
       std::index_sequence_for<Ts...>{}),
   ...);

  return result;
}

template <typename CovFunc, typename SubCaller, typename... Ts>
Eigen::MatrixXd
compute_sorted_covariance_cross(const CovFunc &cov_func,
                                const SortedVariantData<Ts...> &x_sorted,
                                const SortedVariantData<Ts...> &y_sorted,
                                ThreadPool *pool) {
  return compute_sorted_covariance_cross_impl<CovFunc, SubCaller>(
      cov_func, x_sorted, y_sorted, pool, std::index_sequence_for<Ts...>{});
}

/*
 * dispatch_heterogeneous_variant_batch_cross - Main entry point for
 * heterogeneous cross-covariance
 *
 * Computes cov(xs, ys) where xs and ys are variant vectors that may contain
 * a mix of different types.
 */
template <typename CovFunc, typename SubCaller, typename... Ts>
Eigen::MatrixXd dispatch_heterogeneous_variant_batch_cross(
    const CovFunc &cov_func, const std::vector<variant<Ts...>> &xs,
    const std::vector<variant<Ts...>> &ys, ThreadPool *pool) {

  if (xs.empty()) {
    return Eigen::MatrixXd(0, cast::to_index(ys.size()));
  }
  if (ys.empty()) {
    return Eigen::MatrixXd(cast::to_index(xs.size()), 0);
  }

  // 1. Sort both input vectors by type (single copy of each element)
  auto x_sorted = sort_variants_by_type(xs);
  auto y_sorted = sort_variants_by_type(ys);

  // 2. Compute covariance in sorted order (contiguous block operations)
  Eigen::MatrixXd sorted_result =
      compute_sorted_covariance_cross<CovFunc, SubCaller>(cov_func, x_sorted,
                                                          y_sorted, pool);

  // 3. Unpermute rows and columns to original order
  // result(i, j) = sorted_result(to_sorted[i], to_sorted[j])
  // Equivalently: result = P_x^{-1} * sorted_result * P_y^{-T}
  // Note: We need to evaluate the inverse before calling transpose()
  Eigen::PermutationMatrix<Eigen::Dynamic> x_inv = x_sorted.to_sorted.inverse();
  Eigen::PermutationMatrix<Eigen::Dynamic> y_inv = y_sorted.to_sorted.inverse();
  return x_inv * sorted_result * y_inv.transpose();
}

/*
 * compute_sorted_covariance_symmetric - Compute symmetric covariance in sorted
 * order
 *
 * Optimized for the case where xs == ys. Computes diagonal blocks as symmetric
 * and off-diagonal blocks once.
 */

// Helper: process one row block (X type XI) against column blocks (Y types >= XI)
template <typename CovFunc, typename SubCaller, std::size_t XI,
          typename TypeListT, std::size_t... YIs>
struct SymmetricRowBlockProcessor;

template <typename CovFunc, typename SubCaller, std::size_t XI, typename... Ts,
          std::size_t... YIs>
struct SymmetricRowBlockProcessor<CovFunc, SubCaller, XI, TypeList<Ts...>,
                                  YIs...> {
  static void process(const CovFunc &cov_func,
                      const SortedVariantData<Ts...> &sorted,
                      Eigen::MatrixXd &result, ThreadPool *pool) {
    using XType = std::tuple_element_t<XI, std::tuple<Ts...>>;
    const Eigen::Index x_start = sorted.block_starts[XI];
    const Eigen::Index x_end = sorted.block_starts[XI + 1];
    if (x_start == x_end)
      return; // Empty block

    auto x_block = extract_block<XType>(sorted, XI);

    // For each Y type (inner fold with fixed XI)
    (
        [&]() {
          using YType = std::tuple_element_t<YIs, std::tuple<Ts...>>;
          const Eigen::Index y_start = sorted.block_starts[YIs];
          const Eigen::Index y_end = sorted.block_starts[YIs + 1];
          if (y_start == y_end)
            return; // Empty block

          // Only compute upper triangle (including diagonal blocks)
          if constexpr (YIs < XI)
            return;

          if constexpr (has_valid_cov_caller<CovFunc, SubCaller, XType,
                                             YType>::value) {
            // Use SubCaller to go through full caller chain
            auto y_block = extract_block<YType>(sorted, YIs);
            Eigen::MatrixXd block_result;

            if constexpr (XI == YIs) {
              // Diagonal block: compute symmetric
              block_result = SubCaller::call_vector(cov_func, x_block, pool);
            } else {
              // Off-diagonal block
              block_result =
                  SubCaller::call_vector(cov_func, x_block, y_block, pool);
            }

            // Place in upper triangle
            result.block(x_start, y_start, x_end - x_start, y_end - y_start) =
                block_result;

            // Mirror to lower triangle for off-diagonal blocks
            if constexpr (XI != YIs) {
              result.block(y_start, x_start, y_end - y_start, x_end - x_start) =
                  block_result.transpose();
            }
          }
          // Unsupported pairs: leave as zero
        }(),
        ...);
  }
};

// Helper to call the processor with an index sequence
template <typename CovFunc, typename SubCaller, std::size_t XI, typename... Ts,
          std::size_t... YIs>
void compute_symmetric_row_block(const CovFunc &cov_func,
                                 const SortedVariantData<Ts...> &sorted,
                                 Eigen::MatrixXd &result, ThreadPool *pool,
                                 std::index_sequence<YIs...>) {
  SymmetricRowBlockProcessor<CovFunc, SubCaller, XI, TypeList<Ts...>,
                             YIs...>::process(cov_func, sorted, result, pool);
}

template <typename CovFunc, typename SubCaller, typename... Ts,
          std::size_t... XIs>
Eigen::MatrixXd compute_sorted_covariance_symmetric_impl(
    const CovFunc &cov_func, const SortedVariantData<Ts...> &sorted,
    ThreadPool *pool, std::index_sequence<XIs...>) {

  const Eigen::Index n = cast::to_index(sorted.sorted.size());
  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(n, n);

  // For each X type, process Y types >= X
  (compute_symmetric_row_block<CovFunc, SubCaller, XIs>(
       cov_func, sorted, result, pool, std::index_sequence_for<Ts...>{}),
   ...);

  return result;
}

template <typename CovFunc, typename SubCaller, typename... Ts>
Eigen::MatrixXd
compute_sorted_covariance_symmetric(const CovFunc &cov_func,
                                    const SortedVariantData<Ts...> &sorted,
                                    ThreadPool *pool) {
  return compute_sorted_covariance_symmetric_impl<CovFunc, SubCaller>(
      cov_func, sorted, pool, std::index_sequence_for<Ts...>{});
}

/*
 * dispatch_heterogeneous_variant_batch_symmetric - Main entry point for
 * heterogeneous symmetric covariance
 *
 * Computes cov(xs, xs) where xs is a variant vector that may contain
 * a mix of different types.
 */
template <typename CovFunc, typename SubCaller, typename... Ts>
Eigen::MatrixXd dispatch_heterogeneous_variant_batch_symmetric(
    const CovFunc &cov_func, const std::vector<variant<Ts...>> &xs,
    ThreadPool *pool) {

  if (xs.empty()) {
    return Eigen::MatrixXd(0, 0);
  }

  // 1. Sort input by type (single copy of each element)
  auto sorted = sort_variants_by_type(xs);

  // 2. Compute symmetric covariance in sorted order
  Eigen::MatrixXd sorted_result =
      compute_sorted_covariance_symmetric<CovFunc, SubCaller>(cov_func, sorted,
                                                              pool);

  // 3. Unpermute: single permutation matrix since xs == ys
  // result = P^{-1} * sorted_result * P^{-T}
  // Note: We need to evaluate the inverse before calling transpose()
  Eigen::PermutationMatrix<Eigen::Dynamic> P_inv = sorted.to_sorted.inverse();
  return P_inv * sorted_result * P_inv.transpose();
}

/*
 * dispatch_heterogeneous_variant_batch_diagonal - Main entry point for
 * heterogeneous diagonal extraction
 *
 * Computes diag(cov(xs, xs)) where xs is a variant vector that may contain
 * a mix of different types.
 */
template <typename CovFunc, typename SubCaller, typename... Ts,
          std::size_t... Is>
Eigen::VectorXd dispatch_heterogeneous_variant_batch_diagonal_impl(
    const CovFunc &cov_func, const SortedVariantData<Ts...> &sorted,
    ThreadPool *pool, std::index_sequence<Is...>) {

  const Eigen::Index n = cast::to_index(sorted.sorted.size());
  Eigen::VectorXd sorted_diag = Eigen::VectorXd::Zero(n);

  // For each type
  (
      [&]() {
        using XType = std::tuple_element_t<Is, std::tuple<Ts...>>;
        const Eigen::Index start = sorted.block_starts[Is];
        const Eigen::Index end = sorted.block_starts[Is + 1];
        if (start == end)
          return; // Empty block

        auto block = extract_block<XType>(sorted, Is);

        // Use SubCaller to go through full caller chain for diagonal
        if constexpr (has_valid_cov_caller<CovFunc, SubCaller, XType,
                                           XType>::value) {
          auto block_diag = SubCaller::call_vector_diagonal(cov_func, block, pool);
          sorted_diag.segment(start, end - start) = block_diag;
        }
        // Unsupported types: leave as zero
      }(),
      ...);

  return sorted_diag;
}

template <typename CovFunc, typename SubCaller, typename... Ts>
Eigen::VectorXd dispatch_heterogeneous_variant_batch_diagonal(
    const CovFunc &cov_func, const std::vector<variant<Ts...>> &xs,
    ThreadPool *pool) {

  if (xs.empty()) {
    return Eigen::VectorXd(0);
  }

  // 1. Sort input by type
  auto sorted = sort_variants_by_type(xs);

  // 2. Compute diagonal in sorted order
  Eigen::VectorXd sorted_diag =
      dispatch_heterogeneous_variant_batch_diagonal_impl<CovFunc, SubCaller>(
          cov_func, sorted, pool, std::index_sequence_for<Ts...>{});

  // 3. Unpermute: apply inverse permutation to get original order
  // With Eigen convention: indices[orig_idx] = sorted_idx
  // We want: result[orig_idx] = sorted_diag[sorted_idx]
  const Eigen::Index n = cast::to_index(xs.size());
  Eigen::VectorXd result(n);
  for (Eigen::Index orig_idx = 0; orig_idx < n; ++orig_idx) {
    const Eigen::Index sorted_idx = sorted.to_sorted.indices()[orig_idx];
    result[orig_idx] = sorted_diag[sorted_idx];
  }

  return result;
}

} // namespace variant_batch_detail

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

  // Variant on left, concrete on right: extract and dispatch if homogeneous,
  // otherwise fall back to pointwise
  template <typename CovFunc, typename... Ts, typename Y,
            typename std::enable_if<!is_variant<Y>::value, int>::type = 0>
  static Eigen::MatrixXd
  call_vector(const CovFunc &cov_func, const std::vector<variant<Ts...>> &xs,
              const std::vector<Y> &ys, ThreadPool *pool = nullptr) {
    if (is_homogeneous_variant_vector(xs)) {
      return variant_batch_detail::dispatch_variant_batch_left<CovFunc,
                                                               SubCaller>(
          cov_func, xs, ys, pool);
    }
    // Heterogeneous: fall back to pointwise
    auto caller = [&](const auto &x, const auto &y) {
      return VariantForwarder::call(cov_func, x, y);
    };
    return compute_covariance_matrix(caller, xs, ys, pool);
  }

  // Concrete on left, variant on right: extract and dispatch if homogeneous,
  // otherwise fall back to pointwise
  template <typename CovFunc, typename X, typename... Ts,
            typename std::enable_if<!is_variant<X>::value, int>::type = 0>
  static Eigen::MatrixXd
  call_vector(const CovFunc &cov_func, const std::vector<X> &xs,
              const std::vector<variant<Ts...>> &ys, ThreadPool *pool = nullptr) {
    if (is_homogeneous_variant_vector(ys)) {
      return variant_batch_detail::dispatch_variant_batch_right<CovFunc,
                                                                SubCaller>(
          cov_func, xs, ys, pool);
    }
    // Heterogeneous: fall back to pointwise
    auto caller = [&](const auto &x, const auto &y) {
      return VariantForwarder::call(cov_func, x, y);
    };
    return compute_covariance_matrix(caller, xs, ys, pool);
  }

  // Both sides variant (same variant type): use heterogeneous batch dispatch
  // This handles both homogeneous (single type) and heterogeneous (mixed types)
  // cases efficiently using sort-compute-unpermute.
  template <typename CovFunc, typename... Ts>
  static Eigen::MatrixXd
  call_vector(const CovFunc &cov_func, const std::vector<variant<Ts...>> &xs,
              const std::vector<variant<Ts...>> &ys, ThreadPool *pool = nullptr) {
    // Use heterogeneous dispatch for all cases - it handles homogeneous
    // efficiently (single non-empty block) and heterogeneous correctly
    return variant_batch_detail::dispatch_heterogeneous_variant_batch_cross<
        CovFunc, SubCaller>(cov_func, xs, ys, pool);
  }

  // Both sides variant but DIFFERENT variant types: fall back to pointwise
  // This handles cases like variant<A,B,C> vs variant<B,C> which can happen
  // in cross-covariance scenarios.
  template <typename CovFunc, typename... XTs, typename... YTs,
            typename std::enable_if<
                !std::is_same<variant<XTs...>, variant<YTs...>>::value,
                int>::type = 0>
  static Eigen::MatrixXd
  call_vector(const CovFunc &cov_func, const std::vector<variant<XTs...>> &xs,
              const std::vector<variant<YTs...>> &ys, ThreadPool *pool = nullptr) {
    // Fall back to pointwise computation for mismatched variant types
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

  // Symmetric: use heterogeneous batch dispatch for all variant cases
  // This handles both homogeneous and heterogeneous efficiently
  template <typename CovFunc, typename... Ts>
  static Eigen::MatrixXd call_vector(const CovFunc &cov_func,
                                     const std::vector<variant<Ts...>> &xs,
                                     ThreadPool *pool = nullptr) {
    // Use heterogeneous dispatch for all cases
    return variant_batch_detail::dispatch_heterogeneous_variant_batch_symmetric<
        CovFunc, SubCaller>(cov_func, xs, pool);
  }

  // Diagonal: passthrough for non-variants
  template <typename CovFunc, typename X,
            typename std::enable_if<!is_variant<X>::value, int>::type = 0>
  static Eigen::VectorXd call_vector_diagonal(const CovFunc &cov_func,
                                              const std::vector<X> &xs,
                                              ThreadPool *pool = nullptr) {
    return SubCaller::call_vector_diagonal(cov_func, xs, pool);
  }

  // Diagonal: use heterogeneous batch dispatch for all variant cases
  template <typename CovFunc, typename... Ts>
  static Eigen::VectorXd
  call_vector_diagonal(const CovFunc &cov_func,
                       const std::vector<variant<Ts...>> &xs,
                       ThreadPool *pool = nullptr) {
    // Use heterogeneous dispatch for all cases
    return variant_batch_detail::dispatch_heterogeneous_variant_batch_diagonal<
        CovFunc, SubCaller>(cov_func, xs, pool);
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

  // Symmetric batch Priority 1: single-vector method exists - use it
  template <typename CovFunc, typename X,
            typename std::enable_if<
                has_valid_call_impl_vector_single_arg<CovFunc, X>::value,
                int>::type = 0>
  static Eigen::MatrixXd call_vector(const CovFunc &cov_func,
                                     const std::vector<X> &xs,
                                     ThreadPool *pool = nullptr) {
    return cov_func._call_impl_vector(xs, pool);  // Single-vector call
  }

  // Symmetric batch Priority 2: two-vector symmetric method exists (no single-vector)
  template <typename CovFunc, typename X,
            typename std::enable_if<
                !has_valid_call_impl_vector_single_arg<CovFunc, X>::value &&
                    has_valid_call_impl_vector_symmetric<CovFunc, X>::value,
                int>::type = 0>
  static Eigen::MatrixXd call_vector(const CovFunc &cov_func,
                                     const std::vector<X> &xs,
                                     ThreadPool *pool = nullptr) {
    return cov_func._call_impl_vector(xs, xs, pool);  // Two-vector call
  }

  // Symmetric batch Priority 3: pointwise fallback (no batch methods)
  template <typename CovFunc, typename X,
            typename std::enable_if<
                !has_valid_call_impl_vector_single_arg<CovFunc, X>::value &&
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

  // Synthesize scalar from single-arg batch (for single-arg-only covariances)
  // This handles covariances that define _call_impl_vector(xs, pool) but NOT
  // _call_impl_vector(xs, ys, pool) or _call_impl.
  template <typename CovFunc, typename X, typename Y,
            typename std::enable_if<
                std::is_same<X, Y>::value &&
                    has_valid_call_impl_vector_single_arg<CovFunc, X>::value &&
                    !has_valid_call_impl_vector<CovFunc, X, X>::value &&
                    !has_valid_cov_caller<CovFunc, SubCaller, X, X>::value,
                int>::type = 0>
  static double call(const CovFunc &cov_func, const X &x, const Y &y) {
    // For single-arg-only covariances, create a 2-element vector and extract
    // the cross element (0,1) for the scalar result
    const std::vector<X> both{x, y};
    return cov_func._call_impl_vector(both, nullptr)(0, 1);
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

  // Diagonal for two-arg batch-only covariances - synthesize from _call_impl_vector(xs, ys, pool)
  template <typename CovFunc, typename X,
            typename std::enable_if<
                !has_valid_call_impl_vector_diagonal<CovFunc, X>::value &&
                    !has_valid_cov_caller<CovFunc, SubCaller, X, X>::value &&
                    !has_valid_call_impl_vector_single_arg<CovFunc, X>::value &&
                    has_valid_call_impl_vector<CovFunc, X, X>::value,
                int>::type = 0>
  static Eigen::VectorXd call_vector_diagonal(const CovFunc &cov_func,
                                              const std::vector<X> &xs,
                                              ThreadPool *pool = nullptr) {
    // Extract diagonal from full matrix
    // This is less efficient but correct for batch-only covariances
    return cov_func._call_impl_vector(xs, xs, pool).diagonal();
  }

  // Diagonal for single-arg-only covariances - synthesize from _call_impl_vector(xs, pool)
  template <typename CovFunc, typename X,
            typename std::enable_if<
                !has_valid_call_impl_vector_diagonal<CovFunc, X>::value &&
                    !has_valid_cov_caller<CovFunc, SubCaller, X, X>::value &&
                    has_valid_call_impl_vector_single_arg<CovFunc, X>::value &&
                    !has_valid_call_impl_vector<CovFunc, X, X>::value,
                int>::type = 0>
  static Eigen::VectorXd call_vector_diagonal(const CovFunc &cov_func,
                                              const std::vector<X> &xs,
                                              ThreadPool *pool = nullptr) {
    // Extract diagonal from single-arg batch result
    return cov_func._call_impl_vector(xs, pool).diagonal();
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

// Symmetric version: check if pointwise OR symmetric batch (single-arg or two-arg) is available
template <typename CovFunc, typename X>
class has_valid_caller_or_batch_symmetric {
public:
  static constexpr bool value =
      has_valid_caller<CovFunc, X, X>::value ||
      has_valid_call_impl_vector_single_arg<CovFunc, X>::value ||
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
