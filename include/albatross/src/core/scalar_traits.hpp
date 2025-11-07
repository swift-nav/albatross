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

#ifndef ALBATROSS_CORE_SCALAR_TRAITS_H
#define ALBATROSS_CORE_SCALAR_TRAITS_H

namespace albatross {

// Build-time precision configuration
//
// IMPORTANT: Some operations MUST always use double precision for numerical stability:
// - Matrix decompositions (LDLT, Cholesky, QR)
// - Linear system solves
// - Matrix inverses
// - Log determinants
// - Variance/covariance storage
// - Hyperparameter values
//
// Only use BuildTimeScalar for:
// - Covariance function evaluation (e.g., exp(-d²/l²))
// - Distance computations
// - Intermediate matrix products (if numerically stable)
//
#ifdef ALBATROSS_USE_FLOAT_PRECISION
using BuildTimeScalar = float;
#else
using BuildTimeScalar = double;
#endif

/*
 * Scalar type traits for mixed-precision support.
 *
 * This system allows different scalar types to be used for:
 * - Compute: Fast operations (covariance evaluation, matrix multiplication)
 * - Storage: High-precision storage (decompositions, variance computations)
 * - Parameter: Hyperparameter values (always high precision)
 *
 * Three predefined policies:
 * - DoublePrecision: Current behavior (double everywhere) - DEFAULT
 * - FloatPrecision: Pure float (fastest, lower precision)
 * - MixedPrecision: Float compute + double storage (optimal balance)
 */

template <typename ComputeScalar = double, typename StorageScalar = double>
struct ScalarTraits {
  using Compute = ComputeScalar;  // Fast operations (covariance, matrix ops)
  using Storage = StorageScalar;  // High-precision storage (LDLT, variances)
  using Parameter = StorageScalar; // Hyperparameters always high precision

  // Type aliases for Eigen matrices/vectors
  template <int Rows = Eigen::Dynamic, int Cols = Eigen::Dynamic>
  using ComputeMatrix = Eigen::Matrix<Compute, Rows, Cols>;

  template <int Rows = Eigen::Dynamic>
  using ComputeVector = Eigen::Matrix<Compute, Rows, 1>;

  template <int Rows = Eigen::Dynamic, int Cols = Eigen::Dynamic>
  using StorageMatrix = Eigen::Matrix<Storage, Rows, Cols>;

  template <int Rows = Eigen::Dynamic>
  using StorageVector = Eigen::Matrix<Storage, Rows, 1>;

  using ComputeDiagonalMatrix = Eigen::DiagonalMatrix<Compute, Eigen::Dynamic>;
  using StorageDiagonalMatrix = Eigen::DiagonalMatrix<Storage, Eigen::Dynamic>;

  // Precision conversion utilities
  template <typename Derived>
  static auto to_storage(const Eigen::MatrixBase<Derived>& mat) {
    using Scalar = typename Derived::Scalar;
    if constexpr (std::is_same<Scalar, Storage>::value) {
      return mat.eval();
    } else {
      return mat.template cast<Storage>().eval();
    }
  }

  template <typename Derived>
  static auto to_compute(const Eigen::MatrixBase<Derived>& mat) {
    using Scalar = typename Derived::Scalar;
    if constexpr (std::is_same<Scalar, Compute>::value) {
      return mat.eval();
    } else {
      return mat.template cast<Compute>().eval();
    }
  }
};

// Predefined scalar policies
using DoublePrecision = ScalarTraits<double, double>;  // Pure double precision
using FloatPrecision = ScalarTraits<float, float>;    // Pure float precision (fastest)
using MixedPrecision = ScalarTraits<float, double>;   // Float compute, double storage (optimal)

// Build-time default: uses BuildTimeScalar for compute, double for storage
// When ALBATROSS_USE_FLOAT_PRECISION is defined, this enables mixed-precision mode
using DefaultPrecision = ScalarTraits<BuildTimeScalar, double>;

// Legacy: Default scalar type (for backward compatibility)
using DefaultScalar = double;

// Type trait to check if a type is a ScalarTraits instantiation
template <typename T>
struct is_scalar_traits : std::false_type {};

template <typename C, typename S>
struct is_scalar_traits<ScalarTraits<C, S>> : std::true_type {};

template <typename T>
inline constexpr bool is_scalar_traits_v = is_scalar_traits<T>::value;

/*
 * Precision Conversion Utilities
 *
 * These utilities help convert between float and double precision
 * at key boundaries in mixed-precision workflows.
 */

// Convert Eigen matrix/vector to different precision
template <typename ToScalar, typename Derived>
inline auto convert_precision(const Eigen::MatrixBase<Derived>& mat) {
  using FromScalar = typename Derived::Scalar;
  if constexpr (std::is_same<FromScalar, ToScalar>::value) {
    return mat.eval();
  } else {
    return mat.template cast<ToScalar>().eval();
  }
}

// Convert MarginalDistribution to float precision (for computation)
inline auto to_float(const MarginalDistribution& dist) {
  struct FloatMarginal {
    Eigen::VectorXf mean;
    Eigen::VectorXf variance;

    FloatMarginal(const MarginalDistribution& d)
      : mean(d.mean.cast<float>()),
        variance(d.covariance.diagonal().cast<float>()) {}

    // Convert back to double
    MarginalDistribution to_double() const {
      return MarginalDistribution(
        mean.cast<double>(),
        variance.cast<double>()
      );
    }
  };
  return FloatMarginal(dist);
}

// Convert JointDistribution to float precision (for computation)
inline auto to_float(const JointDistribution& dist) {
  struct FloatJoint {
    Eigen::VectorXf mean;
    Eigen::MatrixXf covariance;

    FloatJoint(const JointDistribution& d)
      : mean(d.mean.cast<float>()),
        covariance(d.covariance.cast<float>()) {}

    // Convert back to double
    JointDistribution to_double() const {
      return JointDistribution(
        mean.cast<double>(),
        covariance.cast<double>()
      );
    }
  };
  return FloatJoint(dist);
}

} // namespace albatross

#endif // ALBATROSS_CORE_SCALAR_TRAITS_H
