/*
 * Copyright (C) 2026 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef INCLUDE_ALBATROSS_SRC_POLYNOMIAL_CHEBYSHEV_H_
#define INCLUDE_ALBATROSS_SRC_POLYNOMIAL_CHEBYSHEV_H_

#include <Eigen/Core>

namespace albatross {

namespace detail {

// Chebyshev polynomials of the first kind T_n base cases
inline Eigen::ArrayXd chebyshev_t0(const Eigen::ArrayXd &x) {
  return Eigen::ArrayXd::Constant(x.size(), 1.0);
}

inline Eigen::ArrayXd chebyshev_t1(const Eigen::ArrayXd &x) { return x; }

inline Eigen::ArrayXd chebyshev_t2(const Eigen::ArrayXd &x) {
  return 2 * x * x - chebyshev_t0(x);
}

// Chebyshev polynomials of the second kind U_n base cases
inline Eigen::ArrayXd chebyshev_u0(const Eigen::ArrayXd &x) {
  return Eigen::ArrayXd::Constant(x.size(), 1.0);
}

inline Eigen::ArrayXd chebyshev_u1(const Eigen::ArrayXd &x) { return 2 * x; }

inline Eigen::ArrayXd chebyshev_u2(const Eigen::ArrayXd &x) {
  return 4 * x * x - chebyshev_u0(x);
}

} // namespace detail

// Compute the degree-n Chebyshev polynomial of the first kind T_n(x)
// at the given locations x.  All elements of x must lie in [-1, 1].
inline Eigen::ArrayXd chebyshev_t(Eigen::Index n, const Eigen::ArrayXd &x) {
  ALBATROSS_ASSERT(n >= 0 &&
                   "Cannot compute Chebyshev T function for negative degree!");
  if (n == 0) {
    return detail::chebyshev_t0(x);
  }

  if (n == 1) {
    return detail::chebyshev_t1(x);
  }

  if (n == 2) {
    return detail::chebyshev_t2(x);
  }

  Eigen::ArrayXd tn_minus_1(detail::chebyshev_t2(x));
  Eigen::ArrayXd tn_minus_2(detail::chebyshev_t1(x));
  Eigen::ArrayXd tn(tn_minus_1);

  for (Eigen::Index i = 3; i <= n; ++i) {
    tn = 2 * x * tn_minus_1 - tn_minus_2;
    tn_minus_2 = tn_minus_1;
    tn_minus_1 = tn;
  }

  return tn;
}

// Compute the Chebyshev basis matrix up to degree n_max at the given
// locations x.  Column k of the resulting basis matrix is T_k(x).
// All elements of x must lie in [-1, 1].
inline Eigen::ArrayXXd chebyshev_t_phi(Eigen::Index n_max,
                                       const Eigen::ArrayXd &x) {
  ALBATROSS_ASSERT(n_max >= 0 &&
                   "Cannot produce empty Chebyshev basis matrix!");
  Eigen::ArrayXXd phi(x.size(), n_max + 1);
  phi.col(0) = detail::chebyshev_t0(x);

  if (n_max > 0) {
    phi.col(1) = detail::chebyshev_t1(x);
  }

  if (n_max > 1) {
    phi.col(2) = detail::chebyshev_t2(x);
  }

  for (Eigen::Index i = 3; i < n_max + 1; ++i) {
    phi.col(i) = 2 * x * phi.col(i - 1) - phi.col(i - 2);
  }

  return phi;
}

// Compute the degree-n Chebyshev polynomial of the second kind U_n(x)
// at the given locations x.  All elements of x must lie in [-1, 1].
inline Eigen::ArrayXd chebyshev_u(Eigen::Index n, const Eigen::ArrayXd &x) {
  ALBATROSS_ASSERT(n >= 0 &&
                   "Cannot compute Chebyshev U function for negative degree!");
  if (n == 0) {
    return detail::chebyshev_u0(x);
  }

  if (n == 1) {
    return detail::chebyshev_u1(x);
  }

  if (n == 2) {
    return detail::chebyshev_u2(x);
  }

  Eigen::ArrayXd un_minus_1(detail::chebyshev_u2(x));
  Eigen::ArrayXd un_minus_2(detail::chebyshev_u1(x));
  Eigen::ArrayXd un(un_minus_1);

  for (Eigen::Index i = 3; i <= n; ++i) {
    un = 2 * x * un_minus_1 - un_minus_2;
    un_minus_2 = un_minus_1;
    un_minus_1 = un;
  }

  return un;
}

// Compute the Chebyshev basis matrix of the second kind up to degree n_max
// at the given locations x.  Column k of the resulting basis matrix is U_k(x).
// All elements of x must lie in [-1, 1].
inline Eigen::ArrayXXd chebyshev_u_phi(Eigen::Index n_max,
                                       const Eigen::ArrayXd &x) {
  ALBATROSS_ASSERT(n_max >= 0 &&
                   "Cannot produce empty Chebyshev basis matrix!");
  Eigen::ArrayXXd phi(x.size(), n_max + 1);
  phi.col(0) = detail::chebyshev_u0(x);

  if (n_max > 0) {
    phi.col(1) = detail::chebyshev_u1(x);
  }

  if (n_max > 1) {
    phi.col(2) = detail::chebyshev_u2(x);
  }

  for (Eigen::Index i = 3; i < n_max + 1; ++i) {
    phi.col(i) = 2 * x * phi.col(i - 1) - phi.col(i - 2);
  }

  return phi;
}

} // namespace albatross

#endif // INCLUDE_ALBATROSS_SRC_POLYNOMIAL_CHEBYSHEV_H_
