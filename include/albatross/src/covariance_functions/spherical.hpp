/*
 * Copyright (C) 2025 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef ALBATROSS_COVARIANCE_FUNCTIONS_SPHERICAL_H
#define ALBATROSS_COVARIANCE_FUNCTIONS_SPHERICAL_H

namespace albatross {

namespace constant {
// Default number of terms to use in a cosine series in longitude
static constexpr const std::size_t cDefaultLongitudinalMaternSeriesLength{500};
} // namespace constant

namespace detail {

// Convert a Whittle-Matern SPDE distance parameter (canonically
// `\kappa`) to a Matern GP covariance kernel length scale
// (canonically `\rho` or `l`), or the reverse.  They are inverses
// with a constant factor that decouples the degrees of freedom from
// either formula.
inline double length_scale_spde_to_gp(double kappa, double degrees_of_freedom) {
  return std::sqrt(2 * degrees_of_freedom) / kappa;
}

// Compute the ratio cosh(kappa * distance) / sinh(kappa * pi),
// stably.  For small kappa, we use a Taylor series expansion:
//
//   cosh(kd) ~ 1 + 1/2 (kd)^2
//   sinh(kpi) ~ kpi + (kpi)^3 / 6
//
// Otherwise, we use the definitions of `cosh` and `sinh`:
//
//   cosh(x) = (e^{x} + e^{-x}) / 2
//   sinh(x) = (e^{x} - e^{-x}) / 2
//
// and combine them:
//
//   2 * cosh(kd) = e^{kd} + e^{-kd}
//   2 * sinh(kpi) = e^{kpi} - e^{-kpi}
//
//   cosh(kd) / sinh(kpi) = (e^{kd} + e^{-kd}) / (e^{kpi} - e^{-kpi})
//                        = e^{-k(pi - d)} (1 + e^{-2kd}) / (1 - e^{-2kpi})
//
// into a formula that only takes e to negative powers, avoiding
// underflow.
inline double cosh_over_sinh_stable(double kappa, double distance) {
  double kd = kappa * distance;
  double kpi = kappa * M_PI;

  if (kappa < 1e-3) {
    return (1 + 0.5 * kd * kd) / (kpi + kpi * kpi * kpi / 6.);
  }

  double ekd = std::exp(-2 * kd);
  double ky = kappa * (M_PI - distance);
  double eky = std::exp(-ky);

  // use `e^(kpi) - 1` built-in
  return eky * (1 + ekd) / (-(std::expm1(-2 * kpi)));
}

// This is the derivative of `cosh_over_sinh_stable()` with respect to
// argument kappa.  We differentiate the exponent form to try to
// preserve some semblance of numerical stability.
//
// Let
//
//   a = pi - distance
//   d = 2 * distance
//   tau = 2 * pi
//   d - tau = 2 * (distance - pi)
//           = -2a
//
// Then one application of the product rule and one application of the
// quotient rule gives:
//
//   d/dkappa [e^{-ka} (1 + e^{-kd}) / (1 - e^{-ktau})] =
//     -a e^{-ka} * (1 + e^{-kd}) / (1 - e^{-ktau}) -
//     e^{-ka} / (1 - e^{-ktau})^2 *
//       (d e^{-kd} + tau e^{-ktau} + 2a e^{-k (d + tau)})
//
inline double ddkappa_cosh_over_sinh_stable(double kappa, double distance) {
  double kpi = kappa * M_PI;
  double a = M_PI - distance;
  double ka = kappa * a;
  double eka = std::exp(-ka);
  double ekd = std::exp(-2 * kappa * distance);
  double ekpi = std::exp(-2 * kpi);
  double denom = -std::expm1(-2 * kpi);
  double ekdpi = std::exp(-2 * kappa * (distance + M_PI));

  return -a * eka * (1 + ekd) / denom -
         2 * eka / (denom * denom) * (distance * ekd + M_PI * ekpi + a * ekdpi);
}

// Second derivative of `cosh_over_sinh_stable()` with respect to
// argument kappa.  This is done mechanically as above, more
// tediously.  As before, let
//
//   a = pi - distance
//   d = 2 * distance
//   tau = 2 * pi
//   d - tau = 2 * (distance - pi)
//           = -2a
//
// then by repeated application of calculus rules:
//
//   d^2 / dkappa^2 [e^{-ka} (1 + e^{-kd}) / (1 - e^{-ktau})] =
//     -a * d/dkappa [...] +
//     e^{-ka} (a (1 - e^{-ktau})^2 +
//              2 * tau * (e^{-ktau} - e^{-2ktau})) /
//         (1 - e^{-ktau})^4 +
//     e^{-ka} / (1 - e^{-ktau})^2 * 4 * (distance^2 e^{-kd} +
//                                        pi^2 e^{-ktau} +
//                                        a(distance + pi) e^{-k(d + tau)}
inline double d2dkappa2_cosh_over_sinh_stable(double kappa, double distance) {
  double kpi = kappa * M_PI;
  double a = M_PI - distance;
  double ka = kappa * a;
  double eka = std::exp(-ka);
  double ekd = std::exp(-2 * kappa * distance);
  double ekpi = std::exp(-2 * kpi);
  double denom = -std::expm1(-2 * kpi);
  double denom2 = denom * denom;
  double ekdpi = std::exp(-2 * kappa * (distance + M_PI));

  double beta = 2 * (distance * ekd + M_PI * ekpi + a * ekdpi);
  double h1 = eka * (a / denom2 + 4 * M_PI * (ekpi - std::exp(-4 * kpi)) /
                                      (denom2 * denom2));
  double h2 = eka / denom2 * 4 *
              (distance * distance * ekd + M_PI * M_PI * ekpi +
               a * (distance + M_PI) * ekdpi);

  double hk = ddkappa_cosh_over_sinh_stable(kappa, distance);

  return -a * hk + h1 * beta + h2;
}

} // namespace detail

struct SumAndLastUpdate {
  double sum;
  double last_update;
};

// Because longitude wraps around, we have to define a Matern-like
// kernel spectrally, in terms of a cosine series:
//
//   \Sum_{m = 1}^\infty \frac{cos(m \Delta)}{(\kappa^2 + m^2)^{-\alpha}}
//
// For special values of interest, we can do some algebra and get a
// nicer form, but for testing purposes and for arbitrary degrees of
// freedom, this function does the actual sum of a series.
inline SumAndLastUpdate matern_longitudinal_covariance_series(
    double distance, double length_scale, double degrees_of_freedom,
    std::size_t num_terms = constant::cDefaultLongitudinalMaternSeriesLength) {
  // This kernel operates only on longitude, so it's unidimensional.
  static constexpr const double cDimensions = 1.;
  const double alpha{degrees_of_freedom + cDimensions / 2.};
  const double kappa =
      detail::length_scale_spde_to_gp(length_scale, degrees_of_freedom);

  double sum = 0;
  double last_update = 0;
  // Sum downwards in `m` to preserve accuracy
  for (Eigen::Index m_ = cast::to_index(num_terms); m_ >= 0; --m_) {
    const double m = static_cast<double>(m_ + 1);
    double update =
        std::cos(m * distance) * std::pow(kappa * kappa + m * m, -alpha);
    sum += update;
    last_update = update;
  }
  return {sum, last_update};
}

// Special case for exponential / Brownian motion kernel with nu = 1/2
//
// Here we use a closed-form expression for the value of a Matern-like
// kernel on a circle that we derive from the cosine series
// definition.  This derivation is too long to write out in code
// comments, but check the following MO questions for inspiration.
//
// https://math.stackexchange.com/questions/208317/show-sum-n-0-infty-frac1a2n2-frac1a-pi-coth-a-pi2a2
// https://math.stackexchange.com/questions/1479666/calculate-sum-limits-n-1-infty-fraca-cosnxa2n2/1843540
// https://math.stackexchange.com/questions/433588/show-frac-cosha-pi-x-sinha-pi-frac1a-pi-frac2-pi
//
// Steps:
//
//  - write the fourier series expansion of e^{\kappa x} (note that x
//    here is a dummy variable we will define later to make the series
//    work)
//
//  - do two rounds of integration by parts on the cosine coefficient
//    definition to get an analytic version
//
//  - ignore the sine coefficients because the kernel is even / we
//    want a cosine series
//
//  - even part of e^{\kappa x} is equal to cosh(\kappa x)
//
//  - cos(m \pi) cos(m x) = cos(m (\pi - x)) because cos(m \pi) is
//    just a sign flip (-1)^m
//
//  - define x = \pi - \Delta (\Delta being the longitude difference)
//    and you get
//
//    \Sum_{m = 1}^{\infty \frac{\cos(m \Delta)}{\kappa^2 + m^2} =
//      \frac{\pi \cosh(\kappa \|\pi - \Delta\|)}{2 \kappa \sinh(\pi \kappa)} -
//      \frac{1}{2 * \kappa^2}
//
//  - ask an LLM to check your work
inline double matern_12_longitudinal_covariance(double distance,
                                                double length_scale) {
  ALBATROSS_ASSERT(length_scale >= 0 &&
                   "Longitude length scale must be nonnegative!");
  ALBATROSS_ASSERT(distance >= 0 && "Longitude distance must be nonnegative!");
  ALBATROSS_ASSERT(distance <= M_PI &&
                   "Longitude distance can't be greater than half a circle!");
  static constexpr const double cDegreesOfFreedom{0.5};
  const double kappa =
      detail::length_scale_spde_to_gp(length_scale, cDegreesOfFreedom);
  // std::cerr << "kappa = " << kappa << "\n";
  const double pi_minus_distance{std::fabs(M_PI - distance)};
  // std::cerr << "pi_minus_distance = " << pi_minus_distance << "\n";
  // std::cerr << "kappa * pmd = " << kappa * pi_minus_distance << "\n";
  // std::cerr << "kappa * pi = " << kappa * M_PI << "\n";
  // std::cerr << "cosh(kappa * pmd) = " << std::cosh(kappa * pi_minus_distance)
  //           << "\n";
  // std::cerr << "sinh(kappa * pi) = " << std::sinh(kappa * M_PI) << "\n";
  double ratio = detail::cosh_over_sinh_stable(kappa, pi_minus_distance);
  // std::cerr << "ratio = " << ratio << "\n";
  return 0.5 * (M_PI / kappa * ratio - 1 / (kappa * kappa));
}

// This special case for Matern-like covariance on a circle is derived
// from the case for nu = 1/2 above.  For nu = 3/2, the series formula
// is the same except that the denominator is (kappa^2 + m^2)^2 rather
// than (kappa^2 + m^2).  The trick here is to differentiate the whole
// series with respect to kappa^2; then we get the right denominator.
// Now just (just!) differentiate the closed-form expression for nu =
// 1/2 with respect to kappa^2 and we have another closed-form
// expression.
//
// The derivative of the ratio cosh(kappa * (pi - distance)) /
// sinh(kappa * pi) term with respect to kappa is given by the
// derivative function in the `detail` namespace; here we implement
// the product rule.  Note that we wanted to differentiate originally
// by kappa^2, so we must prepend a 1 / (2 * kappa) term (chain rule)
// to the ratio terms.
inline double matern_32_longitudinal_covariance(double distance,
                                                double length_scale) {
  ALBATROSS_ASSERT(length_scale >= 0 &&
                   "Longitude length scale must be nonnegative!");
  ALBATROSS_ASSERT(distance >= 0 && "Longitude distance must be nonnegative!");
  ALBATROSS_ASSERT(distance <= M_PI &&
                   "Longitude distance can't be greater than half a circle!");
  static constexpr const double cDegreesOfFreedom{1.5};
  const double kappa =
      detail::length_scale_spde_to_gp(length_scale, cDegreesOfFreedom);
  const double pi_minus_distance{std::fabs(M_PI - distance)};

  double h = detail::cosh_over_sinh_stable(kappa, pi_minus_distance);
  double hk = detail::ddkappa_cosh_over_sinh_stable(kappa, pi_minus_distance);
  double fk = M_PI / (2 * kappa) * hk - M_PI / (2 * kappa * kappa) * h +
              1 / (kappa * kappa * kappa);
  return -1 / (2 * kappa) * fk;
}

// In this special case, the denominator of the series term is just
// (kappa^2 + m^2)^3, so we can repeat the same process of
// differentiation.
inline double matern_52_longitudinal_covariance(double distance,
                                                double length_scale) {
  ALBATROSS_ASSERT(length_scale >= 0 &&
                   "Longitude length scale must be nonnegative!");
  ALBATROSS_ASSERT(distance >= 0 && "Longitude distance must be nonnegative!");
  ALBATROSS_ASSERT(distance <= M_PI &&
                   "Longitude distance can't be greater than half a circle!");
  static constexpr const double cDegreesOfFreedom{2.5};
  const double kappa =
      detail::length_scale_spde_to_gp(length_scale, cDegreesOfFreedom);
  const double kappa2 = kappa * kappa;
  const double pi_minus_distance{std::fabs(M_PI - distance)};

  double h = detail::cosh_over_sinh_stable(kappa, pi_minus_distance);
  double hk = detail::ddkappa_cosh_over_sinh_stable(kappa, pi_minus_distance);
  double hkk =
      detail::d2dkappa2_cosh_over_sinh_stable(kappa, pi_minus_distance);

  double fk =
      M_PI / (2 * kappa) * hk - M_PI / (2 * kappa2) * h + 1 / (kappa2 * kappa);

  double fkk = M_PI / (2 * kappa) * hkk - M_PI / kappa2 * hk +
               M_PI / (kappa2 * kappa) - 3 / (kappa2 * kappa2);

  return 1 / (8 * kappa2) * fkk - 1 / (8 * kappa2 * kappa) * fk;
}

} // namespace albatross

#endif // ALBATROSS_COVARIANCE_FUNCTIONS_SPHERICAL_H
