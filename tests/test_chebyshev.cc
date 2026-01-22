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
#include <gtest/gtest.h>

#include <albatross/Common>

#include <cmath>
#include <random>

namespace albatross {

/*
 * Edge case tests for Chebyshev polynomials
 */
TEST(test_chebyshev, test_empty_input) {
  Eigen::ArrayXd empty;

  // chebyshev_t with empty input should return empty
  for (Eigen::Index n = 0; n <= 5; ++n) {
    Eigen::ArrayXd result = chebyshev_t(n, empty);
    EXPECT_EQ(result.size(), 0);
  }

  // chebyshev_t_phi with empty input should return empty matrix
  Eigen::ArrayXXd phi = chebyshev_t_phi(5, empty);
  EXPECT_EQ(phi.rows(), 0);
  EXPECT_EQ(phi.cols(), 6);
}

TEST(test_chebyshev, test_single_element) {
  Eigen::ArrayXd x(1);
  x << 0.5;

  // T_0(0.5) = 1
  Eigen::ArrayXd t0 = chebyshev_t(0, x);
  EXPECT_EQ(t0.size(), 1);
  EXPECT_DOUBLE_EQ(t0(0), 1.0);

  // T_1(0.5) = 0.5
  Eigen::ArrayXd t1 = chebyshev_t(1, x);
  EXPECT_EQ(t1.size(), 1);
  EXPECT_DOUBLE_EQ(t1(0), 0.5);

  // T_2(0.5) = 2*(0.5)^2 - 1 = -0.5
  Eigen::ArrayXd t2 = chebyshev_t(2, x);
  EXPECT_EQ(t2.size(), 1);
  EXPECT_DOUBLE_EQ(t2(0), -0.5);
}

TEST(test_chebyshev, test_boundary_values) {
  Eigen::ArrayXd x(3);
  x << -1.0, 0.0, 1.0;

  // T_n(-1) = (-1)^n, T_n(1) = 1 for all n
  for (Eigen::Index n = 0; n <= 10; ++n) {
    Eigen::ArrayXd tn = chebyshev_t(n, x);

    double expected_at_minus_one = (n % 2 == 0) ? 1.0 : -1.0;
    EXPECT_DOUBLE_EQ(tn(0), expected_at_minus_one)
        << "T_" << n << "(-1) should be " << expected_at_minus_one;

    // T_n(0) = cos(n*pi/2)
    double expected_at_zero = std::cos(n * M_PI / 2.0);
    EXPECT_NEAR(tn(1), expected_at_zero, 1e-14)
        << "T_" << n << "(0) should be " << expected_at_zero;

    EXPECT_DOUBLE_EQ(tn(2), 1.0) << "T_" << n << "(1) should be 1";
  }
}

TEST(test_chebyshev, test_base_cases) {
  Eigen::ArrayXd x(5);
  x << -0.8, -0.3, 0.0, 0.4, 0.9;

  // T_0(x) = 1
  Eigen::ArrayXd t0 = chebyshev_t(0, x);
  for (Eigen::Index i = 0; i < x.size(); ++i) {
    EXPECT_DOUBLE_EQ(t0(i), 1.0);
  }

  // T_1(x) = x
  Eigen::ArrayXd t1 = chebyshev_t(1, x);
  for (Eigen::Index i = 0; i < x.size(); ++i) {
    EXPECT_DOUBLE_EQ(t1(i), x(i));
  }

  // T_2(x) = 2x^2 - 1
  Eigen::ArrayXd t2 = chebyshev_t(2, x);
  for (Eigen::Index i = 0; i < x.size(); ++i) {
    double expected = 2.0 * x(i) * x(i) - 1.0;
    EXPECT_DOUBLE_EQ(t2(i), expected);
  }
}

TEST(test_chebyshev, test_phi_n_max_zero) {
  Eigen::ArrayXd x(3);
  x << -0.5, 0.0, 0.5;

  Eigen::ArrayXXd phi = chebyshev_t_phi(0, x);

  EXPECT_EQ(phi.rows(), 3);
  EXPECT_EQ(phi.cols(), 1);

  // All values should be 1 (T_0 = 1)
  for (Eigen::Index i = 0; i < phi.rows(); ++i) {
    EXPECT_DOUBLE_EQ(phi(i, 0), 1.0);
  }
}

/*
 * Trigonometric identity test: T_n(cos(theta)) = cos(n * theta)
 */
TEST(test_chebyshev, test_trig_identity) {
  std::default_random_engine gen(2012);
  std::uniform_real_distribution<double> theta_dist(0.0, M_PI);
  std::uniform_int_distribution<Eigen::Index> size_dist(10, 50);

  constexpr std::size_t kNumIterations = 100;
  constexpr Eigen::Index kMaxDegree = 10;
  constexpr double kTolerance = 1e-12;

  for (std::size_t iter = 0; iter < kNumIterations; ++iter) {
    Eigen::Index size = size_dist(gen);
    Eigen::ArrayXd theta(size);
    for (Eigen::Index i = 0; i < size; ++i) {
      theta(i) = theta_dist(gen);
    }

    Eigen::ArrayXd x = theta.cos();

    for (Eigen::Index n = 0; n <= kMaxDegree; ++n) {
      Eigen::ArrayXd tn = chebyshev_t(n, x);
      Eigen::ArrayXd expected = (n * theta).cos();

      for (Eigen::Index i = 0; i < size; ++i) {
        EXPECT_NEAR(tn(i), expected(i), kTolerance)
            << "T_" << n << "(cos(" << theta(i) << ")) should equal cos(" << n
            << " * " << theta(i) << ")";
      }
    }
  }
}

/*
 * Symmetry test: T_n(-x) = (-1)^n * T_n(x)
 */
TEST(test_chebyshev, test_symmetry) {
  std::default_random_engine gen(2012);
  std::uniform_real_distribution<double> x_dist(0.0, 1.0);
  std::uniform_int_distribution<Eigen::Index> size_dist(10, 50);

  constexpr std::size_t kNumIterations = 100;
  constexpr Eigen::Index kMaxDegree = 10;
  constexpr double kTolerance = 1e-14;

  for (std::size_t iter = 0; iter < kNumIterations; ++iter) {
    Eigen::Index size = size_dist(gen);
    Eigen::ArrayXd x(size);
    for (Eigen::Index i = 0; i < size; ++i) {
      x(i) = x_dist(gen);
    }

    for (Eigen::Index n = 0; n <= kMaxDegree; ++n) {
      Eigen::ArrayXd tn_pos = chebyshev_t(n, x);
      Eigen::ArrayXd tn_neg = chebyshev_t(n, -x);

      double sign = (n % 2 == 0) ? 1.0 : -1.0;

      for (Eigen::Index i = 0; i < size; ++i) {
        EXPECT_NEAR(tn_neg(i), sign * tn_pos(i), kTolerance)
            << "T_" << n << "(-" << x(i) << ") should equal " << sign << " * T_"
            << n << "(" << x(i) << ")";
      }
    }
  }
}

/*
 * Scipy reference values test
 * Values generated using scipy.special.eval_chebyt
 */
TEST(test_chebyshev, test_scipy_reference) {
  // Reference values: {degree, x, T_n(x)}
  // Generated with scipy.special.eval_chebyt
  struct ReferencePoint {
    Eigen::Index n;
    double x;
    double expected;
  };

  // clang-format off
  const std::vector<ReferencePoint> reference_points = {
      {0, -1.0, 1.000000000000000e+00},
      {0, -0.5, 1.000000000000000e+00},
      {0, 0.0, 1.000000000000000e+00},
      {0, 0.5, 1.000000000000000e+00},
      {0, 1.0, 1.000000000000000e+00},
      {0, 0.123, 1.000000000000000e+00},
      {0, -0.789, 1.000000000000000e+00},
      {0, 0.999, 1.000000000000000e+00},
      {1, -1.0, -1.000000000000000e+00},
      {1, -0.5, -5.000000000000000e-01},
      {1, 0.0, 0.000000000000000e+00},
      {1, 0.5, 5.000000000000000e-01},
      {1, 1.0, 1.000000000000000e+00},
      {1, 0.123, 1.230000000000000e-01},
      {1, -0.789, -7.890000000000000e-01},
      {1, 0.999, 9.990000000000000e-01},
      {2, -1.0, 1.000000000000000e+00},
      {2, -0.5, -5.000000000000000e-01},
      {2, 0.0, -1.000000000000000e+00},
      {2, 0.5, -5.000000000000000e-01},
      {2, 1.0, 1.000000000000000e+00},
      {2, 0.123, -9.697420000000000e-01},
      {2, -0.789, 2.450420000000002e-01},
      {2, 0.999, 9.960020000000001e-01},
      {3, -1.0, -1.000000000000000e+00},
      {3, -0.5, 1.000000000000000e+00},
      {3, 0.0, 0.000000000000000e+00},
      {3, 0.5, -1.000000000000000e+00},
      {3, 1.0, 1.000000000000000e+00},
      {3, 0.123, -3.615565320000000e-01},
      {3, -0.789, 4.023237239999997e-01},
      {3, 0.999, 9.910119959999998e-01},
      {4, -1.0, 1.000000000000000e+00},
      {4, -0.5, -5.000000000000000e-01},
      {4, 0.0, 1.000000000000000e+00},
      {4, 0.5, -5.000000000000000e-01},
      {4, 1.0, 1.000000000000000e+00},
      {4, 0.123, 8.807990931280000e-01},
      {4, -0.789, -8.799088364719999e-01},
      {4, 0.999, 9.840399680079999e-01},
      {5, -1.0, -1.000000000000000e+00},
      {5, -0.5, -5.000000000000000e-01},
      {5, 0.0, 0.000000000000000e+00},
      {5, 0.5, 5.000000000000000e-01},
      {5, 1.0, 1.000000000000000e+00},
      {5, 0.123, 5.782331089094880e-01},
      {5, -0.789, 9.861724199528162e-01},
      {5, 0.999, 9.750998600799843e-01},
      {6, -1.0, 1.000000000000000e+00},
      {6, -0.5, 1.000000000000000e+00},
      {6, 0.0, -1.000000000000000e+00},
      {6, 0.5, 1.000000000000000e+00},
      {6, 1.0, 1.000000000000000e+00},
      {6, 0.123, -7.385537483362659e-01},
      {6, -0.789, -6.762712422135442e-01},
      {6, 0.999, 9.642095524318082e-01},
      {7, -1.0, -1.000000000000000e+00},
      {7, -0.5, -5.000000000000000e-01},
      {7, 0.0, 0.000000000000000e+00},
      {7, 0.5, 5.000000000000000e-01},
      {7, 1.0, 1.000000000000000e+00},
      {7, 0.123, -7.599173310002094e-01},
      {7, -0.789, 8.098360026015650e-02},
      {7, 0.999, 9.513908256787689e-01},
  };
  // clang-format on

  constexpr double kTolerance = 1e-14;

  for (const auto &ref : reference_points) {
    Eigen::ArrayXd x(1);
    x << ref.x;
    Eigen::ArrayXd result = chebyshev_t(ref.n, x);

    EXPECT_NEAR(result(0), ref.expected, kTolerance)
        << "T_" << ref.n << "(" << ref.x << ") mismatch with scipy reference";
  }
}

/*
 * Basis matrix consistency test:
 * chebyshev_t_phi(n_max, x).col(k) should equal chebyshev_t(k, x)
 */
TEST(test_chebyshev, test_phi_consistency) {
  std::default_random_engine gen(2012);
  std::uniform_real_distribution<double> x_dist(-1.0, 1.0);
  std::uniform_int_distribution<Eigen::Index> size_dist(10, 50);

  constexpr std::size_t kNumIterations = 100;
  constexpr Eigen::Index kMaxDegree = 15;
  constexpr double kTolerance = 1e-14;

  for (std::size_t iter = 0; iter < kNumIterations; ++iter) {
    Eigen::Index size = size_dist(gen);
    Eigen::ArrayXd x(size);
    for (Eigen::Index i = 0; i < size; ++i) {
      x(i) = x_dist(gen);
    }

    Eigen::ArrayXXd phi = chebyshev_t_phi(kMaxDegree, x);

    // Check dimensions
    EXPECT_EQ(phi.rows(), size);
    EXPECT_EQ(phi.cols(), kMaxDegree + 1);

    // Check each column matches chebyshev_t
    for (Eigen::Index k = 0; k <= kMaxDegree; ++k) {
      Eigen::ArrayXd tk = chebyshev_t(k, x);
      for (Eigen::Index i = 0; i < size; ++i) {
        EXPECT_NEAR(phi(i, k), tk(i), kTolerance)
            << "phi.col(" << k << ") should equal chebyshev_t(" << k << ", x)";
      }
    }
  }
}

/*
 * Recurrence relation test:
 * T_n(x) = 2*x*T_{n-1}(x) - T_{n-2}(x) for n >= 2
 */
TEST(test_chebyshev, test_recurrence_relation) {
  std::default_random_engine gen(2012);
  std::uniform_real_distribution<double> x_dist(-1.0, 1.0);
  std::uniform_int_distribution<Eigen::Index> size_dist(10, 50);

  constexpr std::size_t kNumIterations = 100;
  constexpr Eigen::Index kMinDegree = 3;
  constexpr Eigen::Index kMaxDegree = 15;
  constexpr double kTolerance = 1e-13;

  for (std::size_t iter = 0; iter < kNumIterations; ++iter) {
    Eigen::Index size = size_dist(gen);
    Eigen::ArrayXd x(size);
    for (Eigen::Index i = 0; i < size; ++i) {
      x(i) = x_dist(gen);
    }

    for (Eigen::Index n = kMinDegree; n <= kMaxDegree; ++n) {
      Eigen::ArrayXd tn = chebyshev_t(n, x);
      Eigen::ArrayXd tn_minus_1 = chebyshev_t(n - 1, x);
      Eigen::ArrayXd tn_minus_2 = chebyshev_t(n - 2, x);

      Eigen::ArrayXd expected = 2.0 * x * tn_minus_1 - tn_minus_2;

      for (Eigen::Index i = 0; i < size; ++i) {
        EXPECT_NEAR(tn(i), expected(i), kTolerance)
            << "T_" << n << "(x) should equal 2*x*T_" << n - 1 << "(x) - T_"
            << n - 2 << "(x)";
      }
    }
  }
}

/*
 * Test higher degrees to verify numerical stability
 */
TEST(test_chebyshev, test_higher_degrees) {
  Eigen::ArrayXd x(5);
  x << -0.9, -0.5, 0.0, 0.5, 0.9;

  // Test degrees up to 20
  for (Eigen::Index n = 0; n <= 20; ++n) {
    Eigen::ArrayXd tn = chebyshev_t(n, x);

    // All Chebyshev polynomials should be bounded by [-1, 1] on [-1, 1]
    for (Eigen::Index i = 0; i < x.size(); ++i) {
      EXPECT_GE(tn(i), -1.0 - 1e-10)
          << "T_" << n << "(" << x(i) << ") should be >= -1";
      EXPECT_LE(tn(i), 1.0 + 1e-10)
          << "T_" << n << "(" << x(i) << ") should be <= 1";
    }
  }

  // Verify T_n(1) = 1 and T_n(-1) = (-1)^n for higher degrees
  Eigen::ArrayXd endpoints(2);
  endpoints << -1.0, 1.0;

  for (Eigen::Index n = 0; n <= 20; ++n) {
    Eigen::ArrayXd tn = chebyshev_t(n, endpoints);
    double expected_minus_one = (n % 2 == 0) ? 1.0 : -1.0;
    EXPECT_DOUBLE_EQ(tn(0), expected_minus_one);
    EXPECT_DOUBLE_EQ(tn(1), 1.0);
  }
}

/*
 * =========================================================================
 * Chebyshev polynomials of the second kind U_n tests
 * =========================================================================
 */

/*
 * Edge case tests for U_n
 */
TEST(test_chebyshev_u, test_empty_input) {
  Eigen::ArrayXd empty;

  // chebyshev_u with empty input should return empty
  for (Eigen::Index n = 0; n <= 5; ++n) {
    Eigen::ArrayXd result = chebyshev_u(n, empty);
    EXPECT_EQ(result.size(), 0);
  }

  // chebyshev_u_phi with empty input should return empty matrix
  Eigen::ArrayXXd phi = chebyshev_u_phi(5, empty);
  EXPECT_EQ(phi.rows(), 0);
  EXPECT_EQ(phi.cols(), 6);
}

TEST(test_chebyshev_u, test_single_element) {
  Eigen::ArrayXd x(1);
  x << 0.5;

  // U_0(0.5) = 1
  Eigen::ArrayXd u0 = chebyshev_u(0, x);
  EXPECT_EQ(u0.size(), 1);
  EXPECT_DOUBLE_EQ(u0(0), 1.0);

  // U_1(0.5) = 2*0.5 = 1
  Eigen::ArrayXd u1 = chebyshev_u(1, x);
  EXPECT_EQ(u1.size(), 1);
  EXPECT_DOUBLE_EQ(u1(0), 1.0);

  // U_2(0.5) = 4*(0.5)^2 - 1 = 0
  Eigen::ArrayXd u2 = chebyshev_u(2, x);
  EXPECT_EQ(u2.size(), 1);
  EXPECT_DOUBLE_EQ(u2(0), 0.0);
}

TEST(test_chebyshev_u, test_boundary_values) {
  Eigen::ArrayXd x(3);
  x << -1.0, 0.0, 1.0;

  // U_n(-1) = (-1)^n * (n+1), U_n(1) = n+1 for all n
  for (Eigen::Index n = 0; n <= 10; ++n) {
    Eigen::ArrayXd un = chebyshev_u(n, x);

    double expected_at_minus_one = ((n % 2 == 0) ? 1.0 : -1.0) * (n + 1);
    EXPECT_DOUBLE_EQ(un(0), expected_at_minus_one)
        << "U_" << n << "(-1) should be " << expected_at_minus_one;

    // U_n(0) = cos(n*pi/2) * (n % 4 pattern: 1, 0, -1, 0, ...)
    // More precisely: U_n(0) = 0 if n odd, (-1)^(n/2) if n even
    double expected_at_zero;
    if (n % 2 == 1) {
      expected_at_zero = 0.0;
    } else {
      expected_at_zero = ((n / 2) % 2 == 0) ? 1.0 : -1.0;
    }
    EXPECT_NEAR(un(1), expected_at_zero, 1e-14)
        << "U_" << n << "(0) should be " << expected_at_zero;

    EXPECT_DOUBLE_EQ(un(2), static_cast<double>(n + 1))
        << "U_" << n << "(1) should be " << n + 1;
  }
}

TEST(test_chebyshev_u, test_base_cases) {
  Eigen::ArrayXd x(5);
  x << -0.8, -0.3, 0.0, 0.4, 0.9;

  // U_0(x) = 1
  Eigen::ArrayXd u0 = chebyshev_u(0, x);
  for (Eigen::Index i = 0; i < x.size(); ++i) {
    EXPECT_DOUBLE_EQ(u0(i), 1.0);
  }

  // U_1(x) = 2x
  Eigen::ArrayXd u1 = chebyshev_u(1, x);
  for (Eigen::Index i = 0; i < x.size(); ++i) {
    EXPECT_DOUBLE_EQ(u1(i), 2.0 * x(i));
  }

  // U_2(x) = 4x^2 - 1
  Eigen::ArrayXd u2 = chebyshev_u(2, x);
  for (Eigen::Index i = 0; i < x.size(); ++i) {
    double expected = 4.0 * x(i) * x(i) - 1.0;
    EXPECT_DOUBLE_EQ(u2(i), expected);
  }
}

TEST(test_chebyshev_u, test_phi_n_max_zero) {
  Eigen::ArrayXd x(3);
  x << -0.5, 0.0, 0.5;

  Eigen::ArrayXXd phi = chebyshev_u_phi(0, x);

  EXPECT_EQ(phi.rows(), 3);
  EXPECT_EQ(phi.cols(), 1);

  // All values should be 1 (U_0 = 1)
  for (Eigen::Index i = 0; i < phi.rows(); ++i) {
    EXPECT_DOUBLE_EQ(phi(i, 0), 1.0);
  }
}

/*
 * Trigonometric identity test: U_n(cos(theta)) = sin((n+1)*theta) / sin(theta)
 * Note: This identity is singular at theta = 0, pi, so we avoid those
 */
TEST(test_chebyshev_u, test_trig_identity) {
  std::default_random_engine gen(2012);
  // Avoid theta near 0 and pi where sin(theta) is near zero
  std::uniform_real_distribution<double> theta_dist(0.1, M_PI - 0.1);
  std::uniform_int_distribution<Eigen::Index> size_dist(10, 50);

  constexpr std::size_t kNumIterations = 100;
  constexpr Eigen::Index kMaxDegree = 10;
  constexpr double kTolerance = 1e-11;

  for (std::size_t iter = 0; iter < kNumIterations; ++iter) {
    Eigen::Index size = size_dist(gen);
    Eigen::ArrayXd theta(size);
    for (Eigen::Index i = 0; i < size; ++i) {
      theta(i) = theta_dist(gen);
    }

    Eigen::ArrayXd x = theta.cos();
    Eigen::ArrayXd sin_theta = theta.sin();

    for (Eigen::Index n = 0; n <= kMaxDegree; ++n) {
      Eigen::ArrayXd un = chebyshev_u(n, x);
      Eigen::ArrayXd expected = ((n + 1) * theta).sin() / sin_theta;

      for (Eigen::Index i = 0; i < size; ++i) {
        EXPECT_NEAR(un(i), expected(i), kTolerance)
            << "U_" << n << "(cos(" << theta(i) << ")) should equal sin("
            << n + 1 << " * " << theta(i) << ") / sin(" << theta(i) << ")";
      }
    }
  }
}

/*
 * Symmetry test: U_n(-x) = (-1)^n * U_n(x)
 */
TEST(test_chebyshev_u, test_symmetry) {
  std::default_random_engine gen(2012);
  std::uniform_real_distribution<double> x_dist(0.0, 1.0);
  std::uniform_int_distribution<Eigen::Index> size_dist(10, 50);

  constexpr std::size_t kNumIterations = 100;
  constexpr Eigen::Index kMaxDegree = 10;
  constexpr double kTolerance = 1e-14;

  for (std::size_t iter = 0; iter < kNumIterations; ++iter) {
    Eigen::Index size = size_dist(gen);
    Eigen::ArrayXd x(size);
    for (Eigen::Index i = 0; i < size; ++i) {
      x(i) = x_dist(gen);
    }

    for (Eigen::Index n = 0; n <= kMaxDegree; ++n) {
      Eigen::ArrayXd un_pos = chebyshev_u(n, x);
      Eigen::ArrayXd un_neg = chebyshev_u(n, -x);

      double sign = (n % 2 == 0) ? 1.0 : -1.0;

      for (Eigen::Index i = 0; i < size; ++i) {
        EXPECT_NEAR(un_neg(i), sign * un_pos(i), kTolerance)
            << "U_" << n << "(-" << x(i) << ") should equal " << sign << " * U_"
            << n << "(" << x(i) << ")";
      }
    }
  }
}

/*
 * Scipy reference values test for U_n
 * Values generated using scipy.special.eval_chebyu
 */
TEST(test_chebyshev_u, test_scipy_reference) {
  struct ReferencePoint {
    Eigen::Index n;
    double x;
    double expected;
  };

  // clang-format off
  const std::vector<ReferencePoint> reference_points = {
      {0, -1.0, 1.000000000000000e+00},
      {0, -0.5, 1.000000000000000e+00},
      {0, 0.0, 1.000000000000000e+00},
      {0, 0.5, 1.000000000000000e+00},
      {0, 1.0, 1.000000000000000e+00},
      {0, 0.123, 1.000000000000000e+00},
      {0, -0.789, 1.000000000000000e+00},
      {0, 0.999, 1.000000000000000e+00},
      {1, -1.0, -2.000000000000000e+00},
      {1, -0.5, -1.000000000000000e+00},
      {1, 0.0, 0.000000000000000e+00},
      {1, 0.5, 1.000000000000000e+00},
      {1, 1.0, 2.000000000000000e+00},
      {1, 0.123, 2.460000000000000e-01},
      {1, -0.789, -1.578000000000000e+00},
      {1, 0.999, 1.998000000000000e+00},
      {2, -1.0, 3.000000000000000e+00},
      {2, -0.5, 0.000000000000000e+00},
      {2, 0.0, -1.000000000000000e+00},
      {2, 0.5, 0.000000000000000e+00},
      {2, 1.0, 3.000000000000000e+00},
      {2, 0.123, -9.394840000000000e-01},
      {2, -0.789, 1.490084000000000e+00},
      {2, 0.999, 2.992004000000000e+00},
      {3, -1.0, -4.000000000000000e+00},
      {3, -0.5, 1.000000000000000e+00},
      {3, 0.0, 0.000000000000000e+00},
      {3, 0.5, -1.000000000000000e+00},
      {3, 1.0, 4.000000000000000e+00},
      {3, 0.123, -4.771130640000000e-01},
      {3, -0.789, -7.733525520000006e-01},
      {3, 0.999, 3.980023992000000e+00},
      {4, -1.0, 5.000000000000000e+00},
      {4, -0.5, -1.000000000000000e+00},
      {4, 0.0, 1.000000000000000e+00},
      {4, 0.5, -1.000000000000000e+00},
      {4, 1.0, 5.000000000000000e+00},
      {4, 0.123, 8.221141862560000e-01},
      {4, -0.789, -2.697336729439994e-01},
      {4, 0.999, 4.960083936016000e+00},
      {5, -1.0, -6.000000000000000e+00},
      {5, -0.5, 0.000000000000000e+00},
      {5, 0.0, 0.000000000000000e+00},
      {5, 0.5, 0.000000000000000e+00},
      {5, 1.0, 6.000000000000000e+00},
      {5, 0.123, 6.793531538189760e-01},
      {5, -0.789, 1.198992287905632e+00},
      {5, 0.999, 5.930223712159968e+00},
      {6, -1.0, 7.000000000000000e+00},
      {6, -0.5, 1.000000000000000e+00},
      {6, 0.0, -1.000000000000000e+00},
      {6, 0.5, 1.000000000000000e+00},
      {6, 1.0, 7.000000000000000e+00},
      {6, 0.123, -6.549933104165319e-01},
      {6, -0.789, -1.622276157371088e+00},
      {6, 0.999, 6.888503040879616e+00},
      {7, -1.0, -8.000000000000000e+00},
      {7, -0.5, -1.000000000000000e+00},
      {7, 0.0, 0.000000000000000e+00},
      {7, 0.5, 1.000000000000000e+00},
      {7, 1.0, 8.000000000000000e+00},
      {7, 0.123, -8.404815081814428e-01},
      {7, -0.789, 1.360959488425945e+00},
      {7, 0.999, 7.833005363517506e+00},
  };
  // clang-format on

  constexpr double kTolerance = 1e-13;

  for (const auto &ref : reference_points) {
    Eigen::ArrayXd x(1);
    x << ref.x;
    Eigen::ArrayXd result = chebyshev_u(ref.n, x);

    EXPECT_NEAR(result(0), ref.expected, kTolerance)
        << "U_" << ref.n << "(" << ref.x << ") mismatch with scipy reference";
  }
}

/*
 * Basis matrix consistency test for U_n:
 * chebyshev_u_phi(n_max, x).col(k) should equal chebyshev_u(k, x)
 */
TEST(test_chebyshev_u, test_phi_consistency) {
  std::default_random_engine gen(2012);
  std::uniform_real_distribution<double> x_dist(-1.0, 1.0);
  std::uniform_int_distribution<Eigen::Index> size_dist(10, 50);

  constexpr std::size_t kNumIterations = 100;
  constexpr Eigen::Index kMaxDegree = 15;
  constexpr double kTolerance = 1e-14;

  for (std::size_t iter = 0; iter < kNumIterations; ++iter) {
    Eigen::Index size = size_dist(gen);
    Eigen::ArrayXd x(size);
    for (Eigen::Index i = 0; i < size; ++i) {
      x(i) = x_dist(gen);
    }

    Eigen::ArrayXXd phi = chebyshev_u_phi(kMaxDegree, x);

    // Check dimensions
    EXPECT_EQ(phi.rows(), size);
    EXPECT_EQ(phi.cols(), kMaxDegree + 1);

    // Check each column matches chebyshev_u
    for (Eigen::Index k = 0; k <= kMaxDegree; ++k) {
      Eigen::ArrayXd uk = chebyshev_u(k, x);
      for (Eigen::Index i = 0; i < size; ++i) {
        EXPECT_NEAR(phi(i, k), uk(i), kTolerance)
            << "phi.col(" << k << ") should equal chebyshev_u(" << k << ", x)";
      }
    }
  }
}

/*
 * Recurrence relation test for U_n:
 * U_n(x) = 2*x*U_{n-1}(x) - U_{n-2}(x) for n >= 2
 */
TEST(test_chebyshev_u, test_recurrence_relation) {
  std::default_random_engine gen(2012);
  std::uniform_real_distribution<double> x_dist(-1.0, 1.0);
  std::uniform_int_distribution<Eigen::Index> size_dist(10, 50);

  constexpr std::size_t kNumIterations = 100;
  constexpr Eigen::Index kMinDegree = 3;
  constexpr Eigen::Index kMaxDegree = 15;
  constexpr double kTolerance = 1e-13;

  for (std::size_t iter = 0; iter < kNumIterations; ++iter) {
    Eigen::Index size = size_dist(gen);
    Eigen::ArrayXd x(size);
    for (Eigen::Index i = 0; i < size; ++i) {
      x(i) = x_dist(gen);
    }

    for (Eigen::Index n = kMinDegree; n <= kMaxDegree; ++n) {
      Eigen::ArrayXd un = chebyshev_u(n, x);
      Eigen::ArrayXd un_minus_1 = chebyshev_u(n - 1, x);
      Eigen::ArrayXd un_minus_2 = chebyshev_u(n - 2, x);

      Eigen::ArrayXd expected = 2.0 * x * un_minus_1 - un_minus_2;

      for (Eigen::Index i = 0; i < size; ++i) {
        EXPECT_NEAR(un(i), expected(i), kTolerance)
            << "U_" << n << "(x) should equal 2*x*U_" << n - 1 << "(x) - U_"
            << n - 2 << "(x)";
      }
    }
  }
}

/*
 * =========================================================================
 * Cross-polynomial identity tests linking T_n and U_n
 * =========================================================================
 */

/*
 * Complex power identity test:
 * For all real a, b: (a + i*b)^n = T_n(a) + i*b*U_{n-1}(a)
 *
 * This identity holds when a^2 + b^2 = 1 (i.e., a + ib is on the unit circle).
 * Writing a = cos(theta), b = sin(theta), we have:
 *   (cos(theta) + i*sin(theta))^n = cos(n*theta) + i*sin(n*theta)
 *   T_n(cos(theta)) = cos(n*theta)
 *   sin(theta)*U_{n-1}(cos(theta)) = sin(theta)*sin(n*theta)/sin(theta)
 *                                   = sin(n*theta)
 *
 * So the identity becomes: cos(n*theta) + i*sin(n*theta) = T_n(a) +
 * i*b*U_{n-1}(a)
 */
TEST(test_chebyshev_identity, test_complex_power) {
  std::default_random_engine gen(2012);
  // Generate theta in (0, pi) to avoid singularities at 0 and pi
  std::uniform_real_distribution<double> theta_dist(0.1, M_PI - 0.1);
  std::uniform_int_distribution<Eigen::Index> size_dist(10, 50);

  constexpr std::size_t kNumIterations = 100;
  constexpr Eigen::Index kMaxDegree = 15;
  constexpr double kTolerance = 1e-12;

  for (std::size_t iter = 0; iter < kNumIterations; ++iter) {
    Eigen::Index size = size_dist(gen);
    Eigen::ArrayXd theta(size);
    for (Eigen::Index i = 0; i < size; ++i) {
      theta(i) = theta_dist(gen);
    }

    Eigen::ArrayXd a = theta.cos(); // a = cos(theta)
    Eigen::ArrayXd b = theta.sin(); // b = sin(theta)

    for (Eigen::Index n = 1; n <= kMaxDegree; ++n) {
      // Compute T_n(a) and U_{n-1}(a)
      Eigen::ArrayXd tn = chebyshev_t(n, a);
      Eigen::ArrayXd un_minus_1 = chebyshev_u(n - 1, a);

      // Expected values from (a + ib)^n = e^{in*theta} = cos(n*theta) +
      // i*sin(n*theta)
      Eigen::ArrayXd expected_real = (n * theta).cos();
      Eigen::ArrayXd expected_imag = (n * theta).sin();

      // Verify: real part T_n(a) = cos(n*theta)
      for (Eigen::Index i = 0; i < size; ++i) {
        EXPECT_NEAR(tn(i), expected_real(i), kTolerance)
            << "Real part: T_" << n << "(cos(" << theta(i)
            << ")) should equal cos(" << n << " * " << theta(i) << ")";
      }

      // Verify: imaginary part b*U_{n-1}(a) = sin(n*theta)
      Eigen::ArrayXd computed_imag = b * un_minus_1;
      for (Eigen::Index i = 0; i < size; ++i) {
        EXPECT_NEAR(computed_imag(i), expected_imag(i), kTolerance)
            << "Imaginary part: sin(" << theta(i) << ") * U_" << n - 1
            << "(cos(" << theta(i) << ")) should equal sin(" << n << " * "
            << theta(i) << ")";
      }
    }
  }
}

/*
 * Edge case for complex power identity: n = 0
 * (a + ib)^0 = 1 = T_0(a) + i*b*U_{-1}(a)
 * This requires U_{-1}(a) = 0 by convention
 * We test this by verifying T_0 = 1 (already covered) and that the identity
 * holds trivially for n=0.
 */
TEST(test_chebyshev_identity, test_complex_power_n_zero) {
  Eigen::ArrayXd a(5);
  a << -0.9, -0.5, 0.0, 0.5, 0.9;

  // T_0(a) should be 1
  Eigen::ArrayXd t0 = chebyshev_t(0, a);
  for (Eigen::Index i = 0; i < a.size(); ++i) {
    EXPECT_DOUBLE_EQ(t0(i), 1.0);
  }
}

} // namespace albatross
