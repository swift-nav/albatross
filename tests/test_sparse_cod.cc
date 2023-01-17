#include <Eigen/QR>
#include <Eigen/SPQRSupport>
#include <Eigen/SVD>
#include <Eigen/SparseCore>
#include <albatross/Core>
#include <albatross/Indexing>
#include <albatross/src/eigen/serializable_spqr.hpp>
#include <albatross/src/eigen/sparse_cod.hpp>
#include <albatross/utils/RandomUtils>

#include <Eigen/src/Core/util/Constants.h>
#include <gtest/gtest.h>

TEST(SparseCOD, ConstructDestruct) {
  Eigen::SparseCOD cod;
  std::cerr << &cod << std::endl;
}

constexpr double kCODTolerance = 1e-15;

static const Eigen::spqr_config spqr_config{SPQR_ORDERING_COLAMD, 2,
                                            kCODTolerance};

constexpr double kVectorTolerance = 1e-9;

template <typename TestType, typename RefType>
static double relativeNorm(const TestType &test, const RefType &ref) {
  const double refnorm = ref.norm();
  const double diffnorm = (test - ref).norm();
  const double result = diffnorm / refnorm;
  if (isnan(diffnorm)) {
    std::cerr << "NaN difference norm: " << test - ref << std::endl;
    return result;
  }
  if (isnan(refnorm)) {
    std::cerr << "NaN ref norm: " << ref << std::endl;
    return result;
  }
  if (refnorm < kCODTolerance) {
    // std::cerr << "Tiny ref norm (" << refnorm << "): " << ref << std::endl;
    return 0;
  }
  if (isnan(result)) {
    // std::cerr << "NaN ratio: " << test - ref << std::endl
    //           << std::endl
    //           << ref << std::endl;
    return 0;
  }
  return result;
}

TEST(SparseCOD, SimpleVector) {
  Eigen::MatrixXd Adense(3, 3);
  Adense << 1., 4, 2, 2, 2, 9.1, 4, 8, 111.2;
  Eigen::VectorXd b(3);
  b << 2, 4, 99;
  Eigen::SparseCOD cod(spqr_config);
  cod.compute(Adense.sparseView());
  EXPECT_EQ(cod.info(), Eigen::Success);
  const Eigen::VectorXd x = cod.solve(b);
  EXPECT_EQ(cod.info(), Eigen::Success);

  auto dense_cod =
      Adense.completeOrthogonalDecomposition();
  dense_cod.setThreshold(kCODTolerance);
  const Eigen::VectorXd x_dense = dense_cod.solve(b);

  EXPECT_LT(relativeNorm(x, x_dense), kVectorTolerance);
}

constexpr double kMatrixTolerance = 1e-10;

TEST(SparseCOD, SimpleMatrix) {
  Eigen::MatrixXd Adense(3, 3);
  Adense << 1., 4, 2, 2, 2, 9.1, 4, 8, 111.2;
  Eigen::MatrixXd b(3, 2);
  b << 2, 1, 4, 19, 99, 101;
  Eigen::SparseCOD cod(spqr_config);
  cod.compute(Adense.sparseView());
  EXPECT_EQ(cod.info(), Eigen::Success);
  const Eigen::MatrixXd x = cod.solve(b);
  EXPECT_EQ(cod.info(), Eigen::Success);

  // const auto dense_cod = Adense.completeOrthogonalDecomposition();
  // Eigen::MatrixXd x_dense(b.rows(), b.cols());
  // for (Eigen::Index i = 0; i < b.cols(); ++i) {
  //   x_dense.col(i) = dense_cod.solve(b.col(i));
  // }
  const auto svd = Adense.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
  const Eigen::MatrixXd x_dense = svd.solve(b);

  EXPECT_LT(relativeNorm(x, x_dense), kMatrixTolerance);
}

template <typename Distribution>
Eigen::Index generate_size(std::default_random_engine &gen, Distribution &&dist,
                           Eigen::Index max_size, Eigen::Index min_size = 1) {
  return std::max(min_size, std::min(max_size, dist(gen)));
}

template <typename Distribution>
double generate_fill(std::default_random_engine &gen, Distribution &&dist) {
  return std::max(0.0, std::min(1.0, dist(gen)));
}

// constexpr Eigen::Index kMaxSize = 10000;
// constexpr double kMeanSize = 200;
constexpr Eigen::Index kMaxSize = 100;
constexpr double kMeanSize = 20;

Eigen::SparseMatrix<double> random_sparse_matrix(
    std::default_random_engine &gen, Eigen::Index max_rows = kMaxSize,
    Eigen::Index max_cols = kMaxSize, double mean_rows = kMeanSize,
    double mean_cols = kMeanSize, double mean_fill = 0.2) {
  const auto rows = generate_size(
      gen, std::poisson_distribution<Eigen::Index>{mean_rows}, max_rows);
  // Tall matrices only?
  const auto cols =
      generate_size(gen, std::poisson_distribution<Eigen::Index>{mean_cols},
                    std::min(rows, max_cols));
  // max_cols);
  const auto fill =
      generate_fill(gen, std::gamma_distribution<double>{2, mean_fill / 2});
  return albatross::random_sparse_matrix<double>(rows, cols, fill, gen);
}

constexpr Eigen::Index kIterations = 2000;

// constexpr Eigen::Index kIterations = 100;

TEST(SparseCOD, RandomMatrix) {
  std::seed_seq seed{22};
  std::default_random_engine gen{seed};

  for (Eigen::Index iter = 0; iter < kIterations; ++iter) {
    Eigen::SparseMatrix<double> A = random_sparse_matrix(gen);
    const Eigen::Index bcols = generate_size(
        gen, std::poisson_distribution<Eigen::Index>{kMeanSize}, kMaxSize);
    Eigen::MatrixXd b = Eigen::MatrixXd::Random(A.rows(), bcols);
    Eigen::SparseCOD cod(spqr_config);
    cod.compute(A);
    EXPECT_EQ(cod.info(), Eigen::Success);
    const Eigen::MatrixXd x = cod.solve(b);
    EXPECT_EQ(cod.info(), Eigen::Success);

    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> dense_cod;

    // auto dense_cod = Eigen::MatrixXd(A).completeOrthogonalDecomposition();
    dense_cod.setThreshold(kCODTolerance);
    dense_cod.compute(A);
    Eigen::MatrixXd x_dense(A.cols(), b.cols());
    for (Eigen::Index i = 0; i < b.cols(); ++i) {
      x_dense.col(i) = dense_cod.solve(b.col(i));
    }

    Eigen::JacobiSVD<Eigen::MatrixXd, Eigen::ColPivHouseholderQRPreconditioner>
        svd;
    svd.setThreshold(kCODTolerance);
    svd.compute(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    // const auto svd =
    //     Eigen::MatrixXd(A).jacobiSvd(Eigen::ComputeThinU |
    //     Eigen::ComputeThinV);
    const Eigen::MatrixXd x_svd = svd.solve(b);
    const double real_condition =
        svd.singularValues().maxCoeff() / svd.singularValues().minCoeff();
    const double nz_condition =
        svd.rank() == 0 ? std::numeric_limits<double>::infinity()
                        : svd.singularValues().head(svd.rank()).maxCoeff() /
                              svd.singularValues().head(svd.rank()).minCoeff();
    EXPECT_EQ(cod.rank(), dense_cod.rank());
    EXPECT_EQ(dense_cod.rank(), svd.rank());

    if (relativeNorm(x, x_dense) >= kMatrixTolerance) {
      std::cerr << "Dense R diag:\n" << dense_cod.matrixRS().diagonal().transpose() << std::endl;
      std::cerr << "Sparse R diag:\n" << cod.Rdiag().transpose() << std::endl;
      std::cerr << "Sparse R eig ratio: "
                << cod.Rdiag().array().abs().minCoeff() /
                       cod.Rdiag().array().abs().maxCoeff()
                << std::endl;
      std::cerr << "Dense T:\n"
                << Eigen::MatrixXd(
                       dense_cod.matrixT()
                           .topLeftCorner(dense_cod.rank(), dense_cod.rank())
                           .template triangularView<Eigen::Upper>())
                << std::endl;
      std::cerr << "Sparse T:\n"
                << Eigen::MatrixXd(cod.matrixL().transpose()) << std::endl;
      std::cerr << "Sparse T diag:\n" << cod.Ldiag().transpose() << std::endl;
      std::cerr << "SVs:\n" << svd.singularValues().transpose() << std::endl;
      std::printf("A(%ldx%ld) cond. %.2e (%.2e) SVD rank %ld sparse %ld sparse_relnorm %.3e\n",
                  A.rows(), A.cols(), real_condition, nz_condition, svd.rank(),
                  cod.rank(), relativeNorm(x, x_dense));
      std::cerr << "A(" << A.rows() << "x" << A.cols() << "; rank "
                << svd.rank() << "):\n"
                << Eigen::MatrixXd(A) << std::endl;
      // std::cerr << "b(" << b.rows() << "x" << b.cols() << "):\n"
      //           << b << std::endl;
      // std::cerr << "x(" << x.rows() << "x" << x.cols() << "): " << x
      //           << std::endl;
      // std::cerr << "x_dense(" << x_dense.rows() << "x" << x_dense.cols()
      //           << "): " << x_dense << std::endl;
      // std::cerr << "x - x_dense (norm = " << relativeNorm(x, x_dense) << "):\n" << x - x_dense
      //           << std::endl;
      exit(1);


      // std::cerr << "x - x_svd (norm = " << relativeNorm(x, x_svd) << "):\n"
      //           << x - x_svd << std::endl;
      // std::cerr << "x_dense - x_svd (norm = " << relativeNorm(x_dense, x_svd)
      //           << "):\n"
      //           << x_dense - x_svd << std::endl;
    }

    EXPECT_LT(relativeNorm(x, x_dense), kMatrixTolerance);
  }
}
