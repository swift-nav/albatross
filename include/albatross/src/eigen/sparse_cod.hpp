/*
 * Copyright (C) 2022 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef ALBATROSS_EIGEN_SPARSE_COD_H
#define ALBATROSS_EIGEN_SPARSE_COD_H

namespace Eigen {

// This struct exists for settings that should be applied to both SPQR
// solves used by the COD object.
struct spqr_config {
  int ordering;
  int nthreads;
  double pivot_threshold;
};

namespace internal {

inline void spqr_setup(SerializableSPQR<SparseMatrix<double>> *spqr,
                       const spqr_config &config) {
  spqr->setSPQROrdering(config.ordering);
  spqr->cholmodCommon()->SPQR_nthreads = config.nthreads;
  spqr->setPivotThreshold(config.pivot_threshold);
}

inline spqr_config make_deficient_solve_config(
    const spqr_config &initial_config) {
  auto config = initial_config;
  config.pivot_threshold = -1;
  return config;
}

}  // namespace internal

class SparseCOD : public SparseSolverBase<SparseCOD> {
  using Base = SparseSolverBase<SparseCOD>;
 public:
  // Our typedefs
  using SparseQR = SerializableSPQR<SparseMatrix<double>>;
  using SparsePermutationMatrix =
    PermutationMatrix<Dynamic, Dynamic, SparseQR::StorageIndex>;

  // Eigen typedefs
  using Scalar = SparseQR::Scalar;
  using RealScalar = SparseQR::RealScalar;
  using MatrixType = SparseQR::MatrixType;
  using StorageIndex = SparseQR::StorageIndex;
  using PermutationType = SparseQR::PermutationType;
  enum { ColsAtCompileTime = Dynamic, MaxColsAtCompileTime = Dynamic };

  SparseCOD() = default;
  explicit SparseCOD(const spqr_config &config) : Base() {
    configure(config);
  }

  explicit SparseCOD(const MatrixType &A) : Base() { compute(A); }

  SparseCOD(const MatrixType &A, const spqr_config &config) : Base() {
    configure(config);
    compute(A);
  }

  void configure(const spqr_config &config) {
    internal::spqr_setup(&spqr1, config);
    internal::spqr_setup(&spqr2, internal::make_deficient_solve_config(config));
    tolerance = config.pivot_threshold;
  }

  void compute(const MatrixType &A) {
    eigen_assert(A.rows() >= A.cols() &&
                 "This decomposition only works on tall problems!");
    spqr1.compute(A);
    corrected_rank = estimate_rank();
    m_isInitialized = true;
    if (rank_deficient()) {
      // After we have decomposed A, we QR-factorize the transpose of
      // the upper-trapezoidal [R S].
      //
      // __NB__: I'm not sure about the efficiency ramifications of
      // transposing a sparse RS factor.  But it is at most cols() *
      // cols(), so hopefully the impact is not too bad.
      spqr2.compute(get_sparse_RS(spqr1).transpose());
    }
  }

  Index cols() const {
    eigen_assert(m_isInitialized &&
                 "The COD should be computed first; call compute()");
    return spqr1.cols();
  }

  Index rows() const {
    eigen_assert(m_isInitialized &&
                 "The COD should be computed first; call compute()");
    return spqr1.rows();
  }

  // Return the SPQR-estimated rank of A.  According to the
  // __IMPORTANT PRECONDITION__ detailed before _solve_impl(), this
  // should be the actual rank of A.
  Index rank() const {
    eigen_assert(m_isInitialized &&
                 "The COD should be computed first; call compute()");
    return corrected_rank;
  }

  // Return the rank-deficiency of the problem, i.e. min(rows(A), cols(A)) -
  // rank(A).  For fully-determined problems, this should be 0.
  Index deficiency() const { return std::min(rows(), cols()) - rank(); }

  bool rank_deficient() const { return deficiency() > 0; }

  ComputationInfo info() const {
    eigen_assert(m_isInitialized &&
                 "The COD should be computed first; call compute()");
    if (spqr1.info() != Success) {
      return spqr1.info();
    }
    if (rank_deficient()) {
      return spqr2.info();
    }
    return Success;
  }

  // L \in \mathbb{R}^{r x r}, r = rank(A)
  //
  // Also referred to as T in [1]
  MatrixType matrixL() const {
    eigen_assert(m_isInitialized &&
                 "The COD should be computed first; call compute()");
    return spqr2.matrixR()
        .topLeftCorner(rank(), rank())
        .triangularView<Upper>()
        .transpose();
  }

  MatrixType matrixRS() const {
    eigen_assert(m_isInitialized &&
                 "The COD should be computed first; call compute()");
    return spqr1.matrixR()
        .topLeftCorner(rank(), cols())
        .triangularView<Upper>();
  }

  VectorXd Rdiag() const {
    return spqr1.matrixR().diagonal().head(rank());
  }

  VectorXd Ldiag() const {
    return matrixL().diagonal().head(rank());
  }

  MatrixXd matrixLDense() const {
    eigen_assert(m_isInitialized &&
                 "The COD should be computed first; call compute()");
    return MatrixXd(matrixL());
  }

  // Solve Ax = b using the /sparse complete orthogonal decomposition/
  // (COD).  b may have multiple columns.
  //
  // Computing the minimum-norm solution to an underdetermined linear
  // system using the SVD is not practical for large systems, and
  // sparsity does not help.  Fortunately we can compute the
  // minimum-norm solution using the QR decomposition, which can be done
  // efficiently by leveraging sparsity.
  //
  // This code computes the COD by using sparse QR solves, more
  // efficiently computing the minimum-norm solution of large, sparse
  // underdetermined linear systems.  The method used is described in
  // [1].
  //
  // __IMPORTANT PRECONDITION__:
  //
  //  - The specified pivot threshold must be low enough that the SPQR
  //    estimate of rank(A) is accurate.  This is unlike the SPQR_COD
  //    algorithm given in [1], which performs subspace iteration to
  //    accurately determine rank(A).  If you are confident that SPQR
  //    will estimate the correct numerical rank for your system, you
  //    may use this code to avoid the additional work of subspace
  //    iteration.
  //
  // Here is a summary of the COD approach to forming the pseudoinverse
  // A+ of A:
  //
  //     A = Q1 ( P2 [L 0; 0 0] Q2^T ) P1^T
  //              ^^^^^^^^^^^^^^^^^^
  //                      R
  //
  //     R^T = Q2 [L^T 0; 0 0] P'2^T
  //
  //     P2 = blkdiag([P'2 I])
  //
  //     A+ = P1 Q2 [L^{-1} 0; 0 0] P2^T Q1^T
  //
  // The only term that must be inverted is the lower-triangular `L`
  // submatrix, and this can be stably and quickly applied to the
  // right-hand side using triangular substitution.
  //
  // [1]
  // https://people.engr.tamu.edu/davis/publications_files/Reliable_Calculation_of_Numerical_Rank_Null_Space_Bases_Pseudoinverse_Solutions_and_Basic_Solutions_using_SuiteSparseQR.pdf
  template <typename Rhs, typename Dest>
  void _solve_impl(const MatrixBase<Rhs> &b, MatrixBase<Dest> &dest) const {
    eigen_assert(m_isInitialized &&
                 "The COD should be computed first; call compute()");
    eigen_assert(b.rows() == rows());
    if (!rank_deficient()) {
      // In the full-rank case, just use the outer QR solver to do the
      // right thing.
      for (Eigen::Index i = 0; i < dest.cols(); ++i) {
        dest.col(i) = spqr1.solve(b.col(i));
      }
      return;
    }

    // std::cerr << "Ranks: SPQR1 = " << spqr1.rank()
    //           << "; SPQR2 = " << spqr2.rank() << std::endl;
    // std::cerr << "RS(" << spqr1.cols() << "):\n"
    //           << Eigen::MatrixXd(get_sparse_RS(spqr1)) << std::endl;
    // std::cerr << "L(" << spqr2.cols() << "):\n" << matrixLDense() << std::endl;

    // Expand P'2 into P2 by appending identity
    //
    // Compute the first orthogonal transform c = P2^T * Q1^T * b
    //
    // Subsitution solve for the top of z, then pad out the bottom with
    // zeros (in the null space)
    MatrixXd z(cols(), b.cols());
    z.topRows(rank()) = matrixL().triangularView<Lower>().solve(
        (expandP2(rows()).transpose() * (spqr1.matrixQ().transpose() * b))
            .topRows(rank()));
    z.bottomRows(deficiency()) = MatrixXd::Zero(deficiency(), b.cols());

    // Compute the second orthogonal transform P1 * Q2 * z
    //
    // __NB__: it's necessary to call the explicit dense matrix
    // constructor here, or Eigen will be confused and give an assert
    // about wrong dimensions.
    //
    // TODO(@peddie) can we use `applyOnTheLeft()`?
    dest = spqr1.colsPermutation() * Eigen::MatrixXd(spqr2.matrixQ() * z);
  }

  // Returns the outer permutation matrix, corresponding to the one
  // returned by SPQR(A).colsPermutation().
  SparsePermutationMatrix colsPermutation() const {
    eigen_assert(m_isInitialized &&
                 "The COD should be computed first; call compute()");
    return spqr1.colsPermutation();
  }

  PermutationMatrix<Dynamic, Dynamic> colsPermutationDense() const {
    PermutationMatrix<Dynamic, Dynamic> Pout;
    Pout.indices() = colsPermutation().indices().cast<int>();
    return Pout;
  }

  // Compute R^-T P1^T rhs
  //
  // In the rank-deficient case:
  //
  //   R^T P2 = Q2 [L 0; 0 0]
  //
  //   R^-T = P2 [L^-1 0; 0 0] Q2^T
  //
  // so we want P2 z, where
  //
  //   z = [L^-1 0; 0 0] Q2^T P1^T rhs
  MatrixXd Rsolve(const MatrixXd &rhs) {
    if (!rank_deficient()) {
      return spqr1.matrixR()
          .topLeftCorner(rank(), rank())
          .triangularView<Upper>()
          .solve(spqr1.colsPermutation() * rhs);
    }

    Eigen::MatrixXd z(cols(), rhs.cols());
    z.topRows(rank()) = matrixL().triangularView<Lower>().solve(
        spqr2.matrixQ().transpose() *
        Eigen::MatrixXd(spqr1.colsPermutation() * rhs));
    z.bottomRows(deficiency()) =
        Eigen::MatrixXd::Zero(deficiency(), rhs.cols());
    return expandP2(cols()) * z;
  }

  template <class Archive>
  void save(Archive &ar, const std::uint32_t version ALBATROSS_UNUSED) const;

  template <class Archive>
  void load(Archive &ar, const std::uint32_t version ALBATROSS_UNUSED);

 private:
  Index estimate_rank() const {
    double max_eig = spqr1.matrixR().diagonal().array().abs().maxCoeff();
    Eigen::Index below = 0;
    for (Eigen::Index i = 0; i < spqr1.rank(); ++i) {
      const double c = fabs(spqr1.matrixR().coeff(i, i));
      if (max_eig * tolerance > c) {
        fprintf(stderr,
                "SPQR overestimate: max_eig * tolerance = %.3e, c = %.3e, i = "
                "%ld, cols = %ld, SPQR1 rank = %ld; %ld tail: ",
                max_eig * tolerance, c, i, spqr1.cols(), spqr1.rank(),
                spqr1.rank() - i);
        std::cerr << spqr1.matrixR()
                         .diagonal()
                         .head(spqr1.rank())
                         .tail(spqr1.rank() - i)
                         .transpose()
                  << std::endl;
        below++;
      }
    }
    return spqr1.rank() - below;
  }

  // Expand P'2 (the permutation from the inner QR solve) by appending
  // identity
  inline SparsePermutationMatrix expandP2(Index size) const {
    assert(size >= spqr2.cols());
    SparsePermutationMatrix P2(size);
    P2.setIdentity();
    P2.indices().head(spqr2.cols()) = spqr2.colsPermutation().indices();
    return P2;
  }

  inline MatrixType get_sparse_RS(const SparseQR &spqr) const {
    return spqr.matrixR().topLeftCorner(rank(), cols()).triangularView<Upper>();
  }

  // We must hold onto these entire objects, because we cannot use the
  // `Q` factors after they are destructed.  Using SPQR, you can only
  // get an expression for `Q` and multiply by it; you cannot convert
  // it to a sparse or dense matrix.
  SparseQR spqr1;
  SparseQR spqr2;
  double tolerance = 1e-14;
  Eigen::Index corrected_rank;
};

}  // namespace Eigen

#endif  // ALBATROSS_EIGEN_SPARSE_COD_H