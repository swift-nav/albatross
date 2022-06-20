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

namespace internal {

// This struct exists for settings that should be applied to both SPQR
// solves used by the COD object.
struct spqr_config {
  int ordering;
  int nthreads;
  double pivot_threshold;
};

inline void spqr_setup(SPQR<SparseMatrix<double>> *spqr,
                       const spqr_config &config) {
  spqr->setSPQROrdering(config.ordering);
  spqr->cholmodCommon()->SPQR_nthreads = config.nthreads;
  spqr->setPivotThreshold(config.pivot_threshold);
}

inline spqr_config make_deficient_solve_config(
    const spqr_config &initial_config) {
  auto config = initial_config;
  config.pivot_threshold = 0;
  return config;
}

}  // namespace internal

class SparseCOD : public SparseSolverBase<SparseCOD> {
 public:
  // Our typedefs
  using SparseQR = SPQR<SparseMatrix<double>>;
  using SparsePermutationMatrix =
      PermutationMatrix<Dynamic, Dynamic, SparseQR::StorageIndex>;

  // Eigen typedefs
  using Scalar = SparseQR::Scalar;
  using RealScalar = SparseQR::RealScalar;
  using StorageIndex = SparseQR::StorageIndex;
  using MatrixType = SparseQR::MatrixType;
  using PermutationType = SparseQR::PermutationType;

  SparseCOD() = default;
  explicit SparseCOD(const internal::spqr_config &config)
      : m_isInitialized{false} {
    configure(config);
  }

  explicit SparseCOD(const MatrixType &A) : m_isInitialized{true} {
    compute(A);
  }

  SparseCOD(const MatrixType &A, const internal::spqr_config &config)
      : m_isInitialized{true} {
    configure(config);
    compute(A);
  }

  void configure(const internal::spqr_config &config) {
    internal::spqr_setup(&spqr1, config);
    internal::spqr_setup(&spqr2, internal::make_deficient_solve_config(config));
  }

  void compute(const MatrixType &A) {
    spqr1.compute(A);
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
    return spqr1.rank();
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
        .topLeftCorner(spqr1.rank(), spqr1.rank())
        .triangularView<Upper>()
        .transpose();
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
      spqr1._solve_impl(b, dest);
      return;
    }

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
    dest = spqr1.colsPermutation() * MatrixXd(spqr2.matrixQ() * z);
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

 private:
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
    return spqr.matrixR()
        .topLeftCorner(spqr.rank(), spqr.cols())
        .triangularView<Upper>();
  }

  // We must hold onto these entire objects, because we cannot use the
  // `Q` factors after they are destructed.  Using SPQR, you can only
  // get an expression for `Q` and multiply by it; you cannot convert
  // it to a sparse or dense matrix.
  SparseQR spqr1;
  SparseQR spqr2;
  bool m_isInitialized;
};

}  // namespace Eigen

#endif // ALBATROSS_EIGEN_SPARSE_COD_H