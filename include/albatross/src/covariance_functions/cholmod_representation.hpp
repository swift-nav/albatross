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

#ifndef ALBATROSS_COVARIANCE_FUNCTION_CHOLMOD_REPRESENTATION_HPP_
#define ALBATROSS_COVARIANCE_FUNCTION_CHOLMOD_REPRESENTATION_HPP_

namespace albatross {

/*
 * CholmodCovariance is a sparse-Cholesky-backed CovarianceRepresentation
 * (see gp.hpp:42-77). Templated on one of the serializable CHOLMOD solver
 * wrappers (SerializableCholmodSupernodalLLT, SerializableCholmodSimplicialLLT,
 * SerializableCholmodSimplicialLDLT). Retains the input matrix as a
 * SparseMatrix<double>; all operations (solve, sqrt_solve, inverse_diagonal,
 * inverse_blocks, log_determinant) route through CHOLMOD and stay
 * sparsity-aware in the factor.
 *
 * The MatrixXd constructor goes through Eigen::sparseView(): dense inputs
 * with many exact zeros (e.g. compactly-supported kernels) produce a truly
 * sparse factor and realise the expected speedup over dense LDLT. Fully
 * dense inputs will produce a fully dense factor, at which point the
 * sparse backend is not expected to be faster than the dense path -- but
 * predictions and derived quantities still match the dense reference.
 */
template <typename Solver> struct CholmodCovariance {
  using Scalar = double;
  using SparseMatrixType = Eigen::SparseMatrix<double>;

  CholmodCovariance() = default;

  // unique_ptr gives us move semantics without making the CHOLMOD solver
  // itself movable -- Eigen's CholmodBase destructor frees the factor, so
  // its implicit copy/move are deleted. Fit<GPFit<...>>::Fit(..., MatrixXd)
  // (gp.hpp:61) move-assigns from a temporary, which requires the
  // representation to be movable.
  explicit CholmodCovariance(
      const Eigen::MatrixXd &cov, double prune_reference = 1.0,
      double prune_epsilon = Eigen::NumTraits<double>::dummy_precision())
      : sparse_A_(cov.sparseView(prune_reference, prune_epsilon)),
        solver_(std::make_unique<Solver>()) {
    sparse_A_.makeCompressed();
    solver_->compute(sparse_A_);
    ALBATROSS_ASSERT(solver_->info() == Eigen::Success);
  }

  explicit CholmodCovariance(SparseMatrixType A)
      : sparse_A_(std::move(A)), solver_(std::make_unique<Solver>()) {
    sparse_A_.makeCompressed();
    solver_->compute(sparse_A_);
    ALBATROSS_ASSERT(solver_->info() == Eigen::Success);
  }

  // Value semantics: CholmodCovariance is a "solved covariance matrix"
  // and should behave like one.  Eigen's CholmodBase is non-copyable
  // (its destructor frees the factor), but sparse_A_ retains everything
  // needed to reproduce the factor -- so on copy we allocate a fresh
  // Solver and re-run compute().  This is O(factorization) per copy but
  // keeps the representation usable in frameworks that store Fit in a
  // variant / container requiring copyability (e.g. Ransac's FitModel
  // slot).
  CholmodCovariance(const CholmodCovariance &other)
      : sparse_A_(other.sparse_A_), solver_() {
    if (other.solver_) {
      solver_ = std::make_unique<Solver>();
      solver_->compute(sparse_A_);
      ALBATROSS_ASSERT(solver_->info() == Eigen::Success);
    }
  }

  CholmodCovariance &operator=(const CholmodCovariance &other) {
    if (this != &other) {
      sparse_A_ = other.sparse_A_;
      if (other.solver_) {
        solver_ = std::make_unique<Solver>();
        solver_->compute(sparse_A_);
        ALBATROSS_ASSERT(solver_->info() == Eigen::Success);
      } else {
        solver_.reset();
      }
    }
    return *this;
  }

  CholmodCovariance(CholmodCovariance &&) noexcept = default;
  CholmodCovariance &operator=(CholmodCovariance &&) noexcept = default;

  // ---- CovarianceRepresentation (duck-typed) interface ----

  Eigen::MatrixXd solve(const Eigen::MatrixXd &rhs) const {
    return solver_->solve(rhs);
  }

  Eigen::Index rows() const { return solver_ ? solver_->rows() : 0; }
  Eigen::Index cols() const { return solver_ ? solver_->cols() : 0; }

  bool operator==(const CholmodCovariance &rhs) const {
    if (rows() != rhs.rows() || cols() != rhs.cols()) {
      return false;
    }
    if (rows() == 0) {
      return true;
    }
    // Functional equivalence: the factor storage for supernodal LLT is not
    // byte-stable (alignment / padding), so compare via solve on a
    // deterministic RHS. Matches how LDLT is compared in test_serialize.cc.
    const Eigen::VectorXd probe = Eigen::VectorXd::Ones(rows());
    return solver_->solve(probe) == rhs.solver_->solve(probe);
  }

  // ---- Extended interface matching SerializableLDLT ----

  double log_determinant() const { return solver_->logDeterminant(); }

  bool is_positive_definite() const {
    return solver_ && solver_->info() == Eigen::Success;
  }

  // D^{-1/2} L^{-1} P * rhs -- matches SerializableLDLT::sqrt_solve
  // (see eigen/serializable_ldlt.hpp:100-104). For LLT factors D=I is
  // implicit so the D-scale step is skipped.
  template <class Rhs> Eigen::MatrixXd sqrt_solve(const Rhs &rhs) const {
    Eigen::MatrixXd y = solve_with_sys(CHOLMOD_P, Eigen::MatrixXd(rhs));
    y = solve_with_sys(CHOLMOD_L, y);
    if (!solver_->factorPtr()->is_ll) {
      y = factor_D_inv_sqrt().asDiagonal() * y;
    }
    return y;
  }

  // P^T L^{-T} D^{-1/2} * rhs -- matches
  // SerializableLDLT::sqrt_transpose_solve
  // (eigen/serializable_ldlt.hpp:116-121).
  template <class Rhs>
  Eigen::MatrixXd sqrt_transpose_solve(const Rhs &rhs) const {
    Eigen::MatrixXd y = Eigen::MatrixXd(rhs);
    if (!solver_->factorPtr()->is_ll) {
      y = factor_D_inv_sqrt().asDiagonal() * y;
    }
    y = solve_with_sys(CHOLMOD_Lt, y);
    y = solve_with_sys(CHOLMOD_Pt, y);
    return y;
  }

  // Diagonal of A^{-1} by n column solves. Each solve is O(nnz_L) when the
  // factor is sparse, so this costs O(n * nnz_L) rather than O(n^3) -- a
  // real win for sparse problems. For a fully-dense factor this is
  // comparable to the dense LDLT path.
  Eigen::VectorXd inverse_diagonal(ThreadPool * /* pool */ = nullptr) const {
    const Eigen::Index n = rows();
    Eigen::VectorXd diag(n);
    // CHOLMOD workspace lives on m_cholmod and is not thread-safe; run
    // this serially regardless of the pool argument.
    for (Eigen::Index i = 0; i < n; ++i) {
      Eigen::VectorXd e_i = Eigen::VectorXd::Zero(n);
      e_i(i) = 1.0;
      const Eigen::VectorXd y = solver_->solve(e_i);
      diag(i) = y(i);
    }
    return diag;
  }

  // Dense (A^{-1})_{B,B} blocks, built per-block via sparse-aware solves.
  // Matches the signature of SerializableLDLT::inverse_blocks
  // (eigen/serializable_ldlt.hpp:132-170).
  std::vector<Eigen::MatrixXd>
  inverse_blocks(const std::vector<std::vector<std::size_t>> &blocks,
                 ThreadPool * /* pool */ = nullptr) const {
    const Eigen::Index n = rows();
    // CHOLMOD workspace is not thread-safe; run serially.
    return albatross::apply(
        blocks,
        [this, n](const std::vector<std::size_t> &idx) -> Eigen::MatrixXd {
          const Eigen::Index b = cast::to_index(idx.size());
          Eigen::MatrixXd E = Eigen::MatrixXd::Zero(n, b);
          for (Eigen::Index j = 0; j < b; ++j) {
            E(cast::to_index(idx[cast::to_size(j)]), j) = 1.0;
          }
          const Eigen::MatrixXd Y = solver_->solve(E);
          Eigen::MatrixXd out(b, b);
          for (Eigen::Index i = 0; i < b; ++i) {
            const Eigen::Index row_i = cast::to_index(idx[cast::to_size(i)]);
            for (Eigen::Index j = 0; j < b; ++j) {
              out(i, j) = Y(row_i, j);
            }
          }
          return out;
        });
  }

  SparseMatrixType sparse_A_;
  std::unique_ptr<Solver> solver_;

private:
  // Thin wrapper around cholmod_solve for a specific sys code. Makes a
  // mutable copy of the RHS since viewAsCholmod requires non-const data.
  Eigen::MatrixXd solve_with_sys(int sys, const Eigen::MatrixXd &rhs) const {
    Eigen::MatrixXd b_copy = rhs;
    cholmod_dense b_cd = Eigen::viewAsCholmod(b_copy);
    cholmod_common *cc = solver_->cholmodCommonPtr();
    cholmod_factor *L = solver_->factorPtr();
    cholmod_dense *x_cd = cholmod_solve(sys, L, &b_cd, cc);
    ALBATROSS_ASSERT(x_cd != nullptr);
    Eigen::MatrixXd out = Eigen::Map<Eigen::MatrixXd>(
        reinterpret_cast<double *>(x_cd->x), cast::to_index(x_cd->nrow),
        cast::to_index(x_cd->ncol));
    cholmod_free_dense(&x_cd, cc);
    return out;
  }

  // Read D^{-1/2} out of a simplicial LDLT factor. For a simplicial LDL'
  // stored in CSC form, CHOLMOD puts D on the diagonal of the factor
  // array, so D[k] = x[p[k]] -- the same walk Eigen uses in
  // CholmodSupport.h's logDeterminant simplicial branch (lines 388-391).
  // Only valid for LDLT; LLT callers must not hit this path.
  Eigen::VectorXd factor_D_inv_sqrt() const {
    const cholmod_factor *L = solver_->factorPtr();
    ALBATROSS_ASSERT(!L->is_ll && "factor_D_inv_sqrt is for LDLT only");
    ALBATROSS_ASSERT(!L->is_super && "LDLT factors are always simplicial");
    ALBATROSS_ASSERT(L->itype == CHOLMOD_INT);
    const int *p = static_cast<const int *>(L->p);
    const double *x = static_cast<const double *>(L->x);
    const Eigen::Index n = cast::to_index(L->n);
    Eigen::VectorXd dinv(n);
    for (Eigen::Index k = 0; k < n; ++k) {
      const double d = x[p[k]];
      dinv(k) = d > 0. ? 1. / std::sqrt(d) : 0.;
    }
    return dinv;
  }
};

} // namespace albatross

#endif /* ALBATROSS_COVARIANCE_FUNCTION_CHOLMOD_REPRESENTATION_HPP_ */
