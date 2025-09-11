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

namespace Eigen {

// A pivoted, partial Cholesky decomposition.
//
// This is meant for use as a preconditioner for symmetric,
// positive-definite problems.  The algorithm is described in:
//
//     On the Low-rank Approximation by the Pivoted Cholesky Decomposition
//     H. Harbrecht, M. Peters and R. Schneider
//     http://www.dfg-spp1324.de/download/preprints/preprint076.pdf
//
// In this implementation, we make the natural choice to avoid storing
// a copy of the input matrix; we only copy its diagonal for use
// during the decomposition.  The L matrix already stores the sequence
// of updates to each relevant column of A, so these updates are
// applied on demand as each pivot column is selected.
template <typename Scalar> class PartialCholesky {
public:
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic>;
  using StorageIndex = typename MatrixType::StorageIndex;
  using RealScalar = typename NumTraits<Scalar>::Real;

  using PermutationType = PermutationMatrix<Dynamic, Dynamic, StorageIndex>;

  static constexpr const Index cDefaultOrder = 22;

  static constexpr const double cDefaultNugget = 1.e-2;

  PartialCholesky() {}

  explicit PartialCholesky(const MatrixType &A)
      : m_rows{A.rows()}, m_cols{A.cols()} {
    compute(A);
  }

  explicit PartialCholesky(Index order) : m_order{order} {}

  PartialCholesky(Index order, RealScalar nugget)
      : m_order{order}, m_nugget{nugget} {}

  PartialCholesky(const MatrixType &A, Index order, RealScalar nugget)
      : m_rows{A.rows()}, m_cols{A.cols()}, m_order{order}, m_nugget{nugget} {
    compute(A);
  }

  inline Index order() const { return m_order; }

  PartialCholesky &setOrder(Index order) {
    m_order = order;
    return *this;
  }

  inline double nugget() const { return m_nugget; }

  PartialCholesky &setNugget(double nugget) {
    m_nugget = nugget;
    return *this;
  }

  // Unlike the normal dense LDLT and friends, we do not want to
  // copy the potentially quite large A.  Unlike the CG solver, we
  // do not want to retain any reference to A's data once we have
  // finished this function.
  PartialCholesky &compute(const MatrixType &A) {
    m_rows = A.rows();
    m_cols = A.cols();
    PermutationType transpositions(rows());
    transpositions.setIdentity();
    const Index max_rank = std::min(order(), rows());
    MatrixType L{MatrixType::Zero(rows(), max_rank)};

    // A's diagonal; needs to keep getting pivoted, shifted and
    // searched at each step.  We track this separately from the rest
    // of A because we have to search each time, and for the relevant
    // off-diagonal columns of A, we only apply the relevant updates
    // we strictly need.
    VectorXd diag{A.diagonal()};

    const auto calc_error = [&diag](Index k) {
      return diag.tail(diag.size() - k).array().sum();
    };

    RealScalar error = calc_error(0);

    for (Index k = 0; k < max_rank; ++k) {
      Index max_index;
      diag.tail(diag.size() - k).maxCoeff(&max_index);
      max_index += k;
      // std::cout << "step " << k << ": found max element " << diag(max_index)
      //           << " at position " << max_index - k << " in "
      //           << diag.tail(diag.size() - k).transpose() << std::endl;

      if (max_index != k) {
        transpositions.applyTranspositionOnTheRight(k, max_index);
        std::swap(diag(k), diag(max_index));
        L.row(k).swap(L.row(max_index));
        // std::cout << max_index << " <-> " << k << std::endl;
      }

      if (diag(k) <= 0.) {
        m_info = InvalidInput;
        return *this;
      }

      const RealScalar alpha = std::sqrt(diag(k));

      L(k, k) = alpha;

      const Index tail_size = rows() - k - 1;
      if (tail_size < 1) {
        break;
      }

      // Everything below here should be ordered appropriately -- we
      // pivot `diag` and the rows of `L` every time we do an update
      VectorXd b =
          ((transpositions.transpose() * A.col(transpositions.indices()(k))))
              .tail(tail_size);

      // I couldn't find this derived anywhere -- basically what
      // happens here is that for the lower-right submatrix below
      // diagonal element k of the pivoted input matrix, we have to
      // update it with b_i b_i^t / a_i = l_i l_i^t where a_i =
      // alpha_i^2 = diagonal element k, _for each preceding pivot
      // column i_ below the current one.  Of course we are modifying
      // the whole input matrix in place, and we only care about the
      // relevant column below the pivot we just chose.  The
      // successive updates to this column A_k,k+1:n are l_i * l_i[k]
      // -- the columns of the bottom-left submatrix of L below k,
      // each column scaled by the element just above it (in row k)
      // respectively.
      for (Index i = 0; i < k; ++i) {
        b -= L.col(i).tail(tail_size) * L(k, i);
      }

      L.col(k).tail(tail_size) = b / alpha;
      diag.tail(tail_size) -=
          L.col(k).tail(tail_size).array().square().matrix();

      for (Index i = 0; i < k; ++i) {
        assert(L(i, i) >= L(i + 1, i + 1) && "failure in ordering invariant!");
      }

      assert(calc_error(k + 1) < error && "failure in convergence criterion!");
      error = calc_error(k + 1);

      // std::cout << "step " << k
      //           << ": error = " << diag.tail(diag.size() - k).array().sum()
      //           << std::endl;
    }

    m_error = diag.tail(diag.size() - max_rank).array().sum();

    // m_nugget = std::sqrt(A.diagonal().minCoeff());

    m_transpositions = transpositions;

    m_L = L;

    // We decompose before returning to save time on repeated solves.
    //
    // Arguably we could pre-apply the outer terms of Woodbury, but
    // computing that full matrix would significantly increase our
    // storage requirements in the typical case where k << n.
    m_decomp = LDLT<MatrixXd>(MatrixXd::Identity(L.cols(), L.cols()) +
                              1 / (m_nugget * m_nugget) * L.transpose() * L);

    MatrixXd Ltall(L.rows() + max_rank, L.cols());
    Ltall.topRows(L.rows()) = transpositions * L;
    Ltall.bottomRows(max_rank) =
        MatrixXd::Identity(max_rank, max_rank) * m_nugget;

    // std::cout << "Ltall (" << Ltall.rows() << "x" << Ltall.cols() << "):\n"
    //           << Ltall << std::endl;
    m_qr = HouseholderQR<MatrixXd>(Ltall);
    MatrixXd thin_Q =
        m_qr.householderQ() * MatrixXd::Identity(m_qr.rows(), m_L.cols());
    // std::cout << "thin_Q (" << thin_Q.rows() << "x" << thin_Q.cols() <<
    // "):\n"
    //           << thin_Q << std::endl;
    m_Q = thin_Q.topRows(rows());
    // std::cout << "m_Q (" << m_Q.rows() << "x" << m_Q.cols() << "):\n"
    //           << m_Q << std::endl;

    m_info = Success;
    return *this;
  }

  template <typename Rhs>
  Matrix<Scalar, Dynamic, Rhs::ColsAtCompileTime> solve(const Rhs &b) const {
    assert(finished() &&
           "Please don't call 'solve()' on an unintialised decomposition!");
    const double n2 = m_nugget * m_nugget;

    // std::cout << "Q^T b:\n" << MatrixXd(m_Q.transpose() * b) << std::endl;
    // std::cout << "Q Q^T b:\n" << MatrixXd(m_Q * (m_Q.transpose() * b)) <<
    // std::endl; std::cout << "b - Q Q^T b:\n" << MatrixXd(b - m_Q *
    // (m_Q.transpose() * b)) << std::endl;

    Matrix<Scalar, Dynamic, Rhs::ColsAtCompileTime> ret =
        1 / n2 * (b - m_Q * (m_Q.transpose() * b));
    // Matrix<Scalar, Dynamic, Rhs::ColsAtCompileTime> ret =
    //     1 / n2 *
    //     (b - (m_transpositions * m_L *
    //           m_decomp.solve(m_L.transpose() * m_transpositions.transpose() *
    //                          b / n2)));
    return ret;
  }

  LDLT<MatrixXd> direct_solve() const {
    assert(finished() && "Please don't call 'direct_solve()' on an "
                         "uninitialised decomposition!");
    return LDLT<MatrixXd>(permutationsP() * matrixL() * matrixL().transpose() *
                              permutationsP().transpose() +
                          MatrixXd::Identity(rows(), cols()) * nugget() *
                              nugget());
  }

  MatrixXd matrixL() const {
    assert(finished() &&
           "Please don't call 'matrixL()' on an uninitialised decomposition!");

    return m_L;
  }

  PermutationType permutationsP() const {
    assert(finished() && "Please don't call 'permutationsP()' on an "
                         "uninitialised decomposition!");

    return m_transpositions;
  }

  inline bool finished() const {
    return rows() > 0 && cols() > 0 & info() == Success;
  }

  inline MatrixType reconstructedMatrix() const {
    assert(finished() && "Please don't call 'reconstructedMatrix()' on an "
                         "unintialised decomposition!");
    return MatrixType(permutationsP() * matrixL() * matrixL().transpose() *
                      permutationsP().transpose());
  }

  inline Index rows() const { return m_rows; }
  inline Index cols() const { return m_cols; }

  ComputationInfo info() const { return m_info; }

  RealScalar error() const {
    assert(finished() &&
           "Please don't call 'error()' on an unintialised decomposition!");
    return m_error;
  }

private:
  StorageIndex m_rows{0};
  StorageIndex m_cols{0};
  Index m_order{cDefaultOrder};
  RealScalar m_nugget{cDefaultNugget};
  RealScalar m_error{0};
  ComputationInfo m_info{Success};
  MatrixType m_L{};
  LDLT<MatrixXd> m_decomp{};
  MatrixXd m_Q{};
  HouseholderQR<MatrixXd> m_qr{};
  PermutationType m_transpositions{};
};

} // namespace Eigen