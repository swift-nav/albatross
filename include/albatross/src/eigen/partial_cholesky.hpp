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

template <typename Scalar> class PartialCholesky {
public:
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic>;
  using StorageIndex = typename MatrixType::StorageIndex;
  using RealScalar = typename NumTraits<Scalar>::Real;

  using PermutationType = PermutationMatrix<Dynamic, Dynamic, StorageIndex>;

  static constexpr const Index cDefaultOrder = 22;

  static constexpr const double cDefaultNugget = 1.e-5;

  PartialCholesky() {}

  explicit PartialCholesky(const MatrixType &A)
      : m_rows{A.rows()}, m_cols{A.cols()} {
    compute(A);
  }

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

  PartialCholesky &compute(const MatrixType &A) {
    // This computes a low-rank pivoted Cholesky decomposition.
    // Unlike the normal dense LDLT and friends, we do not want to
    // copy the potentially quite large A.  Unlike the CG solver, we
    // do not want to retain any reference to A's data once we have
    // finished this function.
    m_rows = A.rows();
    m_cols = A.cols();
    PermutationType transpositions(rows());
    transpositions.setIdentity();
    const Index max_rank = std::min(order(), rows());
    MatrixType L{MatrixType::Zero(rows(), max_rank)};

    // A's diagonal; needs to keep getting pivoted, shifted and
    // searched at each step
    VectorXd diag{A.diagonal()};

    // debug
    MatrixType Ashadow{A};
    MatrixXd Lshadow{L};
    PermutationType Pshadow{rows()};
    Pshadow.setIdentity();

    // Accumulated L column; also gets pivoted and shifted as we
    // update, and contains the corrections to all columns of A below
    // k.
    VectorXd L_accum{VectorXd::Zero(rows())};

    for (Index k = 0; k < max_rank; ++k) {
      Index max_index;
      diag.tail(max_rank - k).maxCoeff(&max_index);
      max_index += k;

      if (max_index != k) {
        std::cout << "Swapping '" << max_index << "' (" << diag(max_index)
                  << ") with '" << k << "' (" << diag(k) << ")" << std::endl;
        transpositions.applyTranspositionOnTheRight(k, max_index);
        std::swap(diag(k), diag(max_index));
        L.row(k).swap(L.row(max_index));
        std::swap(L_accum(k), L_accum(max_index));
      }

      std::cout << "P:\n" << Eigen::MatrixXi(transpositions) << std::endl;

      Index max_index_shadow;
      Ashadow.diagonal().tail(max_rank - k).maxCoeff(&max_index_shadow);
      max_index_shadow += k;

      if (max_index_shadow != k) {
        std::cout << "Shadow: swapping '" << max_index_shadow << "' ("
                  << Ashadow.diagonal()(max_index_shadow) << ") with '" << k
                  << "' (" << Ashadow.diagonal()(k) << ")" << std::endl;
        Lshadow.row(k).swap(Lshadow.row(max_index_shadow));
        Pshadow.applyTranspositionOnTheRight(k, max_index_shadow);
        PermutationType t{rows()};
        t.setIdentity();
        t.applyTranspositionOnTheRight(k, max_index_shadow);
        Ashadow = t.transpose() * Ashadow * t;
      }

      std::cout << "Pshadow:\n" << Eigen::MatrixXi(Pshadow) << std::endl;

      if (diag(k) <= 0.) {
        m_info = InvalidInput;
        std::cerr << "Invalid value '" << diag(k)
                  << "' in pivoted cholesky decomp!" << std::endl;
        return *this;
      }

      std::cout << "L:\n" << L << "\n";
      std::cout << "L_s:\n" << Lshadow << "\n";
      std::cout << "A_s:\n" << Ashadow << "\n";
      // std::cout << "L_accum: " << L_accum.transpose() << "\n";
      std::cout << "   diag: " << diag.transpose() << '\n';

      const RealScalar alpha = std::sqrt(diag(k));

      L(k, k) = alpha;
      Lshadow(k, k) = std::sqrt(Ashadow.diagonal()(k));

      std::cout << "alpha: " << alpha << std::endl;
      std::cout << "shadow alpha: " << Lshadow(k, k) << std::endl;

      const Index tail_size = rows() - k - 1;
      std::cout << "tail_size: " << tail_size << std::endl;

      if (tail_size < 1) {
        break;
      }
      // Everything below here should be ordered appropriately -- we
      // pivot `diag` and `L_accum` and the rows of `L` every time we do an
      // update
      const VectorXd Acol =
          ((transpositions.transpose() * A.col(transpositions.indices()(k))))
              .tail(tail_size);
      const VectorXd Acol_shadow = Ashadow.col(k).tail(tail_size);
      std::cout << "  Acol: " << Acol.transpose() << std::endl;
      std::cout << "Acol_s: " << Acol_shadow.transpose() << std::endl;
      std::cout << "  diff: " << (Acol - Acol_shadow).transpose() << std::endl;

      VectorXd b = Acol;
      // if (k > 0) {
      //   b -= L.col(k - 1).tail(tail_size) * L(k, k - 1);
      // }
      for (Index i = 0; i < k; ++i) {
        std::cout << "delta b(" << i
                  << "): " << L.col(i).tail(tail_size).transpose() << " * "
                  << L(k, i) << " = "
                  << L.col(i).tail(tail_size).transpose() * L(k, i)
                  << std::endl;
        b -= L.col(i).tail(tail_size) * L(k, i);
      }  // + L_accum.tail(tail_size);
      std::cout << "     b: " << b.transpose() << std::endl;

      L.col(k).tail(tail_size) = b / alpha;
      diag.tail(tail_size) -=
          L.col(k).tail(tail_size).array().square().matrix();
      L_accum.tail(tail_size) -= Acol * Acol(0) / (alpha * alpha);

      Lshadow.col(k).tail(tail_size) =
        Acol_shadow / std::sqrt(Ashadow.diagonal()(k));
      std::cout << "shadow bb^t / alpha:\n"
                << 1 / (alpha * alpha) * Acol_shadow * Acol_shadow.transpose()
                << std::endl;
      Ashadow.bottomRightCorner(tail_size, tail_size) -=
          1 / (alpha * alpha) * Acol_shadow * Acol_shadow.transpose();

      std::cout << "L:\n" << L << "\n";
      std::cout << "L_s:\n" << Lshadow << "\n";
      std::cout << "A_s:\n" << Ashadow << "\n";
      // std::cout << "L_accum: " << L_accum.transpose() << "\n";
      std::cout << "   diag: " << diag.transpose() << '\n';
      std::cout << "========" << std::endl;
    }

    m_transpositions = transpositions;
    // std::cout << MatrixXi(m_transpositions) << std::endl;

    m_L = L;
    // std::cout << L << std::endl;

    std::cout << MatrixXd(transpositions * L * L.transpose() *
                          transpositions.transpose())
              << std::endl;

    std::cout << MatrixXd(Pshadow * Lshadow * Lshadow.transpose() *
                          Pshadow.transpose())
              << std::endl;

    m_decomp = LDLT<MatrixXd>(MatrixXd::Identity(L.cols(), L.cols()) +
                              1 / (m_nugget * m_nugget) * L.transpose() * L);

    m_info = Success;
    return *this;
  }

  MatrixXd matrixL() const {
    assert(finished() &&
           "Please don't call this on an uninitialised decomposition!");

    return m_L;
  }

  PermutationType permutationsP() const {
    assert(finished() &&
           "Please don't call this on an uninitialised decomposition!");

    return m_transpositions;
  }

  template <typename Rhs>
  Matrix<Scalar, Dynamic, Rhs::ColsAtCompileTime> solve(const Rhs &b) const {
    assert(finished() &&
           "Please don't call this on an unintialised decomposition!");
    const double n2 = m_nugget * m_nugget;
    std::cout << 1 / n2 * b << std::endl;
    std::cout << m_decomp.solve(m_L.transpose() * b) << std::endl;
    Matrix<Scalar, Dynamic, Rhs::ColsAtCompileTime> ret =
        1 / n2 * b -
        1 / (n2 * n2) *
            (m_transpositions * m_L *
             m_decomp.solve(m_L.transpose() * m_transpositions.transpose() * b));
    return ret;
  }

  inline bool finished() const {
    return rows() > 0 && cols() > 0 & info() == Success;
  }

  inline Index rows() const { return m_rows; }
  inline Index cols() const { return m_cols; }

  ComputationInfo info() const { return m_info; }

private:
  StorageIndex m_rows{0};
  StorageIndex m_cols{0};
  Index m_order{cDefaultOrder};
  RealScalar m_nugget{cDefaultNugget};
  ComputationInfo m_info{Success};
  MatrixType m_L{};
  LDLT<MatrixXd> m_decomp{};
  PermutationType m_transpositions{};
};

} // namespace Eigen