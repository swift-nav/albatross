/*
 * Copyright (C) 2023 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include <albatross/Core>
#include <albatross/GP>
#include <albatross/linalg/Utils>
#include <albatross/linalg/Block>
#include <gtest/gtest.h>

namespace albatross {

TEST(test_linalg_utils, test_block_arrow_ldlt) {
  const std::vector<double> u_0 = {-2., -1.};
  const std::vector<double> u_1 = {1., 2.};
  const std::vector<double> u_c = {0.};

  const std::vector<double> u_g = concatenate(std::vector<std::vector<double>>{u_0, u_1});
  const std::vector<double> u = concatenate(std::vector<std::vector<double>>{u_0, u_1, u_c});

  SquaredExponential<EuclideanDistance> cov;
  cov.squared_exponential_length_scale.value = 1.;
  cov.sigma_squared_exponential.value = 1.;
  Eigen::MatrixXd K_uu = Eigen::MatrixXd::Zero(u.size(), u.size());

  // Diagonal Blocks
  K_uu.block(0, 0, 2, 2) = cov(u_0);
  K_uu.block(2, 2, 2, 2) = cov(u_1);
  K_uu.block(4, 4, 1, 1) = cov(u_c);
  // Upper right
  K_uu.block(0, 4, 2, 1) = cov(u_0, u_c);
  K_uu.block(2, 4, 2, 1) = cov(u_1, u_c);
  // Lower Left
  K_uu.block(4, 0, 1, 2) = cov(u_c, u_0);
  K_uu.block(4, 2, 1, 2) = cov(u_c, u_1);

  std::cout << "K_uu" << std::endl;
  std::cout << K_uu << std::endl;

  const BlockDiagonal K_gg({cov(u_0), cov(u_1)});
  const Eigen::MatrixXd K_gc = cov(u_g, u_c);
  const Eigen::MatrixXd K_cc = cov(u_c);

  std::cout << "K_gg" << std::endl;
  std::cout << K_gg.toDense() << std::endl;

  std::cout << "K_gc" << std::endl;
  std::cout << K_gc << std::endl;

  std::cout << "K_cc" << std::endl;
  std::cout << K_cc << std::endl;
  const auto arrow_ldlt = block_symmetric_arrow_ldlt(K_gg, K_gc, K_cc);

  std::cout << "EXPECTED : " << std::endl;
  const Eigen::MatrixXd rhs = Eigen::MatrixXd::Random(K_uu.rows(), 2);
  const Eigen::SerializableLDLT expected_ldlt(K_uu);
  const Eigen::MatrixXd expected = expected_ldlt.sqrt_solve(rhs);

  std::cout << expected.transpose() * expected << std::endl;

  std::cout << "ACTUAL : " << std::endl;
  const Eigen::MatrixXd actual = arrow_ldlt.sqrt_solve(rhs);
  std::cout << actual.transpose() * actual << std::endl;

  EXPECT_LT((expected - actual).norm(), 1e-8);
}

TEST(test_linalg_utils, test_cod_reconstruction) {
  const Eigen::Index m = 5;
  const Eigen::Index n = 3;
  Eigen::MatrixXd A = Eigen::MatrixXd::Random(m, n);
  A.col(0) = A.col(1);
  Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(A);
  std::cout << A << std::endl;
  
  std::cout << "rank: " << cod.rank() << std::endl;
  std::cout << "Q: " << std::endl;
  const Eigen::MatrixXd Q_full = cod.matrixQ();
  const Eigen::MatrixXd Q = Q_full.leftCols(cod.rank());
  std::cout << Q << std::endl;
  std::cout << "T: " << std::endl;
  const Eigen::MatrixXd T = cod.matrixT().topLeftCorner(cod.rank(), cod.rank()).triangularView<Eigen::Upper>();
  std::cout << T << std::endl;
  const Eigen::MatrixXd Z = cod.matrixZ().topRows(cod.rank());
  const Eigen::MatrixXd PZt = (cod.colsPermutation() * Z.transpose());
  const Eigen::MatrixXd ZPt = PZt.transpose();
  std::cout << "Z: " << std::endl;
  std::cout << Z << std::endl;
  std::cout << "ZP.T: " << std::endl;
  std::cout << ZPt << std::endl;

  const Eigen::MatrixXd recon = Q * T * ZPt;
  std::cout << "recon:" << std::endl;
  std::cout << recon << std::endl;

  EXPECT_LT((recon - A).norm(), 1e-6);

  const Eigen::MatrixXd ZZT = Z * Z.transpose();
  const Eigen::MatrixXd eye = Eigen::MatrixXd::Identity(cod.rank(), cod.rank());
  std::cout << ZZT << std::endl;
  EXPECT_LT((ZZT - eye).norm(), 1e-6);
  const Eigen::MatrixXd QTQ = Q.transpose() * Q;
  EXPECT_LT((QTQ - eye).norm(), 1e-6);

  const Eigen::MatrixXd Z_null = cod.matrixZ().bottomRows(n - cod.rank());
  const Eigen::MatrixXd N = (cod.colsPermutation() * Z_null.transpose());

  EXPECT_LT((A * N).norm(), 1e-6);
}

TEST(test_linalg_utils, test_col_pivot_reconstruction) {
  const Eigen::Index m = 5;
  const Eigen::Index n = 3;
  Eigen::MatrixXd A = Eigen::MatrixXd::Random(m, n);
  A.col(0) = A.col(1);
  std::cout << A << std::endl;
  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(A);
  const Eigen::MatrixXd Q = qr.matrixQ();
  std::cout << Q << std::endl;

  const Eigen::MatrixXd eye = Q * Q.transpose();
  std::cout << eye << std::endl;
  
  const Eigen::MatrixXd R = get_R(qr);
  std::cout << "R:" << std::endl;
  std::cout << R << std::endl;

  //const Eigen::Matrix<int, Eigen::Dynamic, 1> P = qr.colsPermutation().indices();

  const Eigen::MatrixXd RPt = R * qr.colsPermutation().transpose();
  const Eigen::MatrixXd recon = Q.leftCols(R.rows()) * RPt;
  std::cout << recon << std::endl;
  EXPECT_LT((recon - A).norm(), 1e-6);
}

TEST(test_linalg_utils, test_block_structured_qr_reconstruct_square) {
  const Eigen::Index m = 5;
  const Eigen::Index n = 3;
  const Eigen::Index k = 2;
  Eigen::MatrixXd A_0 = Eigen::MatrixXd::Random(m, n);
  //A_0.col(0) = A_0.col(1);

  Eigen::MatrixXd A_1 = Eigen::MatrixXd::Random(m - 1, n);

  BlockDiagonal A({A_0, A_1});
  Eigen::MatrixXd rhs = Eigen::MatrixXd::Random(A_0.rows() + A_1.rows() + k,
                                                k);
  Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(A_0.rows() + A_1.rows() + k,
                                                A_0.cols() + A_1.cols() + k);

  dense.block(0, 0, A_0.rows(), A_0.cols()) = A_0;
  dense.block(A_0.rows(), A_0.cols(), A_1.rows(), A_1.cols()) = A_1;
  dense.rightCols(k) = rhs;

  std::cout << "DENSE ==============" << std::endl;
  std::cout << dense << std::endl;

  const auto structured_qr = create_structured_qr(A, rhs);

  const Eigen::MatrixXd actual = dense.transpose() * dense;

  const Eigen::Index block_rows = structured_qr.R.upper_left.rows();

  Eigen::MatrixXd recon = Eigen::MatrixXd::Zero(actual.rows(), actual.cols());
  Eigen::Index offset = 0;
  for (const auto &block : structured_qr.R.upper_left.blocks) {
    Eigen::MatrixXd RPt = block.R * block.P.transpose();
    std::cout << "----" << std::endl;
    recon.block(offset, offset, RPt.rows(), RPt.cols()) = RPt.transpose() * RPt;
    const Eigen::MatrixXd cross = structured_qr.R.upper_right.block(
        offset, 0, RPt.rows(), structured_qr.R.upper_right.cols());
    recon.block(block_rows, offset, cross.cols(), RPt.rows()) =
        cross.transpose() * RPt;
    recon.block(offset, block_rows, RPt.rows(), cross.cols()) =
        recon.block(block_rows, offset, cross.cols(), RPt.rows()).transpose();
    offset += RPt.rows();
  }
  const DenseR &corner = structured_qr.R.lower_right;
  const Eigen::Index common_rows = corner.R.rows();
  Eigen::MatrixXd RPt = corner.R * corner.P.transpose();

  std::cout << RPt << std::endl;
  const Eigen::MatrixXd D = RPt.transpose() * RPt;
  const Eigen::MatrixXd Z =
      structured_qr.R.upper_right.transpose() * structured_qr.R.upper_right;
  recon.bottomRightCorner(common_rows, common_rows) = D + Z;

  std::cout << "ACTUAL ==============" << std::endl;
  std::cout << actual << std::endl;
  std::cout << "RECON =============" << std::endl;
  std::cout << recon << std::endl;
  EXPECT_LT((actual - recon).norm(), 1e-8);
}

TEST(test_linalg_utils, test_block_structured_qr_reconstruct) {
  const Eigen::Index m = 5;
  const Eigen::Index n = 3;
  const Eigen::Index k = 2;
  Eigen::MatrixXd A_0 = Eigen::MatrixXd::Random(m, n);
  //A_0.col(0) = A_0.col(1);

  Eigen::MatrixXd A_1 = Eigen::MatrixXd::Random(m - 1, n);

  BlockDiagonal A({A_0, A_1});
  Eigen::MatrixXd rhs = Eigen::MatrixXd::Random(A_0.rows() + A_1.rows() + k,
                                                k);
  Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(A_0.rows() + A_1.rows() + k,
                                                A_0.cols() + A_1.cols() + k);

  dense.block(0, 0, A_0.rows(), A_0.cols()) = A_0;
  dense.block(A_0.rows(), A_0.cols(), A_1.rows(), A_1.cols()) = A_1;
  dense.rightCols(k) = rhs;

  std::cout << "DENSE ==============" << std::endl;
  std::cout << dense << std::endl;

  const auto structured_qr = create_structured_qr(A, rhs);

  const Eigen::MatrixXd actual = dense.transpose() * dense;

  const Eigen::Index block_rows = structured_qr.R.upper_left.rows();

  Eigen::MatrixXd recon = Eigen::MatrixXd::Zero(actual.rows(), actual.cols());
  Eigen::Index offset = 0;
  for (const auto &block : structured_qr.R.upper_left.blocks) {
    Eigen::MatrixXd RPt = block.R * block.P.transpose();
    std::cout << "----" << std::endl;
    recon.block(offset, offset, RPt.rows(), RPt.cols()) = RPt.transpose() * RPt;
    const Eigen::MatrixXd cross = structured_qr.R.upper_right.block(
        offset, 0, RPt.rows(), structured_qr.R.upper_right.cols());
    recon.block(block_rows, offset, cross.cols(), RPt.rows()) =
        cross.transpose() * RPt;
    recon.block(offset, block_rows, RPt.rows(), cross.cols()) =
        recon.block(block_rows, offset, cross.cols(), RPt.rows()).transpose();
    offset += RPt.rows();
  }
  const DenseR &corner = structured_qr.R.lower_right;
  const Eigen::Index common_rows = corner.R.rows();
  Eigen::MatrixXd RPt = corner.R * corner.P.transpose();

  std::cout << RPt << std::endl;
  const Eigen::MatrixXd D = RPt.transpose() * RPt;
  const Eigen::MatrixXd Z =
      structured_qr.R.upper_right.transpose() * structured_qr.R.upper_right;
  recon.bottomRightCorner(common_rows, common_rows) = D + Z;

  std::cout << "ACTUAL ==============" << std::endl;
  std::cout << actual << std::endl;
  std::cout << "RECON =============" << std::endl;
  std::cout << recon << std::endl;
  EXPECT_LT((actual - recon).norm(), 1e-8);
}

TEST(test_linalg_utils, test_block_structured_qr_sqrt_solve) {
  const Eigen::Index m = 5;
  const Eigen::Index n = 3;
  const Eigen::Index k = 2;
  Eigen::MatrixXd A_0 = Eigen::MatrixXd::Random(m, n);
  // A_0.col(0) = A_0.col(1);

  Eigen::MatrixXd A_1 = Eigen::MatrixXd::Random(m - 1, n);

  BlockDiagonal A({A_0, A_1});
  Eigen::MatrixXd rhs = Eigen::MatrixXd::Random(A_0.rows() + A_1.rows() + k, k);
  Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(A_0.rows() + A_1.rows() + k,
                                                A_0.cols() + A_1.cols() + k);

  dense.block(0, 0, A_0.rows(), A_0.cols()) = A_0;
  dense.block(A_0.rows(), A_0.cols(), A_1.rows(), A_1.cols()) = A_1;
  dense.rightCols(k) = rhs;

  std::cout << "DENSE ==============" << std::endl;
  std::cout << dense << std::endl;

  const auto structured_qr = create_structured_qr(A, rhs);

  const Eigen::MatrixXd actual = dense.transpose() * dense;

  const Eigen::MatrixXd eye = Eigen::MatrixXd::Identity(dense.cols(), dense.cols());
  const auto sqrt_inv = sqrt_solve(structured_qr.R, eye);
  std::cout << "SQRT INV =============" << std::endl;
  std::cout << sqrt_inv << std::endl;
  const Eigen::MatrixXd inv = sqrt_inv.transpose() * sqrt_inv;

  const Eigen::MatrixXd actual_inv = actual.inverse();

  std::cout << "INV =============" << std::endl;
  std::cout << inv << std::endl;
  std::cout << "ACTUAL INV =============" << std::endl;
  std::cout << actual_inv << std::endl;
  EXPECT_LT((inv - actual_inv).norm(), 1e-8);
}

TEST(test_linalg_utils, test_block_structured_q_transpose) {
  const Eigen::Index m = 5;
  const Eigen::Index n = 3;
  const Eigen::Index k = 2;
  Eigen::MatrixXd A_0 = Eigen::MatrixXd::Random(m, n);
  // A_0.col(0) = A_0.col(1);

  Eigen::MatrixXd A_1 = Eigen::MatrixXd::Random(m - 1, n);

  BlockDiagonal A({A_0, A_1});
  Eigen::MatrixXd rhs = Eigen::MatrixXd::Random(A_0.rows() + A_1.rows() + k, k);
  Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(A_0.rows() + A_1.rows() + k,
                                                A_0.cols() + A_1.cols() + k);

  dense.block(0, 0, A_0.rows(), A_0.cols()) = A_0;
  dense.block(A_0.rows(), A_0.cols(), A_1.rows(), A_1.cols()) = A_1;
  dense.rightCols(k) = rhs;

  std::cout << "DENSE ==============" << std::endl;
  std::cout << dense << std::endl;

  const auto structured_qr = create_structured_qr(A, rhs);

  Eigen::MatrixXd eye = Eigen::MatrixXd::Identity(dense.rows(), dense.rows());
  const Eigen::MatrixXd Qt = dot_transpose(structured_qr.Q, eye);

  // Q^T Q = I
  EXPECT_LT((Qt * Qt.transpose() - eye).norm(), 1e-8);
  // Q^T Q, but with dot_transpose()
  const auto hopefully_eye = dot_transpose(structured_qr.Q, Qt.transpose());
  EXPECT_LT((hopefully_eye - eye).norm(), 1e-8);
}

TEST(test_linalg_utils, test_block_structured_qr_solve) {
  const Eigen::Index m = 5;
  const Eigen::Index n = 3;
  const Eigen::Index k = 2;
  Eigen::MatrixXd A_0 = Eigen::MatrixXd::Random(m, n);
  // A_0.col(0) = A_0.col(1);

  Eigen::MatrixXd A_1 = Eigen::MatrixXd::Random(m - 1, n);

  BlockDiagonal A({A_0, A_1});
  Eigen::MatrixXd rhs = Eigen::MatrixXd::Random(A_0.rows() + A_1.rows() + k, k);
  Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(A_0.rows() + A_1.rows() + k,
                                                A_0.cols() + A_1.cols() + k);

  dense.block(0, 0, A_0.rows(), A_0.cols()) = A_0;
  dense.block(A_0.rows(), A_0.cols(), A_1.rows(), A_1.cols()) = A_1;
  dense.rightCols(k) = rhs;

  const auto structured_qr = create_structured_qr(A, rhs);

  const Eigen::MatrixXd x = Eigen::MatrixXd::Random(dense.cols(), 1);
  const Eigen::MatrixXd b = dense * x;

  const auto qr = dense.colPivHouseholderQr();
  const Eigen::MatrixXd direct = qr.solve(b);

  EXPECT_LT((direct - x).norm(), 1e-8);

  const auto soln = solve(structured_qr, b);
  EXPECT_LT((soln - x).norm(), 1e-8);
}



} // namespace albatross
