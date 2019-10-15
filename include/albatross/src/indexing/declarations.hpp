/*
 * Copyright (C) 2019 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef ALBATROSS_INDEXING_DECLARATIONS_HPP_
#define ALBATROSS_INDEXING_DECLARATIONS_HPP_

namespace albatross {

template <typename SizeType>
Eigen::MatrixXd symmetric_subset(const Eigen::MatrixXd &v,
                                 const std::vector<SizeType> &indices);

template <typename SizeType, typename Scalar, int Size>
Eigen::DiagonalMatrix<Scalar, Size>
symmetric_subset(const Eigen::DiagonalMatrix<Scalar, Size> &v,
                 const std::vector<SizeType> &indices);

template <typename SizeType>
Eigen::VectorXd subset(const Eigen::VectorXd &v,
                       const std::vector<SizeType> &indices);

template <typename SizeType, typename X>
std::vector<X> subset(const std::vector<X> &v,
                      const std::vector<SizeType> &indices);

template <typename SizeType>
Eigen::MatrixXd subset_cols(const Eigen::MatrixXd &v,
                            const std::vector<SizeType> &col_indices);

} // namespace albatross

#endif /* ALBATROSS_INDEXING_DECLARATIONS_HPP_ */
