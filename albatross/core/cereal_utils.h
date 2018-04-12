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

#ifndef ALBATROSS_CEREAL_UTILS_H
#define ALBATROSS_CEREAL_UTILS_H


#include "cereal/cereal.hpp"
#include <cereal/types/map.hpp>
#include <cereal/types/vector.hpp>
#include <map>
#include "Eigen/Dense"

namespace cereal {


template <class Archive>
inline
void save(Archive& archive, const Eigen::VectorXd& v)
{
    size_type s = static_cast<size_type>(v.size());
    archive(cereal::make_size_tag(s));
    for (size_type i = 0; i < s; i++) {
      archive(v[i]);
    }
};

template <class Archive>
inline
void load(Archive& archive, Eigen::VectorXd& v)
{
    size_type s = static_cast<size_type>(v.size());
    archive(cereal::make_size_tag(s));
    v.resize(s);
    for (size_type i = 0; i < s; i++) {
      archive(v[i]);
    }
};

/*
 * For length 3 vectors, we wrap the length L call.
 */
template <typename Archive>
inline
void save(Archive &archive, const Eigen::Vector3d &v) {
  Eigen::VectorXd as_xd(v.size());
  as_xd << v;
  save(archive, as_xd);
}

template <typename Archive>
inline
void load(Archive &archive, Eigen::Vector3d &v) {
  Eigen::VectorXd as_xd(v.size());
  load(archive, as_xd);
  for (int i = 0; i < 3; i++) {v[i] = as_xd[i];}
}

template <class Archive>
inline
void save(Archive& archive, const Eigen::MatrixXd& v)
{
    size_type rows = static_cast<size_type>(v.rows());
    archive(cereal::make_size_tag(rows));
    for (size_type i = 0; i < rows; i++) {
      Eigen::VectorXd row = v.row(i);
      archive(row);
    }
};

template <class Archive>
inline
void load(Archive& archive, Eigen::MatrixXd &v)
{
    size_type rows = static_cast<size_type>(v.rows());
    archive(cereal::make_size_tag(rows));
    v.resize(rows, rows);
    for (size_type i = 0; i < rows; i++) {
      Eigen::VectorXd row;
      archive(row);
      v.row(i) = row;
    }
};

} // namespace cereal

#endif
