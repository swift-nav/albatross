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

#ifndef GP_COVARIANCE_FUNCTIONS_POLYNOMIALS_H
#define GP_COVARIANCE_FUNCTIONS_POLYNOMIALS_H

#include "covariance_base.h"

namespace albatross {

struct MeanTerm {};

class ConstantMean : public CovarianceBase {
 public:
  ConstantMean(double sigma_mean = 10.) {
    this->params_["sigma_constant_mean"] = sigma_mean;
  };

  ~ConstantMean(){};

  std::string get_name() const { return "constant_mean"; }

  template <typename Predictor>
  std::vector<MeanTerm> get_state_space_representation(std::vector<Predictor> &x) {
    std::vector<MeanTerm> terms = {MeanTerm()};
    return terms;
  }

  /*
   * This will create a covariance matrix that looks like,
   *     sigma_mean^2 * ones(m, n)
   * which is saying all observations are perfectly correlated,
   * so you can move one if you move the rest the same amount.
   */
  template <typename First, typename Second>
  double operator()(const First &x __attribute__((unused)),
                    const Second &y __attribute__((unused))) const {
    double sigma_mean = this->params_.at("sigma_constant_mean");
    return sigma_mean * sigma_mean;
  }

};

}

#endif
