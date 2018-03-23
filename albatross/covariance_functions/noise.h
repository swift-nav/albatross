#ifndef GP_COVARIANCE_FUNCTIONS_NOISE_H
#define GP_COVARIANCE_FUNCTIONS_NOISE_H

#include "covariance_base.h"

namespace albatross {

template <class Predictor>
class IndependentNoise : public CovarianceBase<Predictor> {
 public:
  IndependentNoise(double sigma_noise = 0.1) {
    this->params_["sigma_independent_noise"] = sigma_noise;
  };

  ~IndependentNoise(){};

  std::string get_name() const { return "independent_noise"; }

  /*
   * This will create a scaled identity matrix.
   */
  double operator()(const Predictor &x, const Predictor &y) const override {
    if (x == y) {
      double sigma_noise = this->params_.at("sigma_independent_noise");
      return sigma_noise * sigma_noise;
    } else {
      return 0.;
    }
  }
};
}

#endif
