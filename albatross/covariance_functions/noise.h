#ifndef GP_COVARIANCE_FUNCTIONS_NOISE_H
#define GP_COVARIANCE_FUNCTIONS_NOISE_H

#include "covariance_base.h"

namespace albatross {

template <typename Observed>
class IndependentNoise : public CovarianceBase {
 public:
  IndependentNoise(double sigma_noise = 0.1) {
    this->params_["sigma_independent_noise"] = sigma_noise;
  };

  ~IndependentNoise(){};

  std::string get_name() const { return "independent_noise"; }

  /*
   * This will create a scaled identity matrix, but only between
   * two different observations defined by the Observed type.
   */
  double operator()(const Observed &x, const Observed &y) const {
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
