#ifndef GP_DISTANCE_H
#define GP_DISTANCE_H

#include "core/parameter_handler.h"

namespace albatross {

template <class Predictor>
class DistanceMetric : public ParameterHandlingMixin {
 public:
  DistanceMetric(){};
  virtual ~DistanceMetric(){};

  virtual double operator()(const Predictor &x, const Predictor &y) const = 0;

  //  virtual double gradient(const Predictor &x, const Predictor &y,
  //                          const Eigen::VectorXd &grad) = 0;
  //
  //  virtual void x_gradient(const Predictor &x, const Predictor &y,
  //                          Eigen::VectorXd &grad) = 0;
  //
  //  virtual void y_gradient(const Predictor &x, const Predictor &y,
  //                          Eigen::VectorXd &grad) {
  //    x_gradient(x, y, grad);
  //  };

 protected:
};

template <class Predictor>
class EuclideanDistance : public DistanceMetric<Predictor> {
 public:
  EuclideanDistance(){};
  ~EuclideanDistance(){};

  std::string get_name() const override { return "euclidean"; };

  double operator()(const Predictor &x, const Predictor &y) const {
    return (x - y).norm();
  }
};

template <class Predictor>
class RadialDistance : public DistanceMetric<Predictor> {
 public:
  RadialDistance(){};
  ~RadialDistance(){};

  std::string get_name() const override { return "radial"; };

  double operator()(const Predictor &x, const Predictor &y) const {
    return fabs(x.norm() - y.norm());
  }
};
}

#endif
