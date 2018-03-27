#ifndef GP_DISTANCE_H
#define GP_DISTANCE_H

#include "core/parameter_handler.h"
#include <Eigen/Core>

namespace albatross {

class DistanceMetric : public ParameterHandlingMixin {
 public:
  DistanceMetric(){};
  virtual ~DistanceMetric(){};

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

class EuclideanDistance : public DistanceMetric {
 public:
  EuclideanDistance(){};
  ~EuclideanDistance(){};

  std::string get_name() const override { return "euclidean"; };

  double operator()(const Eigen::VectorXd &x, const Eigen::VectorXd &y) const {
    return (x - y).norm();
  }
};

class RadialDistance : public DistanceMetric {
 public:
  RadialDistance(){};
  ~RadialDistance(){};

  std::string get_name() const override { return "radial"; };

  double operator()(const Eigen::VectorXd &x, const Eigen::VectorXd &y) const {
    return fabs(x.norm() - y.norm());
  }
};
}

#endif
