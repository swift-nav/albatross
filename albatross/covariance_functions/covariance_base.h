#ifndef GP_COVARIANCE_BASE_H
#define GP_COVARIANCE_BASE_H

#include <sstream>
#include "map_utils.h"
#include "core/parameter_handler.h"

namespace albatross {

template <class Predictor>
class CovarianceBase : public ParameterHandlingMixin {
 public:
  CovarianceBase() : ParameterHandlingMixin(){};
  virtual ~CovarianceBase(){};

  virtual double operator()(const Predictor &x, const Predictor &y) const = 0;
};

template <class X, class Y, class Predictor>
class CombinationOfCovariances : public CovarianceBase<Predictor> {
 public:
  CombinationOfCovariances(X &x, Y &y) : x_(x), y_(y){};
  virtual ~CombinationOfCovariances(){};

  virtual std::string get_operation_symbol() const = 0;

  std::string get_name() const {
    std::ostringstream oss;
    oss << "(" << x_.get_name() << get_operation_symbol() << y_.get_name()
        << ")";
    return oss.str();
  }

  ParameterStore get_params() const override {
    return map_join(x_.get_params(), y_.get_params());
  }

  void unchecked_set_param(const std::string &name,
                           const double value) override {
    if (map_contains(x_.get_params(), name)) {
      x_.set_param(name, value);
    } else {
      y_.set_param(name, value);
    }
  }

 protected:
  X x_;
  Y y_;
};

template <class X, class Y, class Predictor>
class SumOfCovariance : public CombinationOfCovariances<X, Y, Predictor> {
 public:
  SumOfCovariance(X &x, Y &y)
      : CombinationOfCovariances<X, Y, Predictor>(x, y){};

  std::string get_operation_symbol() const { return "+"; }

  double operator()(const Predictor &x, const Predictor &y) const override {
    return this->x_(x, y) + this->y_(x, y);
  }
};

template <class X, class Y, class Predictor>
class ProductOfCovariance : public CombinationOfCovariances<X, Y, Predictor> {
 public:
  ProductOfCovariance(X &x, Y &y)
      : CombinationOfCovariances<X, Y, Predictor>(x, y){};

  std::string get_operation_symbol() const { return "*"; }

  double operator()(const Predictor &x, const Predictor &y) const override {
    return this->x_(x, y) * this->y_(x, y);
  }
};

}  // albatross

#endif
