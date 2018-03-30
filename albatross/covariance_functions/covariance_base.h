#ifndef GP_COVARIANCE_BASE_H
#define GP_COVARIANCE_BASE_H

#include <sstream>
#include "map_utils.h"
#include "core/parameter_handler.h"

namespace albatross {

/*
 * This struct determines whether or not a class has a method defined for,
 *   `operator() (First &first, Second &second, ...)`
 * The result of the inspection gets stored in the member `value`.
 */
template <typename T, typename... Args>
class has_call_operator
{
    template <typename C,
              typename = decltype( std::declval<C>()(std::declval<Args>()...))>
    static std::true_type test(int);
    template <typename C>
    static std::false_type test(...);

public:
    static constexpr bool value = decltype(test<T>(0))::value;
};


/*
 * An abstract class (though no purely abstract due to templating)
 * which holds anything all Covariance terms should have in common.
 */
class CovarianceBase : public ParameterHandlingMixin {
 public:
  CovarianceBase() : ParameterHandlingMixin(){};
  virtual ~CovarianceBase(){};

  template<typename First, typename Second>
  double operator()(const First &first, const Second &second) const;
};

/*
 * As we start composing Covariance objects we want to keep track of
 * their parameters and names in a friendly way.  This class deals with
 * any of those operations that are shared for all composition operations.
 */
template <class X, class Y>
class CombinationOfCovariances : public CovarianceBase {
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


template <class X, class Y>
class SumOfCovariance : public CombinationOfCovariances<X, Y> {
 public:
  SumOfCovariance(X &x, Y &y)
      : CombinationOfCovariances<X, Y>(x, y){};

  std::string get_operation_symbol() const { return "+"; }

  /*
   * If both X and Y have a valid call method for the types First and Second
   * this will return the sum of the two.
   */
  template <typename First,
            typename Second,
            typename std::enable_if<(has_call_operator<X, First&, Second&>::value &&
                                     has_call_operator<Y, First&, Second&>::value),
                                    int>::type = 0>
  double operator() (First &first, Second &second) const {
    return this->x_(first, second) + this->y_(first, second);
  }

  /*
   * If only X has a valid call method we ignore Y.
   */
  template <typename First,
            typename Second,
            typename std::enable_if<(has_call_operator<X, First&, Second&>::value &&
                                     !has_call_operator<Y, First&, Second&>::value),
                                    int>::type = 0>
  double operator() (First &first, Second &second) const {
    std::cout << "sum[x]" << std::endl;
    return this->x_(first, second);
  }

  /*
   * If only Y has a valid call method we ignore X.
   */
  template <typename First,
            typename Second,
            typename std::enable_if<(!has_call_operator<X, First&, Second&>::value &&
                                     has_call_operator<Y, First&, Second&>::value),
                                    int>::type = 0>
  double operator() (First &first, Second &second) const {
    std::cout << "sum[y]" << std::endl;
    return this->y_(first, second);
  }

  /*
   * If neither have a valid call method we assume zero correlation.
   */
  template <typename First,
            typename Second,
            typename std::enable_if<(!has_call_operator<X, First&, Second&>::value &&
                                     !has_call_operator<Y, First&, Second&>::value),
                                    int>::type = 0>
  double operator() (First &first, Second &second) const {
    std::cout << "sum[]" << std::endl;
    return 0.;
  }


};

template <class X, class Y>
class ProductOfCovariance : public CombinationOfCovariances<X, Y> {
 public:
  ProductOfCovariance(X &x, Y &y)
      : CombinationOfCovariances<X, Y>(x, y){};

  std::string get_operation_symbol() const { return "*"; }

  /*
   * If both X and Y have a valid call method for the types First and Second
   * this will return the sum of the two.
   */
  template <typename First,
            typename Second,
            typename std::enable_if<(has_call_operator<X, First&, Second&>::value &&
                                     has_call_operator<Y, First&, Second&>::value),
                                    int>::type = 0>
  double operator() (First &first, Second &second) const {
    return this->x_(first, second) * this->y_(first, second);
  }

  /*
   * If only X has a valid call method we ignore Y.
   */
  template <typename First,
            typename Second,
            typename std::enable_if<(has_call_operator<X, First&, Second&>::value &&
                                     !has_call_operator<Y, First&, Second&>::value),
                                    int>::type = 0>
  double operator() (First &first, Second &second) const {
    return this->x_(first, second);
  }

  /*
   * If only Y has a valid call method we ignore X.
   */
  template <typename First,
            typename Second,
            typename std::enable_if<(!has_call_operator<X, First&, Second&>::value &&
                                     has_call_operator<Y, First&, Second&>::value),
                                    int>::type = 0>
  double operator() (First &first, Second &second) const {
    return this->y_(first, second);
  }

  /*
   * If neither have a valid call method we assume zero correlation.
   */
  template <typename First,
            typename Second,
            typename std::enable_if<(!has_call_operator<X, First&, Second&>::value &&
                                     !has_call_operator<Y, First&, Second&>::value),
                                    int>::type = 0>
  double operator() (First &first, Second &second) const {
    return 0.;
  }

};

}  // albatross

#endif
