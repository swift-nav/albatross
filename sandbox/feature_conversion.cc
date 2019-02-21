#include <iostream>

namespace albatross {

struct X {int value;};
struct Y {int value;};

template <typename ModelType> class ModelBase {

public:

  template <typename T>
  X convert_(const T &feature) {
    return derived().convert_(feature);
  }

  template <typename FeatureType>
  typename std::enable_if<has_possible_fit_impl<ModelType, FeatureType>::value, void>::type
  fit(const FeatureType &feature) {
    derived().fit_(feature);
  }

  template <typename FeatureType, typename OtherType>
  typename std::enable_if<has_possible_fit_impl<ModelType, FeatureType>::value, void>::type
  fit(const OtherType &other) {
    FeatureType feature = derived().convert_(other);
    fit(feature);
  }

  /*
   * CRTP Helpers
   */
  ModelType &derived() { return *static_cast<ModelType *>(this); }
  const ModelType &derived() const {
    return *static_cast<const ModelType *>(this);
  }
};

template <typename Derived>
class Middle : public ModelBase<Derived> {
public:

  void fit_(const X &feature) {
    std::cout << "Middle.fit_(X) : " << feature.value << std::endl;
  }

};

class Adapted : public Middle<Adapted> {
public:

  X convert_(const Y &feature) const {
    std::cout << "Adapted.convert_(Y) : " << feature.value << std::endl;
    return {feature.value};
  }

};

}

int main()
{
  using namespace albatross;

  Adapted m;

  X x = {1};
  m.fit(x);

  Y y = {2};
  m.fit<X>(y);
  // I'd really like to be able to do:
  //  m.fit(y);

  return 0;
}
