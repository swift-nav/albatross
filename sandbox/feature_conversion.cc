#include <iostream>

namespace albatross {

struct X {int value;};
struct Y {int value;};

template <typename ModelType> class ModelBase {
private:
  // This is nice to have because it makes sure you don't
  // accidentally do something like:
  //     class A : public ModelBase<B>
  ModelBase() {};
  friend ModelType;

public:

  template <typename FeatureType>
  typename std::enable_if<has_possible_fit_impl<ModelType, FeatureType>::value, void>::type
  fit(const FeatureType &feature) {
    derived().fit_(feature);
  }

  /*
   * CRTP Helpers
   */
  ModelType &derived() { return *static_cast<ModelType *>(this); }
  const ModelType &derived() const {
    return *static_cast<const ModelType *>(this);
  }
};

void middle_fit(const X feature) {
  std::cout << "middle_fit(X) : " << feature.value << std::endl;
}

class Middle {
public:

  void fit_(const X &feature) {
    std::cout << "Middle.fit_(X) : " << feature.value << std::endl;
  }

};

class Adapted : public ModelBase<Adapted> {
public:

  void fit_(const X &feature) {
    std::cout << "Adapted.fit_(X) : " << feature.value << std::endl;
    middle_fit(feature);
  }

  X convert_feature(const Y &y) {
    return {y.value};
  }

  void fit_(const Y &feature) {
    std::cout << "Adapted.fit_(Y) : " << feature.value << std::endl;
    fit_(convert_feature(feature));
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
  m.fit(y);
  // I'd really like to be able to do:
  //  m.fit(y);

  return 0;
}
