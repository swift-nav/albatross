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

#ifndef ALBATROSS_CORE_PRIORS_H
#define ALBATROSS_CORE_PRIORS_H

namespace albatross {

constexpr double LOG_2PI_ = 1.8378770664093453;
constexpr double LARGE_VAL = HUGE_VAL;

/*
 * To add a new prior you'll need to implement the Prior abstract
 * class.  If you want to be able to serialize models that use
 * the prior you'll then also need to register the class following
 * examples at the bottom of this file.
 */

class Prior {
public:
  virtual ~Prior(){};
  virtual double log_pdf(double x) const = 0;
  virtual std::string get_name() const = 0;
  virtual double lower_bound() const { return -LARGE_VAL; }
  virtual double upper_bound() const { return LARGE_VAL; }
  virtual bool is_log_scale() const { return false; };
  virtual bool is_fixed() const { return false; }
  virtual bool operator==(const Prior &other) const {
    return typeid(*this) == typeid(other);
  }
  bool operator!=(const Prior &other) { return !((*this) == other); }

  template <typename Archive> void serialize(Archive &, const std::uint32_t) {}
};

class UninformativePrior : public Prior {
public:
  std::string get_name() const override { return "uninformative"; };
  double log_pdf(double) const override { return 0.; }

  template <typename Archive>
  void serialize(Archive &archive, const std::uint32_t) {
    archive(cereal::base_class<Prior>(this));
  }
};

class FixedPrior : public Prior {
public:
  std::string get_name() const override { return "fixed"; };
  double log_pdf(double) const override { return 0.; }

  bool is_fixed() const override { return true; }

  template <typename Archive>
  void serialize(Archive &archive, const std::uint32_t) {
    archive(cereal::base_class<Prior>(this));
  }
};

class PositivePrior : public Prior {
public:
  double log_pdf(double x) const override { return x > 0. ? 0. : -LARGE_VAL; }
  std::string get_name() const override { return "positive"; };
  double lower_bound() const override { return 0.; }
  double upper_bound() const override { return LARGE_VAL; }

  template <typename Archive>
  void serialize(Archive &archive, const std::uint32_t) {
    archive(cereal::base_class<Prior>(this));
  }
};

class NonNegativePrior : public Prior {
public:
  double log_pdf(double x) const override { return x >= 0. ? 0. : -LARGE_VAL; }
  std::string get_name() const override { return "non_negative"; };
  double lower_bound() const override { return 0.; }
  double upper_bound() const override { return LARGE_VAL; }

  template <typename Archive>
  void serialize(Archive &archive, const std::uint32_t) {
    archive(cereal::base_class<Prior>(this));
  }
};

class UniformPrior : public Prior {
public:
  UniformPrior(double lower = 0., double upper = 1.)
      : lower_(lower), upper_(upper) {
    assert(upper_ > lower_);
  };

  std::string get_name() const override {
    std::ostringstream oss;
    oss << "uniform[" << lower_ << "," << upper_ << "]";
    return oss.str();
  };

  double lower_bound() const override { return lower_; }
  double upper_bound() const override { return upper_; }

  double log_pdf(double) const override { return -log(upper_ - lower_); }

  template <typename Archive>
  void serialize(Archive &archive, const std::uint32_t) {
    archive(cereal::base_class<Prior>(this), cereal::make_nvp("lower", lower_),
            cereal::make_nvp("upper", upper_));
  }

protected:
  double lower_;
  double upper_;
};

class LogScaleUniformPrior : public UniformPrior {
public:
  LogScaleUniformPrior(double lower = 1e-12, double upper = 1.e12)
      : UniformPrior(lower, upper) {
    assert(upper_ > 0.);
    assert(lower_ > 0.);
  };

  std::string get_name() const override {
    std::ostringstream oss;
    oss << "log_scale_uniform[" << lower_ << "," << upper_ << "]";
    return oss.str();
  };

  template <typename Archive>
  void serialize(Archive &archive, const std::uint32_t) {
    archive(cereal::base_class<UniformPrior>(this));
  }

  bool is_log_scale() const override { return true; };
};

class GaussianPrior : public Prior {
public:
  GaussianPrior(double mu = 0., double sigma = 1.) : mu_(mu), sigma_(sigma) {}

  bool operator==(const Prior &other) const override {
    // This seems pretty hacky but also seems to be one of the few ways
    // to override the == operator with a possibly polymorphic argument.
    if (Prior::operator==(other)) {
      auto other_cast = static_cast<const GaussianPrior &>(other);
      return other_cast.mu_ == mu_ && other_cast.sigma_ == sigma_;
    } else {
      return false;
    }
  }

  std::string get_name() const override {
    std::ostringstream oss;
    oss << "gaussian[" << mu_ << "," << sigma_ << "]";
    return oss.str();
  }

  double log_pdf(double x) const override {
    double deviation = (x - mu_) / sigma_;
    return -0.5 * (LOG_2PI_ * 2 * log(sigma_) + deviation * deviation);
  }

  template <typename Archive>
  void serialize(Archive &archive, const std::uint32_t) {
    archive(cereal::base_class<Prior>(this), cereal::make_nvp("mu", mu_),
            cereal::make_nvp("sigma", sigma_));
  }

private:
  double mu_;
  double sigma_;
};

class LogNormalPrior : public Prior {
public:
  LogNormalPrior(double mu = 0., double sigma = 1.) : mu_(mu), sigma_(sigma) {}

  bool operator==(const Prior &other) const override {
    // This seems pretty hacky but also seems to be one of the few ways
    // to override the == operator with a possibly polymorphic argument.
    if (Prior::operator==(other)) {
      auto other_cast = static_cast<const LogNormalPrior &>(other);
      return other_cast.mu_ == mu_ && other_cast.sigma_ == sigma_;
    } else {
      return false;
    }
  }

  std::string get_name() const override {
    std::ostringstream oss;
    oss << "log_normal[" << mu_ << "," << sigma_ << "]";
    return oss.str();
  }

  double log_pdf(double x) const override {
    double deviation = (log(x) - mu_) / sigma_;
    return -0.5 * LOG_2PI_ - log(sigma_) - log(x) - deviation * deviation;
  }

  template <typename Archive>
  void serialize(Archive &archive, const std::uint32_t) {
    archive(cereal::base_class<Prior>(this), cereal::make_nvp("mu", mu_),
            cereal::make_nvp("sigma", sigma_));
  }

private:
  double mu_;
  double sigma_;
};

} // namespace albatross

CEREAL_REGISTER_TYPE(albatross::UninformativePrior);
CEREAL_REGISTER_TYPE(albatross::PositivePrior);
CEREAL_REGISTER_TYPE(albatross::NonNegativePrior);
CEREAL_REGISTER_TYPE(albatross::FixedPrior);
CEREAL_REGISTER_TYPE(albatross::UniformPrior);
CEREAL_REGISTER_TYPE(albatross::LogScaleUniformPrior);
CEREAL_REGISTER_TYPE(albatross::GaussianPrior);
CEREAL_REGISTER_TYPE(albatross::LogNormalPrior);

#endif
