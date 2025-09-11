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

constexpr double LOG_2_ = 0.6931471805599453;
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
  virtual ~Prior() {}
  virtual double log_pdf(double x) const = 0;
  virtual std::string get_name() const = 0;
  virtual double lower_bound() const { return -LARGE_VAL; }
  virtual double upper_bound() const { return LARGE_VAL; }
  virtual bool is_log_scale() const { return false; };
  virtual bool is_fixed() const { return false; }
  bool operator==(const Prior &other) const {
    return get_name() == other.get_name();
  }
  bool operator!=(const Prior &other) { return !((*this) == other); }
};

class UninformativePrior : public Prior {
public:
  std::string get_name() const override { return "uninformative"; };
  double log_pdf(double) const override { return 0.; }
};

class FixedPrior : public Prior {
public:
  std::string get_name() const override { return "fixed"; };
  double log_pdf(double) const override { return 0.; }

  bool is_fixed() const override { return true; }
};

class PositivePrior : public Prior {
public:
  double log_pdf(double x) const override { return x > 0. ? 0. : -LARGE_VAL; }
  std::string get_name() const override { return "positive"; };
  double lower_bound() const override {
    return std::numeric_limits<double>::epsilon();
  }
  double upper_bound() const override { return LARGE_VAL; }
};

class NonNegativePrior : public Prior {
public:
  double log_pdf(double x) const override { return x >= 0. ? 0. : -LARGE_VAL; }
  std::string get_name() const override { return "non_negative"; };
  double lower_bound() const override { return 0.; }
  double upper_bound() const override { return LARGE_VAL; }
};

class UniformPrior : public Prior {
public:
  UniformPrior(double lower = 0., double upper = 1.)
      : lower_(lower), upper_(upper) {
    ALBATROSS_ASSERT(upper_ > lower_);
  };

  std::string get_name() const override {
    std::ostringstream oss;
    oss << "uniform[" << lower_ << "," << upper_ << "]";
    return oss.str();
  };

  double lower_bound() const override { return lower_; }
  double upper_bound() const override { return upper_; }

  double log_pdf(double x) const override {
    if (x >= lower_ && x <= upper_) {
      return -log(upper_ - lower_);
    } else {
      return -LARGE_VAL;
    }
  }

  double lower_;
  double upper_;
};

class LogScaleUniformPrior : public UniformPrior {
public:
  LogScaleUniformPrior(double lower = 1e-12, double upper = 1.e12)
      : UniformPrior(lower, upper) {
    ALBATROSS_ASSERT(upper_ > 0.);
    ALBATROSS_ASSERT(lower_ > 0.);
  };

  std::string get_name() const override {
    std::ostringstream oss;
    oss << "log_scale_uniform[" << lower_ << "," << upper_ << "]";
    return oss.str();
  };

  bool is_log_scale() const override { return true; };
};

class GaussianPrior : public Prior {
public:
  GaussianPrior(double mu = 0., double sigma = 1.) : mu_(mu), sigma_(sigma) {}

  bool operator==(const GaussianPrior &other) const {
    return other.mu_ == mu_ && other.sigma_ == sigma_;
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

  double mu_;
  double sigma_;
};

class PositiveGaussianPrior : public Prior {
public:
  PositiveGaussianPrior(double mu = 0., double sigma = 1.)
      : mu_(mu), sigma_(sigma) {}

  bool operator==(const PositiveGaussianPrior &other) const {
    return other.mu_ == mu_ && other.sigma_ == sigma_;
  }

  double lower_bound() const override { return 0.; }

  double upper_bound() const override { return 10. * sigma_; }

  std::string get_name() const override {
    std::ostringstream oss;
    oss << "positive_gaussian[" << mu_ << "," << sigma_ << "]";
    return oss.str();
  }

  double log_pdf(double x) const override {
    double deviation = (x - mu_) / sigma_;
    return -0.5 * (LOG_2PI_ * 2 * log(sigma_) + deviation * deviation) + LOG_2_;
  }

  double mu_;
  double sigma_;
};

class LogNormalPrior : public Prior {
public:
  LogNormalPrior(double mu = 0., double sigma = 1.) : mu_(mu), sigma_(sigma) {}

  bool operator==(const LogNormalPrior &other) const {
    return other.mu_ == mu_ && other.sigma_ == sigma_;
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

  double mu_;
  double sigma_;
};

// NOTE: Order here is very important for backward compatible serialization.
using PossiblePriors =
    variant<UninformativePrior, FixedPrior, NonNegativePrior, PositivePrior,
            UniformPrior, LogScaleUniformPrior, GaussianPrior, LogNormalPrior,
            PositiveGaussianPrior>;

class PriorContainer : public Prior {
public:
  PriorContainer() : priors_(UninformativePrior()) {}

  template <typename PriorType,
            typename std::enable_if<
                is_in_variant<PriorType, PossiblePriors>::value, int>::type = 0>
  PriorContainer(const PriorType &prior) : priors_(prior) {}

  template <
      typename PriorType,
      typename std::enable_if<!is_in_variant<PriorType, PossiblePriors>::value,
                              int>::type = 0>
  PriorContainer(const PriorType &prior ALBATROSS_UNUSED) {
    static_assert(delay_static_assert<PriorType>::value,
                  "Attempt to initialize a prior which is not one of the types "
                  "see PossiblePriors");
  }

  double log_pdf(double x) const override {
    return priors_.match([&x](const auto &p) { return p.log_pdf(x); });
  }

  std::string get_name() const override {
    return priors_.match([](const auto &p) { return p.get_name(); });
  }

  double lower_bound() const override {
    return priors_.match([](const auto &p) { return p.lower_bound(); });
  }

  double upper_bound() const override {
    return priors_.match([](const auto &p) { return p.upper_bound(); });
  }

  bool is_log_scale() const override {
    return priors_.match([](const auto &p) { return p.is_log_scale(); });
  }

  bool is_fixed() const override {
    return priors_.match([](const auto &p) { return p.is_fixed(); });
  }

  bool operator==(const PriorContainer &other) const {
    return priors_ == other.priors_;
  }

  template <typename PriorType> void operator=(const PriorType &prior) {
    priors_ = prior;
  }

  PossiblePriors priors_;
};

} // namespace albatross

#endif
