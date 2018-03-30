#include <random>
#include <fstream>
#include <sstream>
#include <iostream>
#include <Eigen/Core>
#include "csv.h"
#include "gflags/gflags.h"

#include "gp/gp.h"
#include "core/model.h"
#include "covariance_functions/covariance_functions.h"

DEFINE_string(input, "", "path to csv containing input data.");
DEFINE_string(output, "", "path where predictions will be written in csv.");
DEFINE_string(n, "50", "number of training points to use.");

namespace albatross {

class SlopeTerm : public CovarianceBase {
 public:
  SlopeTerm(double sigma_slope = 0.1) {
    this->params_["sigma_slope"] = sigma_slope;
  };

  ~SlopeTerm(){};

  std::string get_name() const { return "slope_term"; }

  double operator()(const double &x,
                    const double &y) const {
    double sigma_slope = this->params_.at("sigma_slope");
    return sigma_slope * sigma_slope * x * y;
  }

};

class ScalarDistance : public DistanceMetric {
 public:

  std::string get_name() const { return "scalar_distance"; }

  double operator()(const double &x,
                    const double &y) const {
    return fabs(x - y);
  }
};

template <typename CovFunc>
GaussianProcessRegression<CovFunc, double> gp_from_covariance(
    CovFunc covariance_function) {
  return GaussianProcessRegression<CovFunc, double>(covariance_function);
};


}


std::vector<double> random_points_on_line(const int n,
                                          const double low,
                                          const double high) {
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(low, high);

  std::vector<double> xs;
  for (int i = 0; i < n; i++) {
    xs.push_back(distribution(generator));
  }
  return xs;
};


std::vector<double> uniform_points_on_line(const int n,
                                           const double low,
                                           const double high) {
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(low, high);

  std::vector<double> xs;
  for (int i = 0; i < n; i++) {
    double ratio = (double) i / (double) (n - 1);
    xs.push_back(low + ratio * (high - low));
  }
  return xs;
};


double truth(double x) {
  double a = sqrt(2.);
  double b = 3.14159;
  return x * a + b;
}


albatross::RegressionDataset<double> create_train_data(const int n,
                       const double low,
                       const double high,
                       const double measurement_noise) {
  auto xs = random_points_on_line(n, low, high);

  std::default_random_engine generator;
  std::normal_distribution<double> noise_distribution(0., measurement_noise);

  Eigen::VectorXd ys(n);

  for (int i = 0; i < n; i++) {
    double noise = noise_distribution(generator);
    ys[i] = (truth(xs[i]) + noise);
  }

  return albatross::RegressionDataset<double>(xs, ys);
}


albatross::RegressionDataset<double> read_linear_input(std::string file_path) {
  std::vector<double> xs;
  std::vector<double> ys;

  io::CSVReader<2> file_in(file_path);

  file_in.read_header(io::ignore_extra_column, "x", "y");
  double x, y;
  bool more_to_parse = true;
  while (more_to_parse) {
    more_to_parse = file_in.read_row(x, y);

    xs.push_back(x);
    ys.push_back(y);
  }
  Eigen::Map<Eigen::VectorXd> eigen_ys(&ys[0],
                                           static_cast<int>(ys.size()));
  return albatross::RegressionDataset<double>(xs, eigen_ys);
}

inline bool file_exists (const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  int n = std::stoi(FLAGS_n);
  const double low = -3.;
  const double high = 13.;
  const double meas_noise = 1.;

  /*
   * Either read the input data from file, or if it doesn't exist
   * generate new input data and write it to file.
   */
  if (file_exists(FLAGS_input)) {
    std::cout << "reading data from : " << FLAGS_input << std::endl;
  } else {
    std::cout << "creating training data and writing it to : " << FLAGS_input << std::endl;
    auto data = create_train_data(n, low, high, meas_noise);
    std::ofstream train;
    train.open(FLAGS_input);
    train << "x,y" << std::endl;
    for (int i = 0; i < static_cast<int>(data.predictors.size()); i++) {
      train << data.predictors[i] << ", " << data.targets[i] << std::endl;
    }
  }
  auto data = read_linear_input(FLAGS_input);

  using namespace albatross;
  using Mean = ConstantMean;
  using Slope = SlopeTerm;
  using Noise = IndependentNoise<double>;

  auto mean_term = Mean(10.);
  CovarianceFunction<Mean> mean = {mean_term};
  CovarianceFunction<Slope> slope = {Slope(10.)};
  CovarianceFunction<Noise> noise = {Noise(meas_noise)};

  auto linear_model = mean + slope + noise;

  std::cout << "Using Model:" << std::endl;
  std::cout << linear_model.to_string() << std::endl;

  auto model = gp_from_covariance(linear_model);

  model.fit(data);

  const auto mean_terms = mean_term.get_state_space_representation(data.predictors);

  auto state = model.inspect(mean_terms);
  std::cout << state.mean << " +/- " << state.covariance << std::endl;
}
