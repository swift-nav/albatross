#include <benchmark/benchmark.h>

#include <albatross/CovarianceFunctions>

#include "./uninlineable.h"

static constexpr double length_scale = 0.5;
static constexpr double sigma = 0.1;

#define BENCHMARK_SIZES(name) \
  BENCHMARK(name)->Arg(8)->Arg(32)->Arg(256)->Arg(1024)->Arg(4096)

static inline std::vector<double> random_vector(std::size_t length,
                                                std::size_t seed) {
  std::random_device random_device{};
  std::mt19937 generator{random_device()};
  generator.seed(seed);
  std::uniform_real_distribution<> dist{-1., 1.};
  std::vector<double> x(length);
  std::generate(x.begin(), x.end(),
                [&dist, &generator]() { return dist(generator); });
  return x;
}

// static inline auto random_index_generator(std::size_t seed, Eigen::Index size) {
//   std::random_device random_device{};
//   std::mt19937 generator{random_device()};
//   generator.seed(seed);
//   std::uniform_int_distribution<Eigen::Index> dist{0, size - 1};
//   return [dist, generator]() mutable { return dist(generator); };
// }

static void BM_RBF(benchmark::State &state) {
  const auto x = random_vector(state.range(0), 22);
  const albatross::SquaredExponential<albatross::EuclideanDistance> cov{
      length_scale, sigma};
  for (auto _ : state) {
    auto mat = compute_covariance_matrix(cov, x);
  }
}

BENCHMARK_SIZES(BM_RBF);

static void BM_FullBlockRBF(benchmark::State &state) {
  const auto x = random_vector(state.range(0), 22);
  for (auto _ : state) {
    Eigen::MatrixXd mat =
        albatross::block_squared_exponential_full(x, x, length_scale, sigma);
  }
}

BENCHMARK_SIZES(BM_FullBlockRBF);

static void BM_FullBlockRowsRBF(benchmark::State &state) {
  const auto x = random_vector(state.range(0), 22);
  for (auto _ : state) {
    Eigen::MatrixXd mat = albatross::block_squared_exponential_full_rows(
        x, x, length_scale, sigma);
  }
}

BENCHMARK_SIZES(BM_FullBlockRowsRBF);

static void BM_FullBlockVecsRBF(benchmark::State &state) {
  auto x = random_vector(state.range(0), 22);
  const auto xsize = static_cast<Eigen::Index>(x.size());
  std::random_device dev{};
  std::seed_seq rng_seed{dev(), dev(), dev(), dev(), dev(), dev(), dev(), dev()};
  std::mt19937 rng{rng_seed};
  for (auto _ : state) {
    Eigen::MatrixXd mat(xsize, xsize);
    benchmark::DoNotOptimize(mat.data());
    mat = albatross::block_squared_exponential_full_vecs_uninlineable(
        x, x, length_scale, sigma);
    benchmark::ClobberMemory();

    // Now do a randomized swizzle with outputs that depend on the
    // previous step and on `mat`.
    state.PauseTiming();
    Eigen::Map<Eigen::VectorXd> xvec(x.data(), xsize);
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> permute(xsize);
    permute.setIdentity();
    std::shuffle(permute.indices().data(), permute.indices().data() + permute.indices().size(), rng);
    xvec += mat * permute * xvec;
    state.ResumeTiming();
  }
}

BENCHMARK_SIZES(BM_FullBlockVecsRBF);

static void BM_ByColsRBF(benchmark::State &state) {
  const auto x = random_vector(state.range(0), 22);
  for (auto _ : state) {
    Eigen::MatrixXd mat =
        albatross::block_squared_exponential_columns(x, x, length_scale, sigma);
  }
}

BENCHMARK_SIZES(BM_ByColsRBF);

static void BM_ColMajorRBF(benchmark::State &state) {
  const auto x = random_vector(state.range(0), 22);
  for (auto _ : state) {
    Eigen::MatrixXd mat =
        albatross::squared_exponential_column_major(x, x, length_scale, sigma);
  }
}

BENCHMARK_SIZES(BM_ColMajorRBF);

static void BM_RowMajorRBF(benchmark::State &state) {
  const auto x = random_vector(state.range(0), 22);
  for (auto _ : state) {
    Eigen::MatrixXd mat =
        albatross::squared_exponential_row_major(x, x, length_scale, sigma);
  }
}

BENCHMARK_SIZES(BM_RowMajorRBF);

BENCHMARK_MAIN();