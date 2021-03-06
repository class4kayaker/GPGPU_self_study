#include "model_utils.h"
#include <iostream>
#include <string>

namespace Model_Data {

ProblemConfig::ProblemConfig()
    : mxiters(30), epsilon(1e-10), hdf5_config_filename(""),
      hdf5_output_filename("") {}
ProblemConfig::ProblemConfig(const unsigned int a_mxiters,
                             const double a_epsilon,
                             const std::string config_fn,
                             const std::string output_fn)
    : mxiters(a_mxiters), epsilon(a_epsilon), hdf5_config_filename(config_fn),
      hdf5_output_filename(output_fn) {}

template <typename T>
ModelState<T>::ModelState()
    : ndx(2), ndy(2), hx(0), hy(0), k((ndx + 1) * (ndy + 1), 0.0),
      heat_source((ndx + 1) * (ndy + 1), 0.0),
      temperature((ndx + 1) * (ndy + 1), 0.0){};
template <typename T>
ModelState<T>::ModelState(const size_t a_ndx, const size_t a_ndy, T a_hx,
                          T a_hy)
    : ndx(a_ndx), ndy(a_ndy), hx(a_hx), hy(a_hy), k((ndx + 1) * (ndy + 1), 0.0),
      heat_source((ndx + 1) * (ndy + 1), 0.0),
      temperature((ndx + 1) * (ndy + 1), 0.0){};

template <typename T>
void ModelState<T>::resize(const size_t a_ndx, const size_t a_ndy, T a_hx,
                           T a_hy) {
  ndx = a_ndx;
  ndy = a_ndy;
  hx = a_hx;
  hy = a_hy;

  k.resize((ndx + 1) * (ndy + 1));
  heat_source.resize((ndx + 1) * (ndy + 1));
  temperature.resize((ndx + 1) * (ndy + 1));
}

void print_config(const ProblemConfig &config) {
  std::cout << "Configuration:" << std::endl
            << "  CG epsilon: " << config.epsilon << std::endl
            << "  Max CG iters: " << config.mxiters << std::endl
            << "  Problem H5:  " << config.hdf5_config_filename << std::endl
            << "  Solution H5:   " << config.hdf5_output_filename << std::endl;
}

const toml::value get_config_from_cli(int argc, char *argv[]) {

  if (argc != 2) {
    std::cout << "More than one argument provided, only path to config file "
                 "should be included."
              << std::endl;
    exit(1);
  }

  const std::string config_fn = argv[1];

  return toml::parse(config_fn);
}

template <typename T> T toml_get_or_default(toml::value file_value, T other) {
  if (file_value.is_uninitialized()) {
    return other;
  } else {
    return toml::get<T>(file_value);
  }
}

template int toml_get_or_default<int>(toml::value file_value, int other);
template double toml_get_or_default<double>(toml::value file_value,
                                            double other);
template std::string toml_get_or_default<std::string>(toml::value file_value,
                                                      std::string other);

struct ProblemConfig init_from_toml(const toml::value input_data) {
  const unsigned int mxiters =
      toml::find_or<int>(input_data, "Max CG iters", 30);
  const double epsilon = toml::find_or<double>(input_data, "CG epsilon", 1e-10);
  const std::string problem_h5_fn =
      toml::find<std::string>(input_data, "Problem file");
  const std::string soln_h5_fn =
      toml::find_or<std::string>(input_data, "Solution file", "output.h5");

  struct ProblemConfig to_return(mxiters, epsilon, problem_h5_fn, soln_h5_fn);

  print_config(to_return);

  return to_return;
}

template class ModelState<double>;

} // namespace Model_Data
