#include "advection_utils.h"
#include <iostream>
#include <string>

namespace FCT_initialization {

ProblemConfig::ProblemConfig()
    : hdf5_config_filename(""), hdf5_output_filename("") {}

ProblemConfig::ProblemConfig(const size_t a_ndx, const size_t a_ndy,
                             const std::string config_fn,
                             const std::string output_fn)
    : hdf5_config_filename(config_fn), hdf5_output_filename(output_fn) {}

template <typename T>
ProblemState<T>::ProblemState()
    : ndx(0), ndy(0), hx(0), hy(0), heat_conductivity(ndx * ndy, 0.0),
      heat_source(ndx * ndy, 0.0), temperature_bnd(2 * (ndx + ndy - 2), 0.0);

template <typename T>
ProblemState<T>::ProblemState(const size_t a_ndx, const size_t a_ndy, T a_hx,
                              T a_hy)
    : ndx(a_ndx), ndy(a_ndy), hx(a_hx), hy(a_hy),
      heat_conductivity(ndx * ndy, 0.0), heat_source(ndx * ndy, 0.0),
      temperature_bnd(2 * (ndx + ndy - 2), 0.0);

template <typename T>
void ProblemState<T>::resize(const size_t a_ndx, const size_t a_ndy, T a_hx,
                             T a_hy) {
  ndx = a_ndx;
  ndy = a_ndy;
  hx = a_hx;
  hy = a_hy;

  temperature.resize(ndx*ndy)
}

template <typename T>
SolutionState<T>::SolutionState()
    : ndx(0), ndy(0), hx(0), hy(0), temperature(ndx * ndy, 0.0);
SolutionState<T>::SolutionState(const size_t a_ndx, const size_t a_ndy, T a_hx,
                                T a_hy)
    : ndx(a_ndx), ndy(a_ndy), hx(a_hx), hy(a_hy), temperature(ndx * ndy, 0.0);

template <typename T>
void SolutionState<T>::resize(const size_t a_ndx, const size_t a_ndy, T a_hx,
                             T a_hy) {
  nx = a_ndx;
  ny = a_ndy;
  hx = a_hx;
  hy = a_hy;
}

void print_config(const ProblemConfig &config) {
  std::cout << "Configuration:" << std::endl
            << "  Problem H5:  " << config.hdf5_init_fn << std::endl
            << "  Solution H5:   " << config.hdf5_end_fn << std::endl;
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
  const std::string problem_h5_fn =
      toml::find<std::string>(input_data, "Problem file");
  const std::string soln_h5_fn = toml_get_or_default<std::string>(
      toml::find(input_data, "Solution file"), "output.h5");

  struct ProblemConfig to_return(problem_h5_fn, soln_h5_fn);

  print_config(to_return);

  return to_return;
}
} // namespace FCT_initialization
