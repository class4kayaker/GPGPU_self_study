/**
 * Basic utilities
 */

#ifndef ADVECTION_UTILS
#define ADVECTION_UTILS

#include "../../utils/toml11/toml.hpp"
#include <cstdio>
#include <cstring>
#include <math.h>
#include <string>
#include <vector>

namespace Model_Data {

struct ProblemConfig {
  ProblemConfig();
  ProblemConfig(const unsigned int mxiters, const double epsilon,
                const std::string config_fn, const std::string output_fn);

  unsigned int mxiters;
  double epsilon;
  std::string hdf5_config_filename;
  std::string hdf5_output_filename;
};

template <typename T> struct ModelState {
  ModelState<T>();
  ModelState<T>(const size_t a_ndx, const size_t a_ndy, T a_hx, T a_hy);

  void resize(const size_t a_ndx, const size_t a_ndy, T a_hx, T a_hy);

  size_t ndx, ndy;
  T hx, hy;
  // note col major due to hdf5 quirk
  std::vector<T> k;
  std::vector<T> heat_source;
  std::vector<T> temperature;
};

const toml::value get_config_from_cli(int argc, char *argv[]);
struct ProblemConfig init_from_toml(const toml::value input_data);

} // namespace Model_Data

#endif
