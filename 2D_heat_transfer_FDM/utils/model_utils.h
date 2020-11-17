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
  ProblemConfig(const unsigned int mxiters, const double epsilon, const std::string config_fn, const std::string output_fn);

  const unsigned int mxiters;
  const double epsilon;
  const std::string hdf5_config_filename;
  const std::string hdf5_output_filename;
};

template <typename T> struct ProblemState {
  ProblemState();
  ProblemState(const size_t a_ndx, const size_t a_ndy, T a_hx, T a_hy);

  void resize(const size_t a_ndx, const size_t a_ndy);

  size_t ndx, ndy;
  typename T hx, hy;
  // note col major due to hdf5 quirk
  std::vector<T> k;
  std::vector<T> heat_source;
  // Clockwise from upper left corner
  std::vector<T> temperature_bnd;
};

template <typename T> struct SolutionState {
  SolutionState();
  SolutionState(const size_t a_ndx, const size_t a_ndy, T a_hx, T a_hy);

  void resize(const size_t a_ndx, const size_t a_ndy);

  size_t ndx, ndy;
  typename T hx, hy;
  // note col major due to hdf5 quirk
  std::vector<T> temperature;
  std::vector<T> k;
  std::vector<T> heat_source;
}

const toml::value
get_config_from_cli(int argc, char *argv[]);
struct ProblemConfig init_from_toml(const toml::value input_data);

} // namespace Model_Data

#endif
