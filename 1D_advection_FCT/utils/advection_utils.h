/**
 * Basic utilities
 */

#ifndef ADVECTION_UTILS
#define ADVECTION_UTILS

#include <cstdio>
#include <cstring>
#include <math.h>
#include <string>
#include <vector>
#include "../../utils/toml11/toml.hpp"

namespace FCT_initialization {

struct ProblemConfig {
  ProblemConfig();
  ProblemConfig(const double a, const double sigma, const double end_time,
                const std::string init_fn, const std::string end_fn);

  void compute_timestep(const double curr_time, const double dx);

  double a;
  double sigma;
  std::string hdf5_init_fn;
  double end_time;
  std::string hdf5_end_fn;

  // Derived values
  int ndt;
  double dt;
};

struct InitState {
  InitState();
  InitState(const size_t ndx);

  size_t ndx;
  double dx;
  double time;
  std::vector<double> u;
};

struct StepState {
    StepState(const size_t ndx);

    size_t ndx;
    std::vector<double> u_state;
    std::vector<double> flux_low;
    std::vector<double> flux_high;
    std::vector<double> adiff_flux;
    std::vector<double> flux_c;
};

void sine_init(InitState &state, const ProblemConfig &config,
               const double time);

struct ProblemConfig parse_args(int argc, char *argv[]);

const toml::value get_config_from_cli(int argc, char *argv[]);
struct ProblemConfig init_from_toml(const toml::value input_data);
} // namespace FCT_initialization

#endif
