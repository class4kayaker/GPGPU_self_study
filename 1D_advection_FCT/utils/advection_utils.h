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

namespace FCT_initialization {

struct ProblemConfig {
  ProblemConfig(const double a, const double sigma, const double end_time,
                const std::string init_fn, const std::string end_fn);

  void compute_timestep(const double curr_time, const double dx);

  const double a;
  const double sigma;
  const std::string hdf5_init_fn;
  const double end_time;
  const std::string hdf5_end_fn;

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

void sine_init(InitState &state, const ProblemConfig &config,
               const double time);

struct ProblemConfig parse_args(int argc, char *argv[]);
} // namespace FCT_initialization

#endif
