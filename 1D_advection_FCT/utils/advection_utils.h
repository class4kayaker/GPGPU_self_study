/**
 * Basic utilities
 */

#ifndef ADVECTION_UTILS
#define ADVECTION_UTILS

#include <cstdio>
#include <cstring>
#include <math.h>
#include <vector>

namespace FCT_initialization {

struct ProblemConfig {
  ProblemConfig(const int ndx, const double a, const double sigma,
                const double init_time, const double end_time);

  void compute_timestep(const double dx);

  const int ndx;
  const double a;
  const double sigma;
  const double init_time;
  const double end_time;

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

void check_bounds(const ProblemConfig &config);
} // namespace FCT_initialization

#endif
