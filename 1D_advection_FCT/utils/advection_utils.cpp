#include "advection_utils.h"
#include <iostream>
#include <string>

namespace FCT_initialization {
const double PI = 3.141592653589793;

ProblemConfig::ProblemConfig(const double a, const double sigma,
                             const double end_time, const std::string init_fn,
                             const std::string end_fn)
    : a(a), sigma(sigma), end_time(end_time), hdf5_init_fn(init_fn),
      hdf5_end_fn(end_fn) {}

void ProblemConfig::compute_timestep(const double curr_time, const double dx) {
  ndt = std::ceil(a * (end_time - curr_time) / (sigma * dx));
  dt = (end_time - curr_time) / ndt;
}

InitState::InitState() : ndx(0), time(0.0), u(ndx, 0.0) {}
InitState::InitState(const size_t ndx) : ndx(ndx), time(0.0), u(ndx, 0.0) {}

void sine_init(InitState &state, const ProblemConfig &config,
               const double time) {
  state.time = time;
  state.dx = 1.0 / state.ndx;

  for (int i = 0; i < state.ndx; ++i) {
    double px = (i + 0.5) * state.dx;
    state.u[i] = sin((px - time * config.a) * 2 * PI);
  }
}

struct ProblemConfig parse_args(int argc, char *argv[]) {
  const double a = 3.0;
  double sigma = 0.9;
  double end_time = 0.0;
  std::string init_h5_fn = "init_state.h5";
  std::string end_h5_fn = "end_state.h5";

  // Read command line arguments.
  for (int i = 0; i < argc; i++) {
    if ((strcmp(argv[i], "-I") == 0)) {
      init_h5_fn = std::string(argv[++i]);
      std::cout << "  User init file is " << init_h5_fn << std::endl;
    } else if ((strcmp(argv[i], "-O") == 0)) {
      end_h5_fn = std::string(argv[++i]);
      std::cout << "  User end file is " << end_h5_fn << std::endl;
    } else if ((strcmp(argv[i], "-T") == 0)) {
      end_time = atof(argv[++i]);
      std::cout << "  User end time is "  << end_time << std::endl;
    } else if ((strcmp(argv[i], "-h") == 0) ||
               (strcmp(argv[i], "-help") == 0)) {
      printf("  -help (-h):            print this message\n\n");
      exit(1);
    }
  }

  struct ProblemConfig to_return(a, sigma, end_time, init_h5_fn, end_h5_fn);

  return to_return;
}
} // namespace FCT_initialization
