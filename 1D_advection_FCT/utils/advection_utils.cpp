#include "advection_utils.h"

namespace FCT_initialization {
const double PI = 3.141592653589793;

ProblemConfig::ProblemConfig(const int ndx, const double a, const double sigma,
                             const double init_time, const double end_time)
    : ndx(ndx), a(a), sigma(sigma), init_time(init_time), end_time(end_time) {}

void ProblemConfig::compute_timestep(const double dx) {
  ndt = std::ceil(a * (end_time - init_time) / (sigma * dx));
  dt = (end_time - init_time) / ndt;
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
  int ndx = -1;
  const double a = 3.0;
  double sigma = 0.9;
  double end_time = 0.0;

  // Read command line arguments.
  for (int i = 0; i < argc; i++) {
    if ((strcmp(argv[i], "-ndx") == 0)) {
      ndx = atoi(argv[++i]);
      printf("  User ndx is %d\n", ndx);
    } else if ((strcmp(argv[i], "-T") == 0)) {
      end_time = atof(argv[++i]);
      printf("  User end_time is %g\n", end_time);
    } else if ((strcmp(argv[i], "-h") == 0) ||
               (strcmp(argv[i], "-help") == 0)) {
      printf("  -help (-h):            print this message\n\n");
      exit(1);
    }
  }

  struct ProblemConfig to_return(ndx, a, sigma, 0.0, end_time);

  check_bounds(to_return);

  return to_return;
}

void check_bounds(const ProblemConfig &config) {
  if (config.ndx <= 0) {
    printf("NDX <= 0, cannot run computation\n");
    exit(1);
  }
}
} // namespace FCT_initialization
