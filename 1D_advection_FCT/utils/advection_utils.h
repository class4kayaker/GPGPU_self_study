/**
 * Basic utilities
 */

#include <math.h>
#include <vector>

namespace FCT_initialization {
const double PI = 3.141592653589793;

struct ProblemConfig {
  ProblemConfig(const int ndx, const double a, const double sigma, const double init_time, const double end_time);

  compute_timestep(const double dx);

  const int ndx;
  const double a;
  const double sigma;
  const double init_time;
  const double end_time;

  // Derived values
  double ndt;
  double dt;
}

ProblemConfig::ProblemConfig(const int ndx, const double a, const double sigma, const double init_time, const double end_time)
    : ndx(ndx), a(a), sigma(sigma), init_time(init_time), end_time(end_time) {
}

ProblemConfig::compute_timestep(const double dx){
    ndt = std::ceil(a*(end_time-init_time)/(sigma*dx));
    dt = (end_time-init_time)/ndt;
}

struct InitState {
  InitState(const int ndx);

  const int ndx;
  double dx;
  double time;
  std::vector<double> px;
  std::vector<double> u;
}

InitState::InitState(const int ndx)
    : ndx(ndx), time(0.0), px(ndx), u(ndx) {
}

void sine_init(InitState &state, const ProblemConfig &config,
               const double time) {
  state.time = time;
  state.dx = 1.0/ndx;

  for (const int i = 0; i < state.ndx; ++i) {
    state.px[i] = (i + 0.5) * state.dx;
    state.u[i] = sin((state.px[i] - time * config.a) * 2 * PI);
  }
}

struct ProblemConfig parse_args(int argc, char *argv[]) {
  int ndx = -1;
  const double a = 3.0;
  double sigma = 0.9;

  // Read command line arguments.
  for (int i = 0; i < argc; i++) {
    if ((strcmp(argv[i], "-ndx") == 0)) {
      ndx = atoi(argv[++i]);
      printf("  User ndx is %d\n", ndx);
    } else if ((strcmp(argv[i], "-T") == 0)) {
      end_time = atof(argv[++i]);
      printf("  User end_time is %d\n", M);
    } else if ((strcmp(argv[i], "-h") == 0) ||
               (strcmp(argv[i], "-help") == 0)) {
      printf("  -help (-h):            print this message\n\n");
      exit(1);
    }
  }

  struct ProblemConfig to_return(ndx, a, sigma);

  return to_return;
}
} // namespace FCT_initialization
