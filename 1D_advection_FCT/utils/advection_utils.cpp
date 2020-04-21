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

StepState::StepState(const size_t ndx)
    : ndx(ndx), u_state(ndx + 4, 0.0), flux_low(ndx + 1, 0.0),
      flux_high(ndx + 1, 0.0), adiff_flux(ndx + 1, 0.0), flux_c(ndx + 1, 0.0) {}

void sine_init(InitState &state, const ProblemConfig &config,
               const double time) {
  state.time = time;
  state.dx = 1.0 / state.ndx;

  for (int i = 0; i < state.ndx; ++i) {
    double px = (i + 0.5) * state.dx;
    state.u[i] = sin((px - time * config.a) * 2 * PI);
  }
}

void print_config(const ProblemConfig &config) {
  std::cout << "Configuration:" << std::endl
            << "  Velocity: " << config.a << std::endl
            << "  Sigma:    " << config.sigma << std::endl
            << "  End Time: " << config.end_time << std::endl
            << "  Init H5:  " << config.hdf5_init_fn << std::endl
            << "  End H5:   " << config.hdf5_end_fn << std::endl;
}

struct ProblemConfig parse_args(int argc, char *argv[]) {
  double a = 3.0;
  double sigma = 0.9;
  double end_time = 0.0;
  std::string init_h5_fn = "init_state.h5";
  std::string end_h5_fn = "end_state.h5";
  std::string device_name = "";

  // Read command line arguments.
  for (int i = 0; i < argc; i++) {
    std::string arg_i = argv[i];
    if ((arg_i == "-I")) {
      if (i + 1 >= argc) {
        std::cout << "  Value required for \"" << arg_i << "\"" << std::endl;
        exit(1);
      }
      init_h5_fn = std::string(argv[++i]);
    } else if ((strcmp(argv[i], "-O") == 0)) {
      if (i + 1 >= argc) {
        std::cout << "  Value required for \"" << arg_i << "\"" << std::endl;
        exit(1);
      }
      end_h5_fn = std::string(argv[++i]);
    } else if ((strcmp(argv[i], "-T") == 0)) {
      if (i + 1 >= argc) {
        std::cout << "  Value required for \"" << arg_i << "\"" << std::endl;
        exit(1);
      }
      end_time = atof(argv[++i]);
    } else if ((strcmp(argv[i], "-dev") == 0)) {
      if (i + 1 >= argc) {
        std::cout << "  Value required for \"" << arg_i << "\"" << std::endl;
        exit(1);
      }
      device_name = std::string(argv[++i]);
    } else if ((strcmp(argv[i], "-sigma") == 0)) {
      if (i + 1 >= argc) {
        std::cout << "  Value required for \"" << arg_i << "\"" << std::endl;
        exit(1);
      }
      sigma = atof(argv[++i]);
    } else if ((strcmp(argv[i], "-vel") == 0)) {
      if (i + 1 >= argc) {
        std::cout << "  Value required for \"" << arg_i << "\"" << std::endl;
        exit(1);
      }
      a = atof(argv[++i]);
    } else if ((strcmp(argv[i], "-h") == 0) ||
               (strcmp(argv[i], "-help") == 0)) {
      printf("  -help (-h):            print this message\n\n");
      exit(1);
    }
  }

  struct ProblemConfig to_return(a, sigma, end_time, init_h5_fn, end_h5_fn);

  print_config(to_return);

  return to_return;
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

template double toml_get_or_default<double>(toml::value file_value,
                                            double other);
template std::string toml_get_or_default<std::string>(toml::value file_value,
                                                      std::string other);

struct ProblemConfig init_from_toml(const toml::value input_data) {
  const double a = toml_get_or_default<double>(
      toml::find(input_data, "Velocity"), 3.0);
  const double sigma = toml_get_or_default<double>(
      toml::find(input_data, "CFL number"), 0.9);
  const double end_time = toml::find<double>(input_data, "End time");
  const std::string init_h5_fn =
      toml::find<std::string>(input_data, "Init file");
  const std::string end_h5_fn = toml_get_or_default<std::string>(
      toml::find(input_data, "Output file"), "output.h5");

  struct ProblemConfig to_return(a, sigma, end_time, init_h5_fn, end_h5_fn);

  print_config(to_return);

  return to_return;
}
} // namespace FCT_initialization
