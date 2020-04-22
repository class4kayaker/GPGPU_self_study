#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

#include <iostream>
#include <vector>

#include "../utils/advection_output.h"
#include "../utils/advection_utils.h"
#include "advection_alloc.h"

void do_computation(const FCT_initialization::ProblemConfig &,
                    FCT_initialization::InitState &);

int main(int argc, char *argv[]) {
  const auto input_config = FCT_initialization::get_config_from_cli(argc, argv);

  FCT_initialization::ProblemConfig config =
      FCT_initialization::init_from_toml(input_config);

  const std::string sel_device_name = toml::find_or(input_config, "Device", "");

  FCT_initialization::InitState external_state;

  FCT_output::read_state(config.hdf5_init_fn, external_state);

  config.compute_timestep(external_state.time, external_state.dx);

  // Do computation
  auto start = std::chrono::steady_clock::now();
  { do_computation(config, external_state); }
  auto end = std::chrono::steady_clock::now();
  auto time =
      (std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
           .count()) /
      1.0e3;

  // Print results (problem size, time and bandwidth in GB/s).
  std::cout << "TIme for " << config.ndt << " timestep computation " << time
            << " sec" << std::endl;

  FCT_output::write_state(config.hdf5_end_fn, external_state);

  return 0;
}

void do_computation(const FCT_initialization::ProblemConfig &config,
                    FCT_initialization::InitState &external_state) {

  cuda_compute_fct(external_state.ndx, external_state.u.data(), config.ndt,
                   config.dt, external_state.dx, config.a);

  external_state.time = config.end_time;
}
