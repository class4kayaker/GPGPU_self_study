#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

#include <iostream>
#include <vector>

#include "../utils/advection_output.h"
#include "../utils/advection_utils.h"
#include "advection_alloc.h"

struct DeviceConfig {
  std::string device_name;
  size_t block_size;
};

void do_computation(const FCT_initialization::ProblemConfig &,
                    const DeviceConfig &, FCT_initialization::InitState &);

int main(int argc, char *argv[]) {
  FCT_initialization::ProblemConfig config;
  DeviceConfig d_config;
  {
    const auto input_config =
        FCT_initialization::get_config_from_cli(argc, argv);

    config = FCT_initialization::init_from_toml(input_config);

    const auto dev_table = toml::find(input_config, "Device");
    d_config.device_name = toml::find_or<std::string>(dev_table, "Name", "");
    d_config.block_size = toml::find_or<int>(dev_table, "Block size", 256);
  }

  FCT_initialization::InitState external_state;

  FCT_output::read_state(config.hdf5_init_fn, external_state);

  config.compute_timestep(external_state.time, external_state.dx);

  // Do computation
  auto start = std::chrono::steady_clock::now();
  { do_computation(config, d_config, external_state); }
  auto end = std::chrono::steady_clock::now();
  auto time =
      (std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
           .count()) /
      1.0e3;

  // Print results (problem size, time and bandwidth in GB/s).
  std::cout << "Time for " << config.ndt << " timestep computation " << time
            << " sec" << std::endl;

  FCT_output::write_state(config.hdf5_end_fn, external_state);

  return 0;
}

void do_computation(const FCT_initialization::ProblemConfig &config,
                    const DeviceConfig &d_config,
                    FCT_initialization::InitState &external_state) {

  // Device selection must be in cuda file

  cuda_compute_fct(d_config.device_name.c_str(), external_state.ndx,
                   d_config.block_size, external_state.u.data(), config.ndt,
                   config.dt, external_state.dx, config.a);

  external_state.time = config.end_time;
}
