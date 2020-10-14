#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sys/time.h>

#include <vector>

#include "../utils/advection_output.h"
#include "../utils/advection_utils.h"

struct DeviceConfig {
  std::string device_name;
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
  std::cout << "TIme for " << config.ndt << " timestep computation " << time
            << " sec" << std::endl;

  FCT_output::write_state(config.hdf5_end_fn, external_state);

  return 0;
}

void do_computation(const FCT_initialization::ProblemConfig &config,
                    const DeviceConfig &d_config,
                    FCT_initialization::InitState &external_state) {

  const size_t ndx = external_state.ndx;
  const size_t halo_state = ndx+4;
  const size_t flux_size = ndx+1;
  double u_state[halo_state];
  double flux_low[flux_size];
  double flux_high[flux_size];
  double adiff_flux[flux_size];
  double flux_c[flux_size];

  // Initialize data
  for (size_t i = 0; i < ndx; ++i) {
    u_state[i + 2] = external_state.u[i];
  }

  #pragma acc data create(flux_low[0:flux_size], flux_high[0:flux_size], adiff_flux[0:flux_size], flux_c[0:flux_size]) copy(u_state[0:halo_state])
  {
    // Copy relevant configs out of structs
    const double a_vel = config.a, dt = config.dt, dx = external_state.dx;

    for (unsigned int timestep = 0; timestep < config.ndt; ++timestep) {
      // Set BCs
      #pragma acc parallel loop
      for(size_t idx=0; idx<4;++idx){
          const size_t dst_index = (idx < 2) ? idx : (ndx + idx);
          const size_t src_index = (idx < 2) ? (ndx + idx) : idx;
          u_state[dst_index] = u_state[src_index];
      }

      // Compute fluxes
      #pragma acc parallel loop
      for(size_t index=0; index<flux_size; ++index){
          flux_low[index] = a_vel * u_state[index + 1];
      }

      #pragma acc parallel loop
      for(size_t index=0; index<flux_size; ++index){
          const double sigma_i = (a_vel * dt / dx);
          flux_high[index] = a_vel * 0.5 *
              ((1 + sigma_i) * u_state[index + 1] +
               (1 - sigma_i) * u_state[index + 2]);
      }

      // Compute diff flux
      #pragma acc parallel loop
      for(size_t index=0; index<flux_size; ++index){
          adiff_flux[index] = flux_high[index] - flux_low[index];
      }

      // Do low update
      #pragma acc parallel loop
      for(size_t index=0; index<ndx; ++index){
          const double dtdx = (dt / dx);
          u_state[index + 2] +=
              dtdx * (flux_low[index] - flux_low[index + 1]);
      }

      // Set BC cells
      #pragma acc parallel loop
      for(size_t idx=0; idx<4;++idx){
          const size_t dst_index = (idx < 2) ? idx : (ndx + idx);
          const size_t src_index = (idx < 2) ? (ndx + idx) : idx;
          u_state[dst_index] = u_state[src_index];
      }

      // Calc FCT flux
      #pragma acc parallel loop
      for(size_t index=0; index<flux_size; ++index){
          const double dxdt = (dx / dt);
          const double sign_a = copysign(1.0, adiff_flux[index]);
          const double left_d = u_state[index + 1] - u_state[index + 0];
          const double right_d = u_state[index + 3] - u_state[index + 2];
          flux_c[index] =
              sign_a *
              std::max(
                      0.0, std::min(std::min(sign_a * dxdt * left_d,
                              sign_a * dxdt * right_d),
                          fabs(adiff_flux[index])));
      }
      // Do full update
      #pragma acc parallel loop
      for(size_t index=0; index<ndx; ++index){
          const double dtdx = (dt / dx);
          u_state[index + 2] +=
              dtdx * (flux_c[index] - flux_c[index + 1]);
      }
    }
  }

  external_state.time = config.end_time;

  // Copy out into u
  for (size_t i = 0; i < ndx; ++i) {
    external_state.u[i] = u_state[i + 2];
  }
}
