#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

#include <CL/sycl.hpp>
#include <vector>

#include "../utils/advection_output.h"
#include "../utils/advection_utils.h"

void do_computation(const FCT_initialization::ProblemConfig &,
                    FCT_initialization::InitState &);

int main(int argc, char *argv[]) {
  FCT_initialization::ProblemConfig config =
      FCT_initialization::parse_args(argc, argv);

  FCT_initialization::InitState external_state;

  FCT_output::read_state(config.hdf5_init_fn, external_state);

  config.compute_timestep(external_state.time, external_state.dx);

  const int ndx = external_state.ndx;

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

class SetStateBC;
class SetStateBC2;
class CalcLowFlux;
class CalcHighFlux;
class CalcADiffFlux;
class LowFluxUpdate;
class CalcFCTFLux;
class FullUpdate;

void do_computation(const FCT_initialization::ProblemConfig &config,
                    FCT_initialization::InitState &external_state) {
  unsigned int ndx = external_state.ndx;
  std::vector<cl::sycl::cl_double> host_u_state(ndx + 4);
  std::vector<cl::sycl::cl_double> host_flux_low(ndx + 1);
  std::vector<cl::sycl::cl_double> host_flux_high(ndx + 1);
  std::vector<cl::sycl::cl_double> host_adiff_flux(ndx + 1);
  std::vector<cl::sycl::cl_double> host_flux_c(ndx + 1);

  cl::sycl::range<1> state_size{ndx + 4}, flux_size{ndx + 1},
      core_state_size{ndx};

  cl::sycl::default_selector device_selector;
  std::unique_ptr<cl::sycl::queue> device_queue;

  try {
    device_queue.reset(new cl::sycl::queue(device_selector));
  } catch (cl::sycl::exception const &e) {
    std::cout << "Caught a synchronous SYCL exception:" << std::endl
              << e.what() << std::endl;
    return;
  }

  std::cout
      << "Device: "
      << device_queue->get_device().get_info<cl::sycl::info::device::name>()
      << std::endl;

  // Copy in data
  for (int i = 0; i < ndx; ++i) {
    host_u_state[i + 2] = external_state.u[i];
  }

  {
    // Device buffers
    cl::sycl::buffer<cl::sycl::cl_double, 1> u_state(host_u_state.data(),
                                                     state_size);
    cl::sycl::buffer<cl::sycl::cl_double, 1> flux_low(host_flux_low.data(),
                                                      flux_size);
    cl::sycl::buffer<cl::sycl::cl_double, 1> flux_high(host_flux_high.data(),
                                                       flux_size);
    cl::sycl::buffer<cl::sycl::cl_double, 1> adiff_flux(host_adiff_flux.data(),
                                                        flux_size);
    cl::sycl::buffer<cl::sycl::cl_double, 1> flux_c(host_flux_c.data(),
                                                    flux_size);
    // Copy relevant configs out of structs
    const double a_vel = config.a, dt = config.dt, dx = external_state.dx;

    for (unsigned int timestep = 0; timestep < config.ndt; ++timestep) {
      // Set BCs
      device_queue->submit([&](cl::sycl::handler &cgh) {
        auto acc_u =
            u_state.get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.parallel_for<SetStateBC>(
            cl::sycl::range<1>{4}, [=](cl::sycl::id<1> index) {
              const unsigned int dst_index = index < 2 ? index : ndx + index;
              const unsigned int src_index = index < 2 ? ndx + index : index;
              acc_u[dst_index] = acc_u[src_index];
            });
      });
      // Compute fluxes
      device_queue->submit([&](cl::sycl::handler &cgh) {
        auto acc_u = u_state.get_access<cl::sycl::access::mode::read>(cgh);
        auto acc_flux_low =
            flux_low.get_access<cl::sycl::access::mode::write>(cgh);
        cgh.parallel_for<CalcLowFlux>(flux_size, [=](cl::sycl::id<1> index) {
          acc_flux_low[index] = a_vel * acc_u[index + 1];
        });
      });
      device_queue->submit([&](cl::sycl::handler &cgh) {
        auto acc_u = u_state.get_access<cl::sycl::access::mode::read>(cgh);
        auto acc_flux_high =
            flux_high.get_access<cl::sycl::access::mode::write>(cgh);
        cgh.parallel_for<CalcHighFlux>(flux_size, [=](cl::sycl::id<1> index) {
          const double sigma_i = (a_vel * dt / dx);
          acc_flux_high[index] = a_vel * 0.5 *
                                 ((1 + sigma_i) * acc_u[index + 1] +
                                  (1 - sigma_i) * acc_u[index + 2]);
        });
      });
      // Compute diff flux
      device_queue->submit([&](cl::sycl::handler &cgh) {
        auto acc_flux_low =
            flux_low.get_access<cl::sycl::access::mode::read>(cgh);
        auto acc_flux_high =
            flux_high.get_access<cl::sycl::access::mode::read>(cgh);
        auto acc_adiff_flux =
            adiff_flux.get_access<cl::sycl::access::mode::write>(cgh);
        cgh.parallel_for<CalcADiffFlux>(flux_size, [=](cl::sycl::id<1> index) {
          acc_adiff_flux[index] = acc_flux_high[index] - acc_flux_low[index];
        });
      });
      // Do low update
      device_queue->submit([&](cl::sycl::handler &cgh) {
        auto acc_flux_low =
            flux_low.get_access<cl::sycl::access::mode::read>(cgh);
        auto acc_u = u_state.get_access<cl::sycl::access::mode::write>(cgh);
        cgh.parallel_for<LowFluxUpdate>(
            core_state_size, [=](cl::sycl::id<1> index) {
              const double dtdx = (dt / dx);
              acc_u[index + 2] +=
                  dtdx * (acc_flux_low[index] - acc_flux_low[index]);
            });
      });
      // Set BC cells
      device_queue->submit([&](cl::sycl::handler &cgh) {
        auto acc_u =
            u_state.get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.parallel_for<SetStateBC2>(
            cl::sycl::range<1>{4}, [=](cl::sycl::id<1> index) {
              const unsigned int dst_index = index < 2 ? index : ndx + index;
              const unsigned int src_index = index < 2 ? ndx + index : index;
              acc_u[dst_index] = acc_u[src_index];
            });
      });
      // Calc FCT flux
      device_queue->submit([&](cl::sycl::handler &cgh) {
        auto acc_adiff_flux =
            adiff_flux.get_access<cl::sycl::access::mode::read>(cgh);
        auto acc_u = u_state.get_access<cl::sycl::access::mode::read>(cgh);
        auto acc_flux_c = flux_c.get_access<cl::sycl::access::mode::write>(cgh);
        cgh.parallel_for<CalcFCTFLux>(flux_size, [=](cl::sycl::id<1> index) {
          const double dxdt = (dx / dt);
          const double sign_a = cl::sycl::copysign(1.0, acc_adiff_flux[index]);
          const double left_d = acc_u[index + 1] - acc_u[index + 0];
          const double right_d = acc_u[index + 3] - acc_u[index + 2];
          acc_flux_c[index] =
              sign_a * std::max(0.0, std::min(std::min(sign_a * dxdt * left_d,
                                                       sign_a * dxdt * right_d),
                                              abs(acc_adiff_flux[index])));
        });
      });
      // Do full update
      device_queue->submit([&](cl::sycl::handler &cgh) {
        auto acc_flux_c = flux_c.get_access<cl::sycl::access::mode::read>(cgh);
        auto acc_u = u_state.get_access<cl::sycl::access::mode::write>(cgh);
        cgh.parallel_for<FullUpdate>(
            core_state_size, [=](cl::sycl::id<1> index) {
              const double dtdx = (dt / dx);
              acc_u[index + 2] +=
                  dtdx * (acc_flux_c[index] - acc_flux_c[index]);
            });
      });
      device_queue->wait_and_throw();
    }
  }

  external_state.time = config.end_time;

  // Copy out into u
  for (int i = 0; i < ndx; ++i) {
    external_state.u[i] = host_u_state[i + 2];
  }
}
