#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

#include <CL/sycl.hpp>
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

    const auto dev_table = toml::find(input_config, "Device");
    d_config.device_name = toml::find_or<std::string>(dev_table, "Name", "");
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

class SetStateBC;
class SetStateBC2;
class CalcLowFlux;
class CalcHighFlux;
class CalcADiffFlux;
class LowFluxUpdate;
class CalcFCTFLux;
class FullUpdate;

void do_computation(const FCT_initialization::ProblemConfig &config,
                    const DeviceConfig &d_config,
                    FCT_initialization::InitState &external_state) {
  cl::sycl::device device_selected;
  {
    unsigned int dev_sel = 0;
    std::vector<cl::sycl::device> devices = cl::sycl::device::get_devices();

    cl::sycl::default_selector backup_selector;
    cl::sycl::device default_device = backup_selector.select_device();
    const std::string default_device_name =
        default_device.get_info<cl::sycl::info::device::name>();

    std::cout << "Device List:" << std::endl;
    unsigned int dev_i = 0;
    for (const auto &dev : devices) {
      const std::string dev_name = dev.get_info<cl::sycl::info::device::name>();
      if (d_config.device_name == dev_name) {
        dev_sel = dev_i + 1;
      }
      std::string dev_type;
      auto d_type_var = dev.get_info<cl::sycl::info::device::device_type>();
      if (d_type_var == cl::sycl::info::device_type::cpu) {
        dev_type = "cpu";
      } else if (d_type_var == cl::sycl::info::device_type::gpu) {
        dev_type = "gpu";
      } else if (d_type_var == cl::sycl::info::device_type::host) {
        dev_type = "hst";
      } else {
        dev_type = "unk";
      }
      std::cout << "\t[" << dev_i + 1
                << (dev_name == d_config.device_name ? "*" : " ")
                << (dev_name == default_device_name ? "D" : " ") << "] ("
                << dev_type << ") " << dev_name << std::endl;
      ++dev_i;
    }
    if (d_config.device_name != "" && dev_sel == 0) {
      std::cout << "Specified device " << d_config.device_name << " not found."
                << std::endl;
      exit(-1);
    }
    device_selected = dev_sel > 0 ? devices[dev_sel - 1] : default_device;
  }
  unsigned int ndx = external_state.ndx;
  std::vector<cl::sycl::cl_double> host_u_state(ndx + 4);
  std::vector<cl::sycl::cl_double> host_flux_low(ndx + 1);
  std::vector<cl::sycl::cl_double> host_flux_high(ndx + 1);
  std::vector<cl::sycl::cl_double> host_adiff_flux(ndx + 1);
  std::vector<cl::sycl::cl_double> host_flux_c(ndx + 1);

  cl::sycl::range<1> state_size{ndx + 4}, flux_size{ndx + 1},
      core_state_size{ndx};

  // cl::sycl::default_selector device_selector;
  std::unique_ptr<cl::sycl::queue> device_queue;

  auto exception_handler = [](cl::sycl::exception_list exceptions) {
    for (std::exception_ptr const &e : exceptions) {
      try {
        std::rethrow_exception(e);
      } catch (sycl::exception const &e) {
        std::cout << "Caught asynchronous SYCL exception:\n"
                  << e.what() << std::endl;
      }
    }
  };

  try {
    device_queue.reset(new cl::sycl::queue(device_selected, exception_handler));
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
  for (size_t i = 0; i < ndx; ++i) {
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
                  dtdx * (acc_flux_low[index] - acc_flux_low[index + 1]);
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
                  dtdx * (acc_flux_c[index] - acc_flux_c[index + 1]);
            });
      });
      device_queue->wait_and_throw();
    }
  }

  external_state.time = config.end_time;

  // Copy out into u
  for (size_t i = 0; i < ndx; ++i) {
    external_state.u[i] = host_u_state[i + 2];
  }
}
