#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

#include <CL/sycl.hpp>
#include <vector>

#include "../utils/model_data.h"
#include "../utils/model_utils.h"

struct DeviceConfig {
  std::string device_name;
};

template <typename T>
void do_computation(const Model_Data::ProblemConfig &config,
                    const DeviceConfig &,
                    const Model_Data::ProblemState<T> &problem,
                    Model_Data::SolutionState<T> &solution);

int main(int argc, char *argv[]) {
  Model_Data::ProblemConfig config;
  DeviceConfig d_config;
  {
    const auto input_config = Model_Data::get_config_from_cli(argc, argv);

    config = Model_Data::init_from_toml(input_config);

    const auto dev_table = toml::find(input_config, "Device");
    d_config.device_name = toml::find_or<std::string>(dev_table, "Name", "");
  }

  Model_Data::ProblemState<double> problem_state;

  Model_IO::read_problem(std::string(config.hdf5_config_filename), problem_state);

  Model_Data::SolutionState<double> solution_state(
      problem_state.ndx, problem_state.ndy, problem_state.hx, problem_state.hy);

  // Do computation
  auto start = std::chrono::steady_clock::now();
  { do_computation(config, d_config, problem_state, solution_state); }
  auto end = std::chrono::steady_clock::now();
  auto time =
      (std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
           .count()) /
      1.0e3;

  // Print results (problem size, time and bandwidth in GB/s).
  std::cout << "TIme for computation " << time << " sec" << std::endl;

  Model_IO::write_vis_metadata(std::string("output.xdmf"), std::string(config.hdf5_output_filename),
                               solution_state);

  return 0;
}

template <typename T> struct MGLevel {
  MGLevel(const size_t ndx, const size_t ndy, const T dx, const T dy);

  // Host data
  const size_t ndx, ndy;
  const T dx, dy;

  // Buffers
  cl::sycl::buffer<T, 3> matrix;
  cl::sycl::buffer<T, 2> u;
  cl::sycl::buffer<T, 2> rhs;
};

// Useful functors
class GenAFromK {};
class RHSFromF {};
class CoarsenA {};
class AMult {};
class InterpolateU {};
class FullWeightU {};
class SmoothWJ {};
class DotProduct {};

template<typename T>
void do_computation(const Model_Data::ProblemConfig &config,
                    const DeviceConfig &d_config,
                    const Model_Data::ProblemState<T> &problem,
                    Model_Data::SolutionState<T> &solution) {
  cl::sycl::device device_selected;
  {
    unsigned int dev_sel = 0;
    std::vector<cl::sycl::platform> platforms;

    cl::sycl::default_selector backup_selector;
    cl::sycl::device default_device = backup_selector.select_device();
    const std::string default_device_name =
        default_device.get_info<cl::sycl::info::device::name>();

    unsigned int plat_i = 0;
    std::cout << "Device List:" << std::endl;
    for (const auto &plat : cl::sycl::platform::get_platforms()) {
      const std::string plat_name =
          plat.get_info<cl::sycl::info::platform::name>();
      std::cout << "\t[" << plat_i + 1 << "] " << plat_name << std::endl;
      std::vector<cl::sycl::device> devices = plat.get_devices();
      unsigned int dev_i = 0;
      for (const auto &dev : devices) {
        const std::string dev_name =
            dev.get_info<cl::sycl::info::device::name>();
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
        std::cout << "\t\t[" << dev_i + 1
                  << (dev_name == d_config.device_name ? "*" : " ")
                  << (dev_name == default_device_name ? "D" : " ") << "] ("
                  << dev_type << ") " << dev_name << std::endl;
        ++dev_i;
      }
      if (d_config.device_name != "" && dev_sel == 0) {
        std::cout << "Specified device " << d_config.device_name
                  << " not found." << std::endl;
        exit(-1);
      }
      device_selected = dev_sel > 0 ? devices[dev_sel - 1] : default_device;
      ++plat_i;
    }
  }

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

  {
    // Device buffers

    device_queue->wait_and_throw();
  }
}
