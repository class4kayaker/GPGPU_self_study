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
                    const Model_Data::ModelState<T> &problem,
                    Model_Data::ModelState<T> &solution);

int main(int argc, char *argv[]) {
  Model_Data::ProblemConfig config;
  DeviceConfig d_config;
  {
    const auto input_config = Model_Data::get_config_from_cli(argc, argv);

    config = Model_Data::init_from_toml(input_config);

    const auto dev_table = toml::find(input_config, "Device");
    d_config.device_name = toml::find_or<std::string>(dev_table, "Name", "");
  }

  Model_Data::ModelState<double> problem_state({});

  Model_IO::read_problem(config.hdf5_config_filename, problem_state);

  Model_Data::ModelState<double> solution_state(
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

  Model_IO::write_vis_metadata(std::string("output.xmf"),
                               std::string(config.hdf5_output_filename),
                               solution_state);

  return 0;
}

// offset functions
static inline cl::sycl::id<2> offset_idx(cl::sycl::id<2> i_idx, int ii,
                                         int jj) {
  return cl::sycl::id<2>(i_idx[0] + ii, i_idx[1] + jj);
}
static inline cl::sycl::id<3> offset_idx(cl::sycl::id<3> i_idx, int ii, int jj,
                                         int kk) {
  return cl::sycl::id<3>(i_idx[0] + ii, i_idx[1] + jj, i_idx[3] + kk);
}

// Multigrid grid class

template <typename T> struct MGLevel {
  MGLevel(const size_t ndx, const size_t ndy, const T dx, const T dy);

  MGLevel coarseLevel();

  void initAfromK(const std::unique_ptr<cl::sycl::queue> &queue,
                  cl::sycl::buffer<T, 2> &prob_k);

  // Host data
  const size_t ndx, ndy;
  const T dx, dy;

  // Buffers
  cl::sycl::buffer<T, 3> matrix;
  cl::sycl::buffer<T, 2> u;
  cl::sycl::buffer<T, 2> rhs;
};

template <typename T>
MGLevel<T>::MGLevel(const size_t a_ndx, const size_t a_ndy, const T a_dx,
                    const T a_dy)
    : ndx(a_ndx), ndy(a_ndy), dx(a_dx), dy(a_dx),
      matrix(cl::sycl::range<3>(5, ndx - 1, ndy - 1)),
      u(cl::sycl::range<2>(ndx - 1, ndy - 1)),
      rhs(cl::sycl::range<2>(ndx - 1, ndy - 1)) {}

template <typename T> MGLevel<T> MGLevel<T>::coarseLevel() {
  const size_t n_ndx = ndx / 2, n_ndy = ndy / 2;
  const T n_dx = 2.0 * dx, n_dy = 2.0 * dy;
  return MGLevel<T>(n_ndx, n_ndy, n_dx, n_dy);
}

template <typename T>
void MGLevel<T>::initAfromK(const std::unique_ptr<cl::sycl::queue> &device_queue,
                            cl::sycl::buffer<T, 2> &prob_k) {
  device_queue->submit([&](cl::sycl::handler &cgh) {
    size_t ndx = ndx;
    size_t ndy = ndy;
    T dx = dx;
    T dy = dy;
    auto ptr_k = prob_k.template get_access<cl::sycl::access::mode::read>(cgh);
    auto ptr_matrix =
        matrix.template get_access<cl::sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for(
        cl::sycl::range<2>(ndy - 1, ndx - 1), [=](cl::sycl::id<2> idx) {
          const T denom_x = 1.0 / (2.0 * dx * dx);
          const T denom_y = 1.0 / (2.0 * dy * dy);

          const T k_c = ptr_k[offset_idx(idx, 1, 1)];
          const T k_xm = ptr_k[offset_idx(idx, 1, 0)];
          const T k_xp = ptr_k[offset_idx(idx, 1, 2)];
          const T k_ym = ptr_k[offset_idx(idx, 0, 1)];
          const T k_yp = ptr_k[offset_idx(idx, 2, 1)];

          cl::sycl::id<3> m_cen_idx(0, idx[1], idx[0]);
          cl::sycl::id<3> m_x_idx(1, idx[1], idx[0]);
          cl::sycl::id<3> m_y_idx(2, idx[1], idx[0]);
          cl::sycl::id<3> m_xpy_idx(3, idx[1], idx[0]);
          cl::sycl::id<3> m_xmy_idx(4, idx[1], idx[0]);

          ptr_matrix[m_cen_idx] = (2.0 * k_c + k_xm + k_xp) / denom_x +
                                  (2.0 * k_c + k_ym + k_yp) / denom_y;

          if (idx[1] < ndx - 2)
            ptr_matrix[m_x_idx] = -1.0 * (k_c + k_xm) / denom_x;

          if (idx[0] < ndy - 2)
            ptr_matrix[m_y_idx] = -1.0 * (k_c + k_ym) / denom_y;

          if (idx[1] < ndx - 2 && idx[0] < ndy - 2) {
            ptr_matrix[m_xpy_idx] = 0.0;
            ptr_matrix[m_xmy_idx] = 0.0;
          }
        });
  });
}

// Useful functors
template <typename T> class GenAFromK {};
class RHSFromF {};
class CoarsenA {};
class AMult {};
class InterpolateU {};
class FullWeightU {};
class SmoothWJ {};
class DotProduct {};

template <typename T>
void do_computation(const Model_Data::ProblemConfig &config,
                    const DeviceConfig &d_config,
                    const Model_Data::ModelState<T> &problem,
                    Model_Data::ModelState<T> &solution) {
  cl::sycl::device device_selected;
  {
    bool dev_found = false;
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
        const bool selected_dev = (d_config.device_name == dev_name);
        if (selected_dev) {
          device_selected = dev;
          dev_found = true;
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
        std::cout << "\t\t[" << dev_i + 1 << (selected_dev ? "*" : " ")
                  << (dev_name == default_device_name ? "D" : " ") << "] ("
                  << dev_type << ") \"" << dev_name << "\"" << std::endl;
        ++dev_i;
      }
      ++plat_i;
    }
    if (d_config.device_name != "" && (!dev_found)) {
      std::cout << "Specified device " << d_config.device_name << " not found."
                << std::endl;
      exit(-1);
    }
    if (!dev_found) {
      device_selected = default_device;
    }
  }

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

  // Compute data storage locations

  // Copy data over
  for (size_t j = 0; j < problem.ndx + 1; ++j) {
    for (size_t i = 0; i < problem.ndy + 1; ++i) {
      const size_t arr_ind = j * (problem.ndy + 1) + i;
      solution.k[arr_ind] = problem.k[arr_ind];
      solution.heat_source[arr_ind] = problem.heat_source[arr_ind];
      if (i == 0 || i == problem.ndy || j == 0 || j == problem.ndx) {
        solution.temperature[arr_ind] = problem.temperature[arr_ind];
      } else {
        solution.temperature[arr_ind] = 0.0;
      }
    }
  }

  {
    // Base data buffers
    cl::sycl::buffer<T, 2> prob_temp(
        solution.temperature.data(),
        cl::sycl::range<2>(problem.ndy + 1, problem.ndx + 1));
    cl::sycl::buffer<T, 2> prob_k(
        solution.k.data(),
        cl::sycl::range<2>(problem.ndy + 1, problem.ndx + 1));
    cl::sycl::buffer<T, 2> prob_heat(
        solution.heat_source.data(),
        cl::sycl::range<2>(problem.ndy + 1, problem.ndx + 1));

    // Multigrid setup
    std::vector<MGLevel<T>> grids;
    grids.push_back(
        MGLevel<T>(problem.ndx, problem.ndy, problem.hx, problem.hy));
    while (grids.back().ndx > 1 && grids.back().ndy > 1) {
      grids.push_back(grids.back().coarseLevel());
    }

    // Do necessary init for MG
    {
      auto front_grid = grids.front();
      front_grid.initAfromK(device_queue, prob_k);
    }

    // Solution computation

    device_queue->wait_and_throw();
  }
}
