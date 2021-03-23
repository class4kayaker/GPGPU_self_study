#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

#include <math.h>

#include <CL/sycl.hpp>
#include <vector>

#include "../utils/model_data.h"
#include "../utils/model_utils.h"

struct DeviceConfig {
  std::string device_name;
  size_t wg_size;
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
    d_config.wg_size = toml::find_or<int>(dev_table, "WG size", 64);
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
  std::cout << "Time for computation " << time << " sec" << std::endl;

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
  return cl::sycl::id<3>(i_idx[0] + ii, i_idx[1] + jj, i_idx[2] + kk);
}

// convenience write fns
template <typename T>
void write_buffer(std::string name, cl::sycl::buffer<T, 1> b) {
  cl::sycl::range<1> r = b.get_range();
  auto acc_b = b.template get_access<cl::sycl::access::mode::read>();
  std::cout << name << ":" << std::endl;
  for (size_t i = 0; i < r[0]; ++i) {
    cl::sycl::id<1> idx(i);
    std::cout << acc_b[idx] << " ";
  }
  std::cout << std::endl;
}
template <typename T>
void write_buffer(std::string name, cl::sycl::buffer<T, 2> b) {
  cl::sycl::range<2> r = b.get_range();
  auto acc_b = b.template get_access<cl::sycl::access::mode::read>();
  std::cout << name << ":" << std::endl;
  for (size_t j = 0; j < r[0]; ++j) {
    for (size_t i = 0; i < r[1]; ++i) {
      cl::sycl::id<2> idx(j, i);
      std::cout << acc_b[idx] << " ";
    }
    std::cout << std::endl;
  }
}
template <typename T>
void write_buffer(std::string name, cl::sycl::buffer<T, 3> b) {
  cl::sycl::range<3> r = b.get_range();
  auto acc_b = b.template get_access<cl::sycl::access::mode::read>();
  std::cout << name << ":" << std::endl;
  for (size_t k = 0; k < r[0]; ++k) {
    for (size_t j = 0; j < r[1]; ++j) {
      for (size_t i = 0; i < r[2]; ++i) {
        cl::sycl::id<3> idx(k, j, i);
        std::cout << acc_b[idx] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << std::endl;
  }
}

// Problem data class w/ sycl buffers

template <typename T> struct SYCL_ModelData {
  SYCL_ModelData<T>(Model_Data::ModelState<T> &state);
  size_t ndx, ndy;
  T hx, hy;

  cl::sycl::buffer<T, 2> k;
  cl::sycl::buffer<T, 2> heat_source;
  cl::sycl::buffer<T, 2> temperature;
};

template <typename T>
SYCL_ModelData<T>::SYCL_ModelData(Model_Data::ModelState<T> &state)
    : ndx(state.ndx), ndy(state.ndy), hx(state.hx), hy(state.hy),
      k(state.k.data(), cl::sycl::range<2>(ndx + 1, ndy + 1)),
      heat_source(state.heat_source.data(),
                  cl::sycl::range<2>(ndx + 1, ndy + 1)),
      temperature(state.temperature.data(),
                  cl::sycl::range<2>(ndx + 1, ndy + 1)) {}

// Multigrid grid class

template <typename T> struct MGLevel {
  MGLevel(const size_t ndx, const size_t ndy, const T dx, const T dy);

  MGLevel coarseLevel();

  void initAfromK(const std::unique_ptr<cl::sycl::queue> &queue,
                  cl::sycl::buffer<T, 2> &prob_k);

  void initAfromFineA(const std::unique_ptr<cl::sycl::queue> &queue,
                      MGLevel<T> &fineLevel);

  void zeroU(const std::unique_ptr<cl::sycl::queue> &queue);

  void mult(const std::unique_ptr<cl::sycl::queue> &queue,
            cl::sycl::buffer<T, 2> &in, cl::sycl::buffer<T, 2> &out);

  void resid(const std::unique_ptr<cl::sycl::queue> &queue,
             cl::sycl::buffer<T, 2> &u_in, sycl::buffer<T, 2> &f_in,
             cl::sycl::buffer<T, 2> &out);

  void wjacobi(const std::unique_ptr<cl::sycl::queue> &queue, T weight);

  // Host data
  const size_t m_ndx, m_ndy;
  const T m_dx, m_dy;

  // Buffers
  cl::sycl::buffer<T, 3> matrix;
  cl::sycl::buffer<T, 2> u;
  cl::sycl::buffer<T, 2> rhs;
};

template <typename T>
MGLevel<T>::MGLevel(const size_t a_ndx, const size_t a_ndy, const T a_dx,
                    const T a_dy)
    : m_ndx(a_ndx), m_ndy(a_ndy), m_dx(a_dx), m_dy(a_dx),
      matrix(cl::sycl::range<3>(5, m_ndy - 1, m_ndx - 1)),
      u(cl::sycl::range<2>(m_ndy - 1, m_ndx - 1)),
      rhs(cl::sycl::range<2>(m_ndy - 1, m_ndx - 1)) {}

template <typename T> MGLevel<T> MGLevel<T>::coarseLevel() {
  const size_t n_ndx = m_ndx / 2, n_ndy = m_ndy / 2;
  const T n_dx = 2.0 * m_dx, n_dy = 2.0 * m_dy;
  return MGLevel<T>(n_ndx, n_ndy, n_dx, n_dy);
}

template <typename T>
void MGLevel<T>::initAfromK(
    const std::unique_ptr<cl::sycl::queue> &device_queue,
    cl::sycl::buffer<T, 2> &prob_k) {
  device_queue->submit([&](cl::sycl::handler &cgh) {
    size_t ndx = m_ndx;
    size_t ndy = m_ndy;
    T dx = m_dx;
    T dy = m_dy;
    auto ptr_k = prob_k.template get_access<cl::sycl::access::mode::read>(cgh);
    auto ptr_matrix =
        matrix.template get_access<cl::sycl::access::mode::discard_write>(cgh);

    cl::sycl::range<2> rng(ndy - 1, ndx - 1);
    cgh.parallel_for(rng, [=](cl::sycl::id<2> idx) {
      const T denom_x = 1.0 / (2.0 * dx * dx);
      const T denom_y = 1.0 / (2.0 * dy * dy);

      const T k_c = ptr_k[offset_idx(idx, 1, 1)];
      const T k_xm = ptr_k[offset_idx(idx, 0, 1)];
      const T k_xp = ptr_k[offset_idx(idx, 2, 1)];
      const T k_ym = ptr_k[offset_idx(idx, 1, 0)];
      const T k_yp = ptr_k[offset_idx(idx, 1, 2)];

      cl::sycl::id<3> m_cen_idx(0, idx[1], idx[0]);
      cl::sycl::id<3> m_x_idx(1, idx[1], idx[0]);
      cl::sycl::id<3> m_y_idx(2, idx[1], idx[0]);
      cl::sycl::id<3> m_xpy_idx(3, idx[1], idx[0]);
      cl::sycl::id<3> m_xmy_idx(4, idx[1], idx[0]);

      ptr_matrix[m_cen_idx] = (2.0 * k_c + k_xm + k_xp) * denom_x +
                              (2.0 * k_c + k_ym + k_yp) * denom_y;

      ptr_matrix[m_x_idx] = (idx[0] > 0) ? -1.0 * (k_c + k_xm) * denom_x : 0.0;

      ptr_matrix[m_y_idx] = (idx[1] > 0) ? -1.0 * (k_c + k_ym) * denom_y : 0.0;
      ptr_matrix[m_xpy_idx] = 0.0;
      ptr_matrix[m_xmy_idx] = 0.0;
    });
  });
}

template <typename T>
void MGLevel<T>::initAfromFineA(
    const std::unique_ptr<cl::sycl::queue> &device_queue,
    MGLevel<T> &fineLevel) {
  cl::sycl::buffer<2> tmp_input(fineLevel.u.get_range());
  cl::sycl::buffer<2> tmp_mult(fineLevel.u.get_range());
    for (int ioff=0; ioff<3; ++ioff){
        for(int joff=0; joff<3; ++joff){
            // Initialize to interpolated values on appropriate points
            // Multi
            fineLevel.mult(tmp_input, tmp_mult);
            // Set matrix values based on full weighting
        }
    }
}

template <typename T>
void MGLevel<T>::mult(const std::unique_ptr<cl::sycl::queue> &queue,
                      cl::sycl::buffer<T, 2> &in, cl::sycl::buffer<T, 2> &out) {
  queue->submit([&](cl::sycl::handler &cgh) {
    size_t ndx = m_ndx;
    size_t ndy = m_ndy;
    auto ptr_in = in.template get_access<cl::sycl::access::mode::read>(cgh);
    auto ptr_matrix =
        matrix.template get_access<cl::sycl::access::mode::read>(cgh);
    auto ptr_out =
        out.template get_access<cl::sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for(cl::sycl::range<2>(ndy - 1, ndx - 1),
                     [=](cl::sycl::id<2> idx) {
                       cl::sycl::id<3> mat_idx(0, idx[0], idx[1]);

                       const bool x_m = (idx[1] > 0), x_p = (idx[1] < ndx - 2),
                                  y_m = (idx[0] > 0), y_p = (idx[0] < ndy - 2);

                       T out = ptr_matrix[mat_idx] * ptr_in[idx];
                       if (x_m)
                         out += ptr_matrix[offset_idx(mat_idx, 1, 0, 0)] *
                                ptr_in[offset_idx(idx, 0, -1)];
                       if (x_p)
                         out += ptr_matrix[offset_idx(mat_idx, 1, 0, 1)] *
                                ptr_in[offset_idx(idx, 0, 1)];
                       if (y_m)
                         out += ptr_matrix[offset_idx(mat_idx, 2, 0, 0)] *
                                ptr_in[offset_idx(idx, -1, 0)];
                       if (y_p)
                         out += ptr_matrix[offset_idx(mat_idx, 2, 1, 0)] *
                                ptr_in[offset_idx(idx, 1, 0)];

                       if (x_m && y_m)
                         out += ptr_matrix[offset_idx(mat_idx, 3, 0, 0)] *
                                ptr_in[offset_idx(idx, -1, -1)];
                       if (x_p && y_p)
                         out += ptr_matrix[offset_idx(mat_idx, 3, 1, 1)] *
                                ptr_in[offset_idx(idx, 1, 1)];

                       if (x_m && y_p)
                         out += ptr_matrix[offset_idx(mat_idx, 4, 0, 0)] *
                                ptr_in[offset_idx(idx, -1, 1)];
                       if (x_p && y_m)
                         out += ptr_matrix[offset_idx(mat_idx, 4, 1, -1)] *
                                ptr_in[offset_idx(idx, 1, -1)];

                       ptr_out[idx] = out;
                     });
  });
}

template <typename T>
void MGLevel<T>::resid(const std::unique_ptr<cl::sycl::queue> &queue,
                       cl::sycl::buffer<T, 2> &u_in, sycl::buffer<T, 2> &f_in,
                       cl::sycl::buffer<T, 2> &out) {
  queue->submit([&](cl::sycl::handler &cgh) {
    size_t ndx = m_ndx;
    size_t ndy = m_ndy;
    auto ptr_in = u_in.template get_access<cl::sycl::access::mode::read>(cgh);
    auto ptr_f_in = f_in.template get_access<cl::sycl::access::mode::read>(cgh);
    auto ptr_matrix =
        matrix.template get_access<cl::sycl::access::mode::read>(cgh);
    auto ptr_out =
        out.template get_access<cl::sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for(cl::sycl::range<2>(ndy - 1, ndx - 1),
                     [=](cl::sycl::id<2> idx) {
                       cl::sycl::id<3> mat_idx(0, idx[0], idx[1]);

                       const bool x_m = (idx[1] > 0), x_p = (idx[1] < ndx - 2),
                                  y_m = (idx[0] > 0), y_p = (idx[0] < ndy - 2);

                       T out = ptr_matrix[mat_idx] * ptr_in[idx];
                       if (x_m)
                         out += ptr_matrix[offset_idx(mat_idx, 1, 0, 0)] *
                                ptr_in[offset_idx(idx, 0, -1)];
                       if (x_p)
                         out += ptr_matrix[offset_idx(mat_idx, 1, 0, 1)] *
                                ptr_in[offset_idx(idx, 0, 1)];
                       if (y_m)
                         out += ptr_matrix[offset_idx(mat_idx, 2, 0, 0)] *
                                ptr_in[offset_idx(idx, -1, 0)];
                       if (y_p)
                         out += ptr_matrix[offset_idx(mat_idx, 2, 1, 0)] *
                                ptr_in[offset_idx(idx, 1, 0)];

                       if (x_m && y_m)
                         out += ptr_matrix[offset_idx(mat_idx, 3, 0, 0)] *
                                ptr_in[offset_idx(idx, -1, -1)];
                       if (x_p && y_p)
                         out += ptr_matrix[offset_idx(mat_idx, 3, 1, 1)] *
                                ptr_in[offset_idx(idx, 1, 1)];

                       if (x_m && y_p)
                         out += ptr_matrix[offset_idx(mat_idx, 4, 0, 0)] *
                                ptr_in[offset_idx(idx, -1, 1)];
                       if (x_p && y_m)
                         out += ptr_matrix[offset_idx(mat_idx, 4, 1, -1)] *
                                ptr_in[offset_idx(idx, 1, -1)];

                       ptr_out[idx] = ptr_f_in[idx] - out;
                     });
  });
}

template <typename T>
void MGLevel<T>::zeroU(const std::unique_ptr<cl::sycl::queue> &queue) {
  queue->submit([&](cl::sycl::handler &cgh) {
    size_t ndx = m_ndx;
    size_t ndy = m_ndy;
    auto ptr_u =
        u.template get_access<cl::sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for(cl::sycl::range<2>(ndy - 1, ndx - 1),
                     [=](cl::sycl::id<2> idx) {
                       cl::sycl::id<2> h5idx(idx[1], idx[0]);

                       ptr_u[idx] = 0.0;
                     });
  });
}

template <typename T>
void MGLevel<T>::wjacobi(const std::unique_ptr<cl::sycl::queue> &queue,
                         T weight) {
  cl::sycl::buffer<2> tmp_mult(u.get_range());
  mult(queue, u, tmp_mult);
  queue->submit([&](cl::sycl::handler &cgh) {
    T wj_weight;
    size_t ndx = m_ndx;
    size_t ndy = m_ndy;
    auto ptr_matrix =
        matrix.template get_access<cl::sycl::access::mode::read>(cgh);
    auto ptr_rhs = rhs.template get_access<cl::sycl::access::mode::read>(cgh);
    auto ptr_Au =
        tmp_mult.template get_access<cl::sycl::access::mode::read>(cgh);
    auto ptr_u = u.template get_access<cl::sycl::access::mode::read_write>(cgh);
    cgh.parallel_for(cl::sycl::range<2>(ndy - 1, ndx - 1),
                     [=](cl::sycl::id<2> idx) {
                       cl::sycl::id<2> mat_diag_idx(0, idx[0], idx[1]);
                       ptr_u[idx] = (1.0 - wj_weight) * ptr_u[idx] +
                                    wj_weight * (ptr_rhs[idx] - ptr_Au[idx]) /
                                        matrix[mat_diag_idx];
                     });
  });
}

// Interpolate and weight

template <typename T>
void interpolateU(const std::unique_ptr<cl::sycl::queue> &queue,
                  size_t coarse_ndx, size_t coarse_ndy,
                  cl::sycl::buffer<T, 2> &coarse_u,
                  cl::sycl::buffer<T, 2> &fine_u) {
  queue->submit([&](cl::sycl::handler &cgh) {
    size_t ndx = 2 * coarse_ndx;
    size_t ndy = 2 * coarse_ndy;
    auto ptr_coarse_u =
        coarse_u.template get_access<cl::sycl::access::mode::read>(cgh);
    auto ptr_fine_u =
        fine_u.template get_access<cl::sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for(cl::sycl::range<2>(ndy - 1, ndx - 1),
                     [=](cl::sycl::id<2> idx) {
                       size_t i = idx[1], j = idx[0];

                       const bool x_m = (idx[1] > 0), x_p = (idx[1] < ndx - 2),
                                  y_m = (idx[0] > 0), y_p = (idx[0] < ndy - 2);

                       ptr_fine_u[idx] = 0.0;

                       if (x_m && y_m) {
                         cl::sycl::id<2> idx_coarse(j / 2, i / 2);

                         ptr_fine_u[idx] += 0.25 * ptr_coarse_u[idx_coarse];
                       }
                       if (x_p && y_m) {
                         cl::sycl::id<2> idx_coarse(j / 2, (i + 1) / 2);

                         ptr_fine_u[idx] += 0.25 * ptr_coarse_u[idx_coarse];
                       }
                       if (x_m && y_p) {
                         cl::sycl::id<2> idx_coarse((j + 1) / 2, i / 2);

                         ptr_fine_u[idx] += 0.25 * ptr_coarse_u[idx_coarse];
                       }
                       if (x_p && y_p) {
                         cl::sycl::id<2> idx_coarse((j + 1) / 2, (i + 1) / 2);

                         ptr_fine_u[idx] += 0.25 * ptr_coarse_u[idx_coarse];
                       }
                     });
  });
}

template <typename T>
void fullWeightU(const std::unique_ptr<cl::sycl::queue> &queue,
                 size_t coarse_ndx, size_t coarse_ndy,
                 cl::sycl::buffer<T, 2> &fine_u,
                 cl::sycl::buffer<T, 2> &coarse_u) {
  queue->submit([&](cl::sycl::handler &cgh) {
    size_t ndx = coarse_ndx;
    size_t ndy = coarse_ndy;
    auto ptr_fine_u =
        fine_u.template get_access<cl::sycl::access::mode::read>(cgh);
    auto ptr_coarse_u =
        coarse_u.template get_access<cl::sycl::access::mode::discard_write>(
            cgh);
    cgh.parallel_for(
        cl::sycl::range<2>(ndy - 1, ndx - 1), [=](cl::sycl::id<2> idx) {
          size_t i = idx[1], j = idx[0];

          const T fWeightStencil[9] = {0.25 / 4.0, 0.5 / 4.0, 0.25 / 4.0,
                                       0.5 / 4.0,  1.0 / 4.0, 0.5 / 4.0,
                                       0.25 / 4.0, 0.5 / 4.0, 0.25 / 4.0};

          cl::sycl::id<2> idx_fine(j * 2, i * 2);

          ptr_coarse_u[idx] = 0.0;
          for (size_t jj = 0; jj < 3; ++jj) {
            for (size_t ii = 0; ii < 3; ++ii) {
              ptr_coarse_u[idx] += fWeightStencil[jj * 3 + ii] *
                                   ptr_fine_u[offset_idx(idx_fine, jj, ii)];
            }
          }
        });
  });
}

// Copy data into local format

template <typename T>
void initU(const std::unique_ptr<cl::sycl::queue> &queue,
           SYCL_ModelData<T> &problem, cl::sycl::buffer<T, 2> &u) {
  queue->submit([&](cl::sycl::handler &cgh) {
    size_t ndx = problem.ndx;
    size_t ndy = problem.ndy;
    auto ptr_prob_temp =
        problem.temperature.template get_access<cl::sycl::access::mode::read>(
            cgh);
    auto ptr_u =
        u.template get_access<cl::sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for(cl::sycl::range<2>(ndy - 1, ndx - 1),
                     [=](cl::sycl::id<2> idx) {
                       cl::sycl::id<2> h5idx(idx[1], idx[0]);

                       ptr_u[idx] = ptr_prob_temp[offset_idx(h5idx, 1, 1)];
                     });
  });
}

template <typename T>
void initRHS(const std::unique_ptr<cl::sycl::queue> &queue,
             SYCL_ModelData<T> &problem, cl::sycl::buffer<T, 2> &rhs) {
  queue->submit([&](cl::sycl::handler &cgh) {
    size_t ndx = problem.ndx;
    size_t ndy = problem.ndy;
    T dx = problem.hx;
    T dy = problem.hy;
    auto ptr_k =
        problem.k.template get_access<cl::sycl::access::mode::read>(cgh);
    auto ptr_heat_source =
        problem.heat_source.template get_access<cl::sycl::access::mode::read>(
            cgh);
    auto ptr_temperature =
        problem.temperature.template get_access<cl::sycl::access::mode::read>(
            cgh);
    auto ptr_rhs =
        rhs.template get_access<cl::sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for(
        cl::sycl::range<2>(ndy - 1, ndx - 1), [=](cl::sycl::id<2> idx) {
          const T denom_x = 1.0 / (2.0 * dx * dx);
          const T denom_y = 1.0 / (2.0 * dy * dy);

          const bool x_m = (idx[1] > 0), x_p = (idx[1] < ndx - 2),
                     y_m = (idx[0] > 0), y_p = (idx[0] < ndy - 2);

          cl::sycl::id<2> h5idx(idx[1] + 1, idx[0] + 1);

          ptr_rhs[idx] = ptr_heat_source[offset_idx(h5idx, 0, 0)];

          if (!x_m)
            ptr_rhs[idx] += (ptr_k[offset_idx(h5idx, -1, 0)] +
                             ptr_k[offset_idx(h5idx, 0, 0)]) *
                            denom_x * ptr_temperature[offset_idx(h5idx, -1, 0)];
          if (!x_p)
            ptr_rhs[idx] += (ptr_k[offset_idx(h5idx, 0, 0)] +
                             ptr_k[offset_idx(h5idx, 1, 0)]) *
                            denom_x * ptr_temperature[offset_idx(h5idx, 1, 0)];
          if (!y_m)
            ptr_rhs[idx] += (ptr_k[offset_idx(h5idx, 0, -1)] +
                             ptr_k[offset_idx(h5idx, 0, 0)]) *
                            denom_y * ptr_temperature[offset_idx(h5idx, 0, -1)];
          if (!y_p)
            ptr_rhs[idx] += (ptr_k[offset_idx(h5idx, 0, 0)] +
                             ptr_k[offset_idx(h5idx, 0, 1)]) *
                            denom_y * ptr_temperature[offset_idx(h5idx, 0, 1)];
        });
  });
}

template <typename T>
void writeTemperature(const std::unique_ptr<cl::sycl::queue> &queue,
                      SYCL_ModelData<T> &problem, cl::sycl::buffer<T, 2> &u) {
  queue->submit([&](cl::sycl::handler &cgh) {
    size_t ndx = problem.ndx;
    size_t ndy = problem.ndy;
    auto ptr_prob_temp =
        problem.temperature.template get_access<cl::sycl::access::mode::write>(
            cgh);
    auto ptr_u = u.template get_access<cl::sycl::access::mode::read>(cgh);
    cgh.parallel_for(cl::sycl::range<2>(ndy - 1, ndx - 1),
                     [=](cl::sycl::id<2> idx) {
                       cl::sycl::id<2> h5idx(idx[1], idx[0]);

                       ptr_prob_temp[offset_idx(h5idx, 1, 1)] = ptr_u[idx];
                     });
  });
}

// Dot product

// Complexity of implementing this without ONEAPI extensions included into the
// SYCL 2020 provisional spec suggests waiting until they are more widespread
// for other implementations
//
// Note this also may be the location of a release compile seg-fault which may
// be in the reduction extension
template <typename T>
T dot_product(const std::unique_ptr<cl::sycl::queue> &queue,
              const DeviceConfig &d_config, cl::sycl::buffer<T, 2> &u1,
              cl::sycl::buffer<T, 2> &u2) {
  const size_t wg_size = d_config.wg_size;
  auto u_range = u1.get_range();
  cl::sycl::buffer<T, 1> sum_buff(cl::sycl::range<1>(1));
  // Begin computation
  queue->submit([&](cl::sycl::handler &cgh) {
    size_t ndx = u_range[1];
    size_t ndy = u_range[0];
    cl::sycl::range<2> l_range(1, std::min(wg_size, ndx));
    size_t l_ndy = l_range[0];
    size_t l_ndx = l_range[1];
    size_t ngy = (ndy % l_ndy == 0) ? ndy / l_ndy : ndy / l_ndy + 1;
    size_t ngx = (ndx % l_ndx == 0) ? ndx / l_ndx : ndx / l_ndx + 1;
    cl::sycl::range<2> g_range(ngy * l_ndy, ngx * l_ndx);
    cl::sycl::nd_range<2> kernel_range(g_range, l_range);
    auto ptr_u1 = u1.template get_access<cl::sycl::access::mode::read>(cgh);
    auto ptr_u2 = u2.template get_access<cl::sycl::access::mode::read>(cgh);
    auto sum = cl::sycl::accessor<T, 0, cl::sycl::access::mode::write,
                                  cl::sycl::access::target::global_buffer>(
        sum_buff, cgh);
    cgh.parallel_for(
        kernel_range,
        cl::sycl::ONEAPI::reduction(sum, 0.0, cl::sycl::ONEAPI::plus<T>()),
        [=](cl::sycl::nd_item<2> item, auto &sum) {
          auto idx = item.get_global_id();
          if (idx[0] < ndy && idx[1] < ndx)
            sum += ptr_u1[idx] * ptr_u2[idx];
        });
  });
  auto acc_sum = sum_buff.template get_access<cl::sycl::access::mode::read>();
  return acc_sum[0];
}

// In place adjust vector by scalar multiple of other vector

template <typename T>
void inc_ay(const std::unique_ptr<cl::sycl::queue> &queue,
            cl::sycl::buffer<T, 2> xv, T scale, cl::sycl::buffer<T, 2> yv) {
  queue->submit([&](cl::sycl::handler &cgh) {
    cl::sycl::range<2> xv_range = xv.get_range();
    T a_mult = scale;
    auto ptr_xv =
        xv.template get_access<cl::sycl::access::mode::read_write>(cgh);
    auto ptr_yv = yv.template get_access<cl::sycl::access::mode::read>(cgh);
    cgh.parallel_for(xv_range, [=](cl::sycl::id<2> idx) {
      ptr_xv[idx] += a_mult * ptr_yv[idx];
    });
  });
}

template <typename T>
void desc_update(const std::unique_ptr<cl::sycl::queue> &queue,
                 cl::sycl::buffer<T, 2> desc, T scale,
                 cl::sycl::buffer<T, 2> resid) {
  queue->submit([&](cl::sycl::handler &cgh) {
    cl::sycl::range<2> desc_range = desc.get_range();
    T a_mult = scale;
    auto ptr_desc =
        desc.template get_access<cl::sycl::access::mode::read_write>(cgh);
    auto ptr_resid =
        resid.template get_access<cl::sycl::access::mode::read>(cgh);
    cgh.parallel_for(desc_range, [=](cl::sycl::id<2> idx) {
      ptr_desc[idx] = ptr_resid[idx] + a_mult * ptr_desc[idx];
    });
  });
}

template <typename T>
void copy_vec(const std::unique_ptr<cl::sycl::queue> &queue,
              cl::sycl::buffer<T, 2> v_in, cl::sycl::buffer<T, 2> v_out) {
  queue->submit([&](cl::sycl::handler &cgh) {
    cl::sycl::range<2> v_range = v_in.get_range();
    auto ptr_v_in = v_in.template get_access<cl::sycl::access::mode::read>(cgh);
    auto ptr_v_out =
        v_out.template get_access<cl::sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for(
        v_range, [=](cl::sycl::id<2> idx) { ptr_v_out[idx] = ptr_v_in[idx]; });
  });
}

// Algorithms
template <typename T>
void basic_CG(const std::unique_ptr<cl::sycl::queue> &queue,
              const Model_Data::ProblemConfig &config,
              const DeviceConfig &d_config, MGLevel<T> &data) {
  cl::sycl::buffer<T, 2> resid(data.u.get_range());
  cl::sycl::buffer<T, 2> descent(data.u.get_range());
  cl::sycl::buffer<T, 2> Adescent(data.u.get_range());

  data.resid(queue, data.u, data.rhs, resid);
  T norm_resid = dot_product(queue, d_config, resid, resid),
    old_norm_resid = norm_resid;

  T normhR = sqrt(norm_resid * data.m_dx * data.m_dy), origNormhR = normhR;

  copy_vec(queue, resid, descent);

  queue->wait_and_throw();

  int itr_cg;

  for (itr_cg = 0;
       normhR > config.epsilon && normhR / origNormhR > config.epsilon &&
       itr_cg < config.mxiters;
       ++itr_cg) {
    std::cout << "Iteration " << itr_cg << " normhR=" << normhR << std::endl;

    data.mult(queue, descent, Adescent);
    T alpha = norm_resid / dot_product(queue, d_config, descent, Adescent);

    queue->wait_and_throw();

    inc_ay(queue, data.u, alpha, descent);
    inc_ay(queue, resid, -alpha, Adescent);

    old_norm_resid = norm_resid;
    norm_resid = dot_product(queue, d_config, resid, resid);
    normhR = sqrt(norm_resid * data.m_dx * data.m_dy);

    queue->wait_and_throw();

    if (normhR < config.epsilon)
      break;

    T beta = norm_resid / old_norm_resid;
    desc_update(queue, descent, beta, resid);
  }

  std::cout << "Converged in " << itr_cg << " iters nh resid=" << normhR
            << std::endl;
}

// Useful functors
class CoarsenA {};
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
    SYCL_ModelData<T> sycl_modeldata(solution);
    cl::sycl::buffer<T, 2> u(
        cl::sycl::range<2>(sycl_modeldata.ndy - 1, sycl_modeldata.ndx - 1));
    cl::sycl::buffer<T, 2> f(
        cl::sycl::range<2>(sycl_modeldata.ndy - 1, sycl_modeldata.ndx - 1));

    initU(device_queue, sycl_modeldata, u);
    initRHS(device_queue, sycl_modeldata, f);

    if (true) {
      // CG Setup
      MGLevel<T> mat_data(problem.ndx, problem.ndy, problem.hx, problem.hy);

      mat_data.initAfromK(device_queue, sycl_modeldata.k);
      copy_vec(device_queue, u, mat_data.u);
      copy_vec(device_queue, f, mat_data.rhs);

      basic_CG(device_queue, config, d_config, mat_data);

      copy_vec(device_queue, mat_data.u, u);

    } else {
      // Multigrid setup
      std::vector<MGLevel<T>> grids;
      grids.push_back(
          MGLevel<T>(problem.ndx, problem.ndy, problem.hx, problem.hy));
      while (grids.back().m_ndx > 2 && grids.back().m_ndy > 2) {
        grids.push_back(grids.back().coarseLevel());
      }

      // Do necessary init for MG
      auto front_grid = grids.front();
      front_grid.initAfromK(device_queue, sycl_modeldata.k);

      // Initialize Coarsened matricies
      // Solution computation
    }

    writeTemperature(device_queue, sycl_modeldata, u);

    device_queue->wait_and_throw();
  }
}
