#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

#include <Kokkos_Core.hpp>

#include "../utils/advection_output.h"

int main(int argc, char *argv[]) {
  double end_time = 0.0;
  const double sigma = 0.9;
  const double a = 3.0;

  FCT_initialization::ProblemConfig config =
      FCT_initialization::parse_args(argc, argv);

  struct FCT_initialization::InitState init;

  FCT_output::read_state(config.hdf5_init_fn, init);

  config.compute_timestep(init.time, init.dx);

  const int ndx = init.ndx;

  Kokkos::initialize(argc, argv);
  {
    // typedef Kokkos::Serial   ExecSpace;
    // typedef Kokkos::Threads  ExecSpace;
    typedef Kokkos::OpenMP ExecSpace;
    // typedef Kokkos::Cuda ExecSpace;
    // typedef Kokkos::ROCm ExecSpace;

    // Allocate y, x vectors and Matrix A
    typedef Kokkos::View<double *> VectorViewType;
    VectorViewType u_state("u_state", ndx + 2 * 2);
    VectorViewType::HostMirror host_u_state =
        Kokkos::create_mirror_view(u_state);
    Kokkos::View<double *> flux_low("flux_low", ndx + 1);
    Kokkos::View<double *> flux_high("flux_high", ndx + 1);
    Kokkos::View<double *> adiff_flux("adiff_flux", ndx + 1);
    Kokkos::View<double *> flux_c("flux_c", ndx + 1);

    // Initialize U
    for (unsigned int i = 0; i < ndx; ++i) {
      host_u_state(i + 2) = init.u[i];
    }

    Kokkos::deep_copy(u_state, host_u_state);

    // Timer products.
    Kokkos::Timer timer;

    // State update loop
    for (int timestep = 0; timestep < config.ndt; ++timestep) {
      const double dtdx = (config.dt / init.dx);
      const double dxdt = (init.dx / config.dt);

      // Set BCs
      Kokkos::parallel_for("set_state_bc", Kokkos::RangePolicy<ExecSpace>(0, 4),
                           KOKKOS_LAMBDA(int i) {
                             if (i < 2) {
                               u_state(i) = u_state(ndx + i);
                             } else {
                               u_state(ndx + i) = u_state(i);
                             }
                           });

      // Compute fluxes
      Kokkos::parallel_for(
          "calc_low_flux", Kokkos::RangePolicy<ExecSpace>(0, ndx + 1),
          KOKKOS_LAMBDA(int i) { flux_low(i) = config.a * u_state(i + 1); });

      // Compute high order flux
      const double sigma_i = (config.a * config.dt / init.dx);
      Kokkos::parallel_for("calc_high_flux",
                           Kokkos::RangePolicy<ExecSpace>(0, ndx + 1),
                           KOKKOS_LAMBDA(int i) {
                             flux_high(i) = config.a * 0.5 *
                                            ((1 + sigma_i) * u_state(i + 1) +
                                             (1 - sigma_i) * u_state(i + 2));
                           });

      // Compute diff flux
      Kokkos::parallel_for(
          "calc_diff_flux", Kokkos::RangePolicy<ExecSpace>(0, ndx + 1),
          KOKKOS_LAMBDA(int i) { adiff_flux(i) = flux_high(i) - flux_low(i); });

      // Do update with low flux
      Kokkos::parallel_for(
          "low_flux_update", Kokkos::RangePolicy<ExecSpace>(0, ndx),
          KOKKOS_LAMBDA(int i) {
            u_state(i + 2) += dtdx * (flux_low(i) - flux_low(i + 1));
          });

      // Set BCs
      Kokkos::parallel_for("set_state_bc", Kokkos::RangePolicy<ExecSpace>(0, 4),
                           KOKKOS_LAMBDA(int i) {
                             if (i < 2) {
                               u_state(i) = u_state(ndx + i);
                             } else {
                               u_state(ndx + i) = u_state(i);
                             }
                           });

      // Compute FCT flux
      Kokkos::parallel_for(
          "calc_fct_flux", Kokkos::RangePolicy<ExecSpace>(0, ndx + 1),
          KOKKOS_LAMBDA(int i) {
            double sign_a = copysign(1.0, adiff_flux(i));
            double left_d = u_state(i + 1) - u_state(i + 0);
            double right_d = u_state(i + 3) - u_state(i + 2);
            flux_c(i) =
                sign_a *
                std::max(0.0, std::min(std::min(sign_a * dxdt * left_d,
                                                sign_a * dxdt * right_d),
                                       abs(adiff_flux(i))));
          });

      // Do update
      Kokkos::parallel_for(
          "full_update", Kokkos::RangePolicy<ExecSpace>(0, ndx),
          KOKKOS_LAMBDA(int i) {
            u_state(i + 2) += dtdx * (flux_c(i) - flux_c(i + 1));
          });
    }
    double time = timer.seconds();

    Kokkos::deep_copy(host_u_state, u_state);

    struct FCT_initialization::InitState end_state(ndx);
    end_state.dx = init.dx;
    end_state.time = config.end_time;

    for (unsigned int i = 0; i < ndx; ++i) {
      end_state.u[i] = host_u_state(i + 2);
    }

    FCT_output::write_state("end_data.h5", end_state);

    // Print results (problem size, time and bandwidth in GB/s).
    printf("Time for %d timestep computation %g s\n", config.ndt, time);
  }
  Kokkos::finalize();

  return 0;
}
