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

  const int ndx = config.ndx;

  // Check sizes.
  FCT_initialization::check_bounds(config);

  struct FCT_initialization::InitState init(ndx);

  FCT_initialization::sine_init(init, config, config.init_time);

  FCT_output::write_state("start_data.h5", init);

  config.compute_timestep(init.dx);

  Kokkos::initialize(argc, argv);
  {
    // Allocate y, x vectors and Matrix A
    Kokkos::View<double *> u_state("u_state", ndx + 2 * 2);
    Kokkos::View<double *> flux_low("flux_low", ndx + 1);
    Kokkos::View<double *> flux_high("flux_high", ndx + 1);
    Kokkos::View<double *> adiff_flux("adiff_flux", ndx + 1);
    Kokkos::View<double *> flux_c("flux_c", ndx + 1);

    // Initialize U
    for (int i = 0; i < ndx; ++i) {
      u_state(i + 2) = init.u[i];
    }

    // Timer products.
    Kokkos::Timer timer;

    // State update loop
    for (int timestep = 0; timestep < config.ndt; ++timestep) {
      const double dtdx = (config.dt / init.dx);
      const double dxdt = (init.dx / config.dt);

      // Set BCs
      u_state( 0 ) = u_state(ndx + 1);
      u_state( 1 ) = u_state(ndx + 2);
      u_state(ndx + 3) = u_state(2);
      u_state(ndx + 4) = u_state(3);

      // Compute fluxes
      Kokkos::parallel_for("calc_low_flux", ndx + 1, KOKKOS_LAMBDA(int i) {
        flux_low(i) = config.a * u_state(i + 1);
      });

      // Compute high order flux
      const double sigma_i = (config.a * config.dt / init.dx);
      Kokkos::parallel_for("calc_high_flux", ndx + 1, KOKKOS_LAMBDA(int i) {
        flux_high(i) =
            config.a * 0.5 *
            ((1 + sigma_i) * u_state(i + 1) + (1 - sigma_i) * u_state(i + 2));
      });

      // Compute diff flux
      Kokkos::parallel_for("calc_diff_flux", ndx + 1, KOKKOS_LAMBDA(int i) {
        adiff_flux(i) = flux_high(i) - flux_low(i);
      });

      // Do update with low flux
      Kokkos::parallel_for("low_flux_update", ndx, KOKKOS_LAMBDA(int i) {
        u_state(i + 2) += dtdx * (flux_low(i) - flux_low(i + 1));
      });

      // Set BCs
      u_state( 0 ) = u_state(ndx + 1);
      u_state( 1 ) = u_state(ndx + 2);
      u_state(ndx + 3) = u_state(2);
      u_state(ndx + 4) = u_state(3);

      // Compute FCT flux
      Kokkos::parallel_for("calc_fct_flux", ndx + 1, KOKKOS_LAMBDA(int i) {
        double sign_a = copysign(1.0, adiff_flux(i));
        double left_d = u_state(i + 1) - u_state(i + 0);
        double right_d = u_state(i + 3) - u_state(i + 2);
        flux_c(i) =
            sign_a * std::max(0.0, std::min(std::min(sign_a * dxdt * left_d,
                                                     sign_a * dxdt * right_d),
                                            abs(adiff_flux(i))));
      });

      // Do update
      Kokkos::parallel_for("full_update", ndx, KOKKOS_LAMBDA(int i) {
        u_state(i + 2) += dtdx * (flux_c(i) - flux_c(i + 1));
      });
    }
    double time = timer.seconds();

    struct FCT_initialization::InitState end_state(ndx);
    end_state.dx = init.dx;
    end_state.time = config.end_time;

    for(size_t i = 0; i < ndx; ++i){
        end_state.u[i] = u_state(i+2);
    }

    FCT_output::write_state("end_data.h5", end_state);

    // Print results (problem size, time and bandwidth in GB/s).
    printf("Time for %d timestep computation %g s\n", config.ndt, time);
  }
  Kokkos::finalize();

  return 0;
}
