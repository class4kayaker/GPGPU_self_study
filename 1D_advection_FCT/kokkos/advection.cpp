#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

#include <Kokkos_Core.hpp>

#include "../utils/advection_utils.h"

int main(int argc, char *argv[]) {
  double end_time = 0.0;
  const double sigma = 0.9;
  const double a = 3.0;

  FCT_initialization::ProblemConfig config = FCT_initialization::parse_args(argc, argv);

  const int ndx = config.ndx;

  // Check sizes.
  
  struct FCT_initialization::InitState init(ndx);

  FCT_initialization::sine_init(init, config, config.init_time);

  config.compute_timestep(init.dx);

  Kokkos::initialize(argc, argv);
  {
    // Allocate y, x vectors and Matrix A:
    double *const u_state = new double[ndx+2*2];
    double *const flux_low = new double[ndx+1];
    double *const flux_high = new double[ndx+1];
    double *const flux_c = new double[ndx+1];

    // Initialize U
    for (int i=0; i<ndx; ++i){
        u_state[i+2] = init.u[i];
    }

    // State update loop
    for (int timestep=0; timestep<config.ndt; ++timestep)
    {
        // Compute low order flux

        // Compute high order flux

        // Compute FCT flux
        
        // Do update
    }

    // Initialize y vector.
    Kokkos::parallel_for("y_init", N, KOKKOS_LAMBDA(int i) { y[i] = 1; });

    // Initialize x vector.
    Kokkos::parallel_for("x_init", M, KOKKOS_LAMBDA(int i) { x[i] = 1; });

    // Initialize A matrix, note 2D indexing computation.
    Kokkos::parallel_for("matrix_init", N, KOKKOS_LAMBDA(int j) {
      for (int i = 0; i < M; ++i) {
        A[j * M + i] = 1;
      }
    });

    // Timer products.
    Kokkos::Timer timer;

    for (int repeat = 0; repeat < nrepeat; repeat++) {
      // Application: <y,Ax> = y^T*A*x
      double result = 0;

      Kokkos::parallel_reduce("yAx", N,
                              KOKKOS_LAMBDA(int j, double &update) {
                                double temp2 = 0;

                                for (int i = 0; i < M; ++i) {
                                  temp2 += A[j * M + i] * x[i];
                                }

                                update += y[j] * temp2;
                              },
                              result);

      // Output result.
      const double solution = (double)N * (double)M;

      if (result != solution) {
        printf("  Error: result( %lf ) != solution( %lf )\n", result, solution);
      }
    }

    double time = timer.seconds();

    // Calculate bandwidth.
    // Each matrix A row (each of length M) is read once.
    // The x vector (of length M) is read N times.
    // The y vector (of length N) is read once.
    // double Gbytes = 1.0e-9 * double( sizeof(double) * ( 2 * M * N + N ) );
    double Gbytes = 1.0e-9 * double(sizeof(double) * (M + M * N + N));

    // Print results (problem size, time and bandwidth in GB/s).
    printf("  N( %d ) M( %d ) nrepeat ( %d ) problem( %g MB ) time( %g s ) "
           "bandwidth( %g GB/s )\n",
           N, M, nrepeat, Gbytes * 1000, time, Gbytes * nrepeat / time);

    delete[] A;
    delete[] y;
    delete[] x;
  }
  Kokkos::finalize();

  return 0;
}
