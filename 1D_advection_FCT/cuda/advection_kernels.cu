#include "advection_alloc.h"

#include <stdio.h>

__global__ void set_periodic_bc(size_t ndx, double *u) {
  const size_t index = threadIdx.x;
  const size_t dst_index = index < 2 ? index : ndx + index;
  const size_t src_index = index < 2 ? ndx + index : index;
  if (index < 4) {
    u[dst_index] = u[src_index];
  }
}

__global__ void calc_low_flux(size_t ndx, double a, double *u, double *flux) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < ndx + 1) {
    flux[index] = a * u[index + 1];
  }
}

__global__ void calc_high_flux(size_t ndx, double a, double dx, double dt,
                               double *u, double *flux) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const double sigma_i = a * dt / dx;
  if (index < ndx + 1) {
    flux[index] =
        a * 0.5 * ((1 + sigma_i) * u[index + 1] + (1 - sigma_i) * u[index + 2]);
  }
}

__global__ void calc_diff_flux(size_t ndx, double *flx_low, double *flx_high,
                               double *flx_diff) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < ndx + 1) {
    flx_diff[index] = flx_high[index] - flx_low[index];
  }
}

__global__ void do_update(size_t ndx, double dt, double dx, double *flx,
                          double *u) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const double dtdx = dt / dx;
  if (index < ndx) {
    u[index + 2] += dtdx * (flx[index] - flx[index + 1]);
  }
}

__global__ void calc_fct_flux(size_t ndx, double a, double dx, double dt,
                              double *flux_diff, double *u, double *flux_c) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const double dxdt = dx / dt;
  if (index < ndx + 1) {
    const double sign_a = copysign(1.0, flux_diff[index]);
    const double left_d = u[index + 1] - u[index + 0];
    const double right_d = u[index + 3] - u[index + 2];
    flux_c[index] =
        sign_a *
        max(0.0, min(min(sign_a * dxdt * left_d, sign_a * dxdt * right_d),
                     abs(flux_diff[index])));
  }
}

void cuda_compute_fct(size_t ndx, size_t block_size, double *u_extern, size_t ndt, double dt,
                      double dx, double a) {
  const size_t state_size = (ndx + 4) * sizeof(double),
               flux_size = (ndx + 1) * sizeof(double);
  double *u_state, *flux_low, *flux_high, *adiff_flux, *flux_c;
  cudaMallocManaged(&u_state, state_size);
  cudaMallocManaged(&flux_low, flux_size);
  cudaMallocManaged(&flux_high, flux_size);
  cudaMallocManaged(&adiff_flux, flux_size);
  cudaMallocManaged(&flux_c, flux_size);

  // Copy in data
  for (size_t i = 0; i < ndx; ++i) {
    u_state[i + 2] = u_extern[i];
  }

  size_t state_block_count = (ndx + block_size - 1) / block_size,
         flx_block_count = (ndx + block_size) / block_size;

  {
    for (unsigned int timestep = 0; timestep < ndt; ++timestep) {
      // Set BCs
      set_periodic_bc<<<1, block_size>>>(ndx, u_state);
      cudaDeviceSynchronize();
      // Compute fluxes
      calc_low_flux<<<flx_block_count, block_size>>>(ndx, a, u_state, flux_low);
      calc_high_flux<<<flx_block_count, block_size>>>(ndx, a, dx, dt, u_state,
                                                      flux_high);
      cudaDeviceSynchronize();
      // Compute diff flux
      calc_diff_flux<<<flx_block_count, block_size>>>(ndx, flux_low, flux_high,
                                                      adiff_flux);
      cudaDeviceSynchronize();
      // Do low update
      do_update<<<state_block_count, block_size>>>(ndx, dt, dx, flux_low,
                                                   u_state);
      cudaDeviceSynchronize();
      // Set BC cells
      set_periodic_bc<<<1, block_size>>>(ndx, u_state);
      cudaDeviceSynchronize();
      // Calc FCT flux
      calc_fct_flux<<<flx_block_count, block_size>>>(ndx, a, dt, dx, adiff_flux,
                                                     u_state, flux_c);
      cudaDeviceSynchronize();
      // Do full update
      do_update<<<state_block_count, block_size>>>(ndx, dt, dx, flux_c,
                                                   u_state);
      cudaDeviceSynchronize();
    }
  }

  // Copy out into u
  for (size_t i = 0; i < ndx; ++i) {
    u_extern[i] = u_state[i + 2];
  }

  cudaFree(u_state);
  cudaFree(flux_low);
  cudaFree(flux_high);
  cudaFree(adiff_flux);
  cudaFree(flux_c);
}
