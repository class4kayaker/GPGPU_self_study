#ifndef ADVECTION_ALLOC_H
#define ADVECTION_ALLOC_H

void cuda_compute_fct(const char *device_name, size_t ndx, size_t block_size,
                      double *u_extern, size_t ndt, double dt, double dx,
                      double a);

#endif
