#ifndef ADVECTION_ALLOC_H
#define ADVECTION_ALLOC_H

void cuda_compute_fct(size_t ndx, double* u, size_t ndt, double dt, double dx, double a);

#endif
