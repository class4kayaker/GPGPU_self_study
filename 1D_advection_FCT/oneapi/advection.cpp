#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

#include <CL/sycl.hpp>
#include <vector>

#include "../utils/advection_output.h"
#include "../utils/advection_utils.h"

void do_computation(const FCT_initialization::ProblemConfig, FCT_initialization::InitState);

int main(int argc, char *argv[]) {
  FCT_initialization::ProblemConfig config =
      FCT_initialization::parse_args(argc, argv);

  FCT_initialization::InitState external_state;

  FCT_output::read_state(config.hdf5_init_fn, external_state);

  config.compute_timestep(external_state.time, external_state.dx);

  const int ndx = external_state.ndx;

  // Do computation
  {
    do_computation(config, external_state);
  }
  double time = 0.0; // Add code to time computation

  // Print results (problem size, time and bandwidth in GB/s).
  printf("Time for %d timestep computation %g s\n", config.ndt, time);

  FCT_output::write_state(config.hdf5_end_fn, external_state);

  return 0;
}

class SetStateBC;
class CalcLowFlux;
class CalcHighFlux;
class LowFluxUpdate;
class CalcFCTFLux;
class FullUpdate;

void do_computation(const FCT_initialization::ProblemConfig config, FCT_initialization::InitState external_state){
    unsigned int ndx = external_state.ndx;
    std::vector<cl::sycl::cl_double> host_u_state(ndx+4);
    std::vector<cl::sycl::cl_double> host_flux_low(ndx+1);
    std::vector<cl::sycl::cl_double> host_flux_high(ndx+1);
    std::vector<cl::sycl::cl_double> host_adiff_flux(ndx+1);
    std::vector<cl::sycl::cl_double> host_flux_c(ndx+1);

    cl::sycl::range<1> state_size{ndx+4}, flux_size{ndx+1};

    cl::sycl::buffer<cl::sycl::cl_double, 1> u_state(host_u_state.data(), state_size);
    cl::sycl::buffer<cl::sycl::cl_double, 1> flux_low(host_flux_low.data(), flux_size);
    cl::sycl::buffer<cl::sycl::cl_double, 1> flux_high(host_flux_high.data(), flux_size);
    cl::sycl::buffer<cl::sycl::cl_double, 1> adiff_flux(host_adiff_flux.data(), flux_size);
    cl::sycl::buffer<cl::sycl::cl_double, 1> flux_c(host_flux_c.data(), flux_size);

    cl::sycl::default_selector device_selector;
    std::unique_ptr<cl::sycl::queue> device_queue;

    try{
        device_queue.reset( new cl::sycl::queue(device_selector));
    } catch (cl::sycl::exception const& e) {
        std::cout << "Caught a synchronous SYCL exception:" << std::endl << e.what() << std::endl;
        return;
    }

    std::cout << "Device: "
            << device_queue->get_device().get_info<cl::sycl::info::device::name>()
            << std::endl;

    // Copy in data
    for (int i=0; i<ndx; ++i){
         host_u_state[i+2] = external_state.u[i];
    }

    for(unsigned int timestep = 0; timestep < config.ndt; ++timestep){
        // Set BCs
        // Compute fluxes
        // Compute diff flux
        // Do low update
        // Calc FCT flux
        // Do full update
    }

    external_state.time = config.end_time;

    // Copy out into u
    for (int i=0; i<ndx; ++i){
        external_state.u[i] = host_u_state[i+2];
    }
}
