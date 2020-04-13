#include <H5Cpp.h>

#include "advection_output.h"

namespace FCT_output {
const H5std_string TIME_DATA_NAME("time");
const H5std_string DX_DATA_NAME("dx");
const H5std_string STATE_DATA_NAME("state");

void write_state(const std::string outputfn,
                 const struct FCT_initialization::InitState &state) {

  try {
    H5::H5File outputf(outputfn.c_str(), H5F_ACC_TRUNC);

    // Create proplist
    double fill_value = 0.0;
    H5::DSetCreatPropList plist;
    plist.setFillValue(H5::PredType::NATIVE_DOUBLE, &fill_value);

    // Store time
    {
      H5::DataSpace time_fspace = H5::DataSpace();

      H5::DataSet time_data(outputf.createDataSet(
          TIME_DATA_NAME, H5::PredType::NATIVE_DOUBLE, time_fspace, plist));

      time_data.write(&(state.time), H5::PredType::NATIVE_DOUBLE);
    }

    // Store dx
    {
      H5::DataSpace dx_fspace = H5::DataSpace();

      H5::DataSet dx_data(outputf.createDataSet(
          DX_DATA_NAME, H5::PredType::NATIVE_DOUBLE, dx_fspace, plist));

      dx_data.write(&(state.dx), H5::PredType::NATIVE_DOUBLE);
    }

    // Store state data
    {
      hsize_t state_fdim[] = {(hsize_t)state.ndx};
      H5::DataSpace state_fspace = H5::DataSpace(1, state_fdim);

      H5::DataSet state_data(outputf.createDataSet(
          STATE_DATA_NAME, H5::PredType::NATIVE_DOUBLE, state_fspace, plist));

      state_data.write(state.u.data(), H5::PredType::NATIVE_DOUBLE);
    }
  } catch (H5::FileIException error) {
    error.printError();
    exit(-1);
  }
}

void read_state(const std::string inputfn,
                FCT_initialization::InitState &to_return) {
  try {
    H5::H5File inputf(inputfn.c_str(), H5F_ACC_RDONLY);

    // Vars for attribs
    hsize_t ndx = 0;
    double dx = 0.0;
    double time = 0.0;
    double *u;

    // Read time
    {
      H5::DataSet time_data = inputf.openDataSet(TIME_DATA_NAME);

      time_data.read(&time, H5::PredType::NATIVE_DOUBLE);
    }

    // Read dx
    {
      H5::DataSet dx_data = inputf.openDataSet(DX_DATA_NAME);

      dx_data.read(&dx, H5::PredType::NATIVE_DOUBLE);
    }

    {
      // Get data size
      H5::DataSet state_data = inputf.openDataSet(STATE_DATA_NAME);
      H5::DataSpace state_fspace = state_data.getSpace();
      state_fspace.getSimpleExtentDims(&ndx, NULL);

      u = new double[ndx];

      // Read state data
      state_data.read(u, H5::PredType::NATIVE_DOUBLE);
    }

    to_return.ndx = ndx;
    to_return.dx = dx;
    to_return.time = time;
    to_return.u.resize(ndx);

    for (hsize_t i = 0; i < ndx; ++i) {
      to_return.u[i] = u[i];
    }
    delete[] u;
  } catch (H5::FileIException error) {
    error.printError();
    exit(-1);
  }
}

void write_hdf5_double_vector(H5::H5File &outputf, const H5std_string &space_name, const std::vector<double> &data){
    const double fill_value = 0.0;
    H5::DSetCreatPropList plist;
    plist.setFillValue(H5::PredType::NATIVE_DOUBLE, &fill_value);

    hsize_t data_fdim[] ={(hsize_t)data.size()};
    H5::DataSpace fspace = H5::DataSpace(1, data_fdim);

    H5::DataSet h5_data(outputf.createDataSet(space_name, H5::PredType::NATIVE_DOUBLE, fspace, plist));

    h5_data.write(data.data(), H5::PredType::NATIVE_DOUBLE);
}

void write_step_state(const std::string outputfn,
                 const struct FCT_initialization::StepState &state) {

  try {
    H5::H5File outputf(outputfn.c_str(), H5F_ACC_TRUNC);


    // Store state data
    write_hdf5_double_vector(outputf, "u_state", state.u_state);
    write_hdf5_double_vector(outputf, "flux_low", state.flux_low);
    write_hdf5_double_vector(outputf, "flux_high", state.flux_high);
    write_hdf5_double_vector(outputf, "adiff_flux", state.adiff_flux);
    write_hdf5_double_vector(outputf, "flux_c", state.flux_c);
  } catch (H5::FileIException error) {
    error.printError();
    exit(-1);
  }
}

} // namespace FCT_output
