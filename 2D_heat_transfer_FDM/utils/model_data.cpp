#include <H5Cpp.h>

#include "advection_output.h"

namespace FCT_output {
const H5std_string DX_DATA_NAME("dx");
const H5std_string DY_DATA_NAME("dy");
const H5std_string TEMPERATURE_DATA_NAME("temperature");
const H5std_string CONDUCTIVITY_DATA_NAME("conductivity");
const H5std_string HEAT_SOURCE_DATA_NAME("source");
const H5std_string TEMP_BND_DATA_NAME("source");

void write_solution(const std::string outputfn,
                    const struct Model_output::SolutionState &state) {

  try {
    H5::H5File outputf(outputfn.c_str(), H5F_ACC_TRUNC);

    // Create proplist
    double fill_value = 0.0;
    H5::DSetCreatPropList plist;
    plist.setFillValue(H5::PredType::NATIVE_DOUBLE, &fill_value);

    // Store hx,hy
    {
      H5::DataSpace hy_fspace = H5::DataSpace();

      H5::DataSet hx_data(outputf.createDataSet(
          DX_DATA_NAME, H5::PredType::NATIVE_DOUBLE, hx_fspace, plist));

      hx_data.write(&(state.hx), H5::PredType::NATIVE_DOUBLE);

      H5::DataSpace hy_fspace = H5::DataSpace();

      H5::DataSet hy_data(outputf.createDataSet(
          DY_DATA_NAME, H5::PredType::NATIVE_DOUBLE, hy_fspace, plist));

      hy_data.write(&(state.hy), H5::PredType::NATIVE_DOUBLE);
    }

    // Store state data
    {
      hsize_t state_fdim[] = {(hsize_t)state.ndx, (hsize_t)state.ndy};
      H5::DataSpace state_fspace = H5::DataSpace(1, state_fdim);

      H5::DataSet state_data(outputf.createDataSet(TEMPERATURE_DATA_NAME,
                                                   H5::PredType::NATIVE_DOUBLE,
                                                   state_fspace, plist));

      state_data.write(state.temperature.data(), H5::PredType::NATIVE_DOUBLE);
    }

  } catch (H5::FileIException error) {
    error.printErrorStack();
    exit(-1);
  }
}

void read_state(const std::string inputfn,
                FCT_initialization::InitState &to_return) {
  try {
    H5::H5File inputf(inputfn.c_str(), H5F_ACC_RDONLY);

    // Vars for attribs
    hsize_t dims[2];
    double dx = 0.0, dy = 0.0;
    double *u;
    double *u_bnd;

    // Read dx
    {
      H5::DataSet dx_data = inputf.openDataSet(DX_DATA_NAME);

      dx_data.read(&dx, H5::PredType::NATIVE_DOUBLE);

      H5::DataSet dy_data = inputf.openDataSet(DY_DATA_NAME);

      dy_data.read(&dy, H5::PredType::NATIVE_DOUBLE);
    }

    {
      // Get data size
      H5::DataSet k_data = inputf.openDataSet(CONDUCTIVITY_DATA_NAME);
      H5::DataSpace k_fspace = k_data.getSpace();
      k_fspace.getSimpleExtentDims(&dims, NULL);

      to_return.resize(dims[0], dims[1], dx, dy);

      u = new double[dims[0] * dims[1]];
      hsize_t nbnd = 2 * (dims[0] + dims[1] - 2);
      u_bnd = new double[nbnd];

      // Read state data
      k_data.read(u, H5::PredType::NATIVE_DOUBLE);
      for (hsize_t j = 0; j < to_return.ndx; ++j) {
        for (hsize_t i = 0; i < to_return.ndy; ++i) {
          to_return.conductivity[j * to_return.ndy + i] =
              u[j * to_return.ndy + i];
        }
      }

      H5::DataSet f_data = inputf.openDataSet(CONDUCTIVITY_DATA_NAME);
      f_data.read(u, H5::PredType::NATIVE_DOUBLE);
      for (hsize_t j = 0; j < to_return.ndx; ++j) {
        for (hsize_t i = 0; i < to_return.ndy; ++i) {
          to_return.conductivity[j * to_return.ndy + i] =
              u[j * to_return.ndy + i];
        }
      }

      H5::DataSet bnd_data = inputf.openDataSet(TEMP_BND_DATA_NAME);
      bnd_data.read(u_bnd, H5::PredType::NATIVE_DOUBLE);
      for (hsize_t i = 0; i < nbnd; ++i) {
        to_return.temperature_bnd[i] = u_bnd[i];
      }
    }

    delete[] u;
    delete[] u_bnd;
  } catch (H5::FileIException error) {
    error.printErrorStack();
    exit(-1);
  }
}

template <typename T>
void write_vis_metadata(const std::string outputfn,
                        const ProblemState<T> &state,
                        const std::string problemfn,
                        const std::string solutionfn) {
  std::ofstream ofile;
  ofile.open(outputfn);
  ofile << "<Xdmf Version=\"2.0\">\n";
  // Write grid
  ofile << "  <Grid>\n";
  ofile << "    <Topology TopologyType=\"2DCoRectMesh\" Dimensions=\""
        << state.ndx << " " << state.ndy << "\"/>\n";
  ofile << "    <Geometry GeometryType=\"Origin_DxDy\">\n";
  ofile << "      <DataItem Format=\"XML\" NumberType=\"Float\" "
           "Precision=\"8\" Dimensions=\"2\">\n";
  ofile << "        0. 0.\n";
  ofile << "      </DataItem>\n";
  ofile << "      <DataItem Format=\"XML\" NumberType=\"Float\" "
           "Precision=\"8\" Dimensions=\"2\">\n";
  ofile << "        " << state.hx << " " << state.hy << "\n";
  ofile << "      </DataItem>\n";
  ofile << "    </Geometry>\n";
  ofile << "    <Attribute Type=\"Scalar\" Name=\"K\" Center=\"Node\">\n";
  ofile << "      <DataItem Format=\"HDF\" NumberType=\"Float\" "
           "Precision=\"8\" Dimensions=\""
        << state.ndx << " " << state.ndy << "\">\n";
  ofile << "        " << problemfn << ":/" << CONDUCTIVITY_DATA_NAME << "\n";
  ofile << "      </DataItem>\n";
  ofile << "    </Attribute>\n";
  ofile << "    <Attribute Type=\"Scalar\" Name=\"F\" Center=\"Node\">\n";
  ofile << "      <DataItem Format=\"HDF\" NumberType=\"Float\" "
           "Precision=\"8\" Dimensions=\""
        << state.ndx << " " << state.ndy << "\">\n";
  ofile << "        " << problemfn << ":/" << HEAT_SOURCE_DATA_NAME << "\n";
  ofile << "      </DataItem>\n";
  ofile << "    </Attribute>\n";
  ofile << "    <Attribute Type=\"Scalar\" Name=\"U\" Center=\"Node\">\n";
  ofile << "      <DataItem Format=\"HDF\" NumberType=\"Float\" "
           "Precision=\"8\" Dimensions=\""
        << state.ndx << " " << state.ndy << "\">\n";
  ofile << "        " << solutionfn << ":/" << TEMPERATURE_DATA_NAME << "\n";
  ofile << "      </DataItem>\n";
  ofile << "    </Attribute>\n";
  ofile << "  </Grid>\n";
  ofile << "</Xdmf>\n";
}

} // namespace FCT_output
