#include <H5Cpp.h>

#include "model_data.h"

namespace Model_IO {
const H5std_string DX_DATA_NAME("dx");
const H5std_string DY_DATA_NAME("dy");
const H5std_string TEMPERATURE_DATA_NAME("temperature");
const H5std_string K_DATA_NAME("K");
const H5std_string HEAT_SOURCE_DATA_NAME("source");
const H5std_string TEMP_BND_DATA_NAME("source");

template <typename T>
void write_solution(const std::string outputfn,
                    const struct Model_Data::SolutionState<T> &state) {

  try {
    H5::H5File outputf(outputfn.c_str(), H5F_ACC_TRUNC);

    // Create proplist
    double fill_value = 0.0;
    H5::DSetCreatPropList plist;
    plist.setFillValue(H5::PredType::NATIVE_DOUBLE, &fill_value);

    // Store hx,hy
    {
      H5::DataSpace hx_fspace = H5::DataSpace();

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
      hsize_t state_fdim[] = {(hsize_t)state.ndx + 1, (hsize_t)state.ndy + 1};
      H5::DataSpace temperature_fspace = H5::DataSpace(1, state_fdim);

      H5::DataSet temperature_data(outputf.createDataSet(
          TEMPERATURE_DATA_NAME, H5::PredType::NATIVE_DOUBLE,
          temperature_fspace, plist));

      temperature_data.write(state.temperature.data(),
                             H5::PredType::NATIVE_DOUBLE);

      H5::DataSpace k_fspace = H5::DataSpace(1, state_fdim);

      H5::DataSet k_data(outputf.createDataSet(
          K_DATA_NAME, H5::PredType::NATIVE_DOUBLE, k_fspace, plist));

      k_data.write(state.k.data(), H5::PredType::NATIVE_DOUBLE);

      H5::DataSpace source_fspace = H5::DataSpace(1, state_fdim);

      H5::DataSet source_data(outputf.createDataSet(HEAT_SOURCE_DATA_NAME,
                                                    H5::PredType::NATIVE_DOUBLE,
                                                    source_fspace, plist));

      source_data.write(state.heat_source.data(), H5::PredType::NATIVE_DOUBLE);
    }

  } catch (H5::FileIException error) {
    error.printErrorStack();
    exit(-1);
  }
}

template <typename T>
void read_problem(const std::string inputfn,
                Model_Data::ProblemState<T> &to_return) {
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
      H5::DataSet k_data = inputf.openDataSet(K_DATA_NAME);
      H5::DataSpace k_fspace = k_data.getSpace();
      k_fspace.getSimpleExtentDims(dims, NULL);

      to_return.resize(dims[0] - 1, dims[1] - 1, dx, dy);

      // Read state data
      {
        const hsize_t ndx = to_return.ndx + 1, ndy = to_return.ndy + 1;
        u = new double[ndx * ndy];
        k_data.read(u, H5::PredType::NATIVE_DOUBLE);
        for (hsize_t j = 0; j < ndx; ++j) {
          for (hsize_t i = 0; i < ndy; ++i) {
            to_return.k[j * ndy + i] = u[j * ndy + i];
          }
        }
        delete[] u;
      }

      H5::DataSet f_data = inputf.openDataSet(K_DATA_NAME);
      {
        const hsize_t ndx = to_return.ndx - 1, ndy = to_return.ndy - 1;
        u = new double[ndx * ndy];
        f_data.read(u, H5::PredType::NATIVE_DOUBLE);
        for (hsize_t j = 0; j < ndx; ++j) {
          for (hsize_t i = 0; i < ndy; ++i) {
            to_return.heat_source[j * ndy + i] = u[j * ndy + i];
          }
        }
        delete[] u;
      }

      H5::DataSet bnd_data = inputf.openDataSet(TEMP_BND_DATA_NAME);
      {
        hsize_t nbnd = 2 * (dims[0] + dims[1]);
        u_bnd = new double[nbnd];
        bnd_data.read(u_bnd, H5::PredType::NATIVE_DOUBLE);
        for (hsize_t i = 0; i < nbnd; ++i) {
          to_return.temperature_bnd[i] = u_bnd[i];
        }
        delete[] u_bnd;
      }
    }

  } catch (H5::FileIException error) {
    error.printErrorStack();
    exit(-1);
  }
}

template <typename T>
void write_vis_metadata(const std::string metafn, const std::string heavy_fn,
                        const Model_Data::SolutionState<T> &state) {
  write_solution(heavy_fn, state);

  std::ofstream ofile;
  ofile.open(metafn);
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
  ofile << "        " << heavy_fn << ":/" << K_DATA_NAME << "\n";
  ofile << "      </DataItem>\n";
  ofile << "    </Attribute>\n";
  ofile << "    <Attribute Type=\"Scalar\" Name=\"F\" Center=\"Node\">\n";
  ofile << "      <DataItem Format=\"HDF\" NumberType=\"Float\" "
           "Precision=\"8\" Dimensions=\""
        << state.ndx << " " << state.ndy << "\">\n";
  ofile << "        " << heavy_fn << ":/" << HEAT_SOURCE_DATA_NAME << "\n";
  ofile << "      </DataItem>\n";
  ofile << "    </Attribute>\n";
  ofile << "    <Attribute Type=\"Scalar\" Name=\"U\" Center=\"Node\">\n";
  ofile << "      <DataItem Format=\"HDF\" NumberType=\"Float\" "
           "Precision=\"8\" Dimensions=\""
        << state.ndx << " " << state.ndy << "\">\n";
  ofile << "        " << heavy_fn << ":/" << TEMPERATURE_DATA_NAME << "\n";
  ofile << "      </DataItem>\n";
  ofile << "    </Attribute>\n";
  ofile << "  </Grid>\n";
  ofile << "</Xdmf>\n";
}

template
void write_solution<double>(
    const std::string outputfn,
    const Model_Data::SolutionState<double> &state);
template
void read_problem<double>(const std::string inputfn,
                        Model_Data::ProblemState<double> &to_return);
template
void write_vis_metadata<double>(const std::string metafn,
                                const std::string heavy_fn,
                                const Model_Data::SolutionState<double> &state);
} // namespace Model_IO
