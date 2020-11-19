/**
 * End/init state recording utilities
 */

#ifndef ADVECTION_OUTPUT
#define ADVECTION_OUTPUT

#include <string>

#include "model_utils.h"

namespace Model_IO {

template<typename T>
void read_problem(const std::string inputfn, Model_Data::ProblemState<T> &state);

template<typename T>
void write_solution(const std::string outputfn, const Model_Data::SolutionState<T> &state);

template<typename T>
void write_vis_metadata(const std::string metafn, const std::string heavy_fn, const Model_Data::SolutionState<T> &state);

} // namespace FCT_output

#endif
