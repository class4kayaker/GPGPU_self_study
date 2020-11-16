/**
 * End/init state recording utilities
 */

#ifndef ADVECTION_OUTPUT
#define ADVECTION_OUTPUT

#include <string>

#include "advection_utils.h"

namespace Model_IO {

template<typename T>
void read_problem(const std::string inputfn, const struct Model_Data::ProblemState<T> &state);

template<typename T>
void write_solution(const std::string outputfn, const struct Model_Data::SolutionState<T> &state);

template<typename T>
void write_vis_metadata(const std::string outputfn, const ProblemState<t> &state, const std::string problemfn, const std::string solutionfn);

} // namespace FCT_output

#endif
