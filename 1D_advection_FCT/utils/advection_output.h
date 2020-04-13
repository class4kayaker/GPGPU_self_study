/**
 * End/init state recording utilities
 */

#ifndef ADVECTION_OUTPUT
#define ADVECTION_OUTPUT

#include <string>

#include "advection_utils.h"

namespace FCT_output {

void write_state(const std::string outputfn, const struct FCT_initialization::InitState &state);

void read_state(const std::string inputfn, FCT_initialization::InitState &state);

void write_step_state(const std::string outputfn, const struct FCT_initialization::StepState &state);
} // namespace FCT_output

#endif
