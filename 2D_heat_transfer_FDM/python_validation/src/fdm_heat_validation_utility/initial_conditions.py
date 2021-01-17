from fdm_heat_validation_utility.model_utils import FDM_State

init_by_name = {}


def register_init(name, includes_soln=False):
    def internal_register(fn_call):
        init_by_name[name] = fn_call

    return internal_register


@register_init("MMS_polynomial", includes_soln=True)
def MMS_Sine(k: int) -> FDM_State:
    assert k > 0
    ndx = 2 ** k
    dx = 1.0 / ndx
    state = FDM_State(ndx, ndx, dx, dx)
    xv, yv = state.get_pos()
    state.k[:, :] = 1.0
    state.heat_source = -2.0 * (
        (1.0 - 6.0 * xv ** 2.0) * yv ** 2 * (1.0 - yv ** 2)
        + (1.0 - 6.0 * yv ** 2) * xv ** 2 * (1.0 - xv ** 2)
    )
    state.temperature = (xv ** 2 - xv ** 4) * (yv ** 2 - yv ** 4)
    return state


@register_init("MMS_linear", includes_soln=True)
def MMS_Sine(k: int) -> FDM_State:
    assert k > 0
    ndx = 2 ** k
    dx = 1.0 / ndx
    state = FDM_State(ndx, ndx, dx, dx)
    xv, yv = state.get_pos()
    state.k[:, :] = 1.0
    state.heat_source[:, :] = 0.0
    state.temperature = 0.5*xv+0.5*yv
    return state
