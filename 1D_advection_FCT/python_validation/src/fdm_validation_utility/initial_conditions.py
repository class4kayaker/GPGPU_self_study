import numpy

from fdm_validation_utility import FDM_Advection_State


def square_wave(x):
    return numpy.where((x >= 0.25) & (x <= 0.75), 1.0, 0.0)


def square_wave_exc(x):
    return numpy.where((x > 0.25) & (x < 0.75), 1.0, 0.0)


def semicircle(x):
    return numpy.sqrt(0.25 - (x - 0.5) ** 2)


def gaussian(x):
    return numpy.exp(-256.0 * (x - 0.5) ** 2)


def sinusoidal(x):
    return 0.5 * (1 - numpy.cos(2 * numpy.pi * x))


init_by_name = {
    "Sinusoidal": lambda config, ndx, time=0.0: FDM_Advection_State.func_init(
        config, ndx, sinusoidal, time
    ),
    "Semicircle": lambda config, ndx, time=0.0: FDM_Advection_State.func_init(
        config, ndx, semicircle, time
    ),
    "Gaussian": lambda config, ndx, time=0.0: FDM_Advection_State.func_init(
        config, ndx, gaussian, time
    ),
    "Square_Wave": lambda config, ndx, time=0.0: FDM_Advection_State.func_init(
        config, ndx, square_wave, time
    ),
}
