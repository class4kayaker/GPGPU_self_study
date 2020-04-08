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
    "Sinusoidal": lambda config, time=0.0: FDM_Advection_State.func_init(
        config, sinusoidal, time
    ),
    "Semicircle": lambda config, time=0.0: FDM_Advection_State.func_init(
        config, semicircle, time
    ),
    "Gaussian": lambda config, time=0.0: FDM_Advection_State.func_init(
        config, gaussian, time
    ),
    "Square_Wave": lambda config, time=0.0: FDM_Advection_State.func_init(
        config, square_wave, time
    ),
}
