import numpy
import h5py


class FDM_Problem_Config:
    def __init__(self, a=3.0, sigma=0.9):
        self.a = a
        self.sigma = sigma


class FDM_Advection_State:
    def __init__(self, u, dx, time=0.0):
        self.ndx = u.shape[0]
        self.dx = dx
        self.time = time
        self.u = numpy.copy(u)

    @classmethod
    def from_h5(cls, h5fn):
        with h5py.File(h5fn, "r") as f:
            dx = f["dx"][()]
            time = f["time"][()]
            u = f["state"]
            return cls(u, dx, time)

    def to_h5(self, h5fn):
        with h5py.File(h5fn, "w") as f:
            f["dx"] = self.dx
            f["time"] = self.time
            f["state"] = self.u

    @classmethod
    def func_init(cls, config, ndx, func, time=0.0):
        dx = 1.0 / ndx
        x = numpy.linspace(0.0, 1.0, num=ndx + 1)
        shift_x = 0.5 * (x[1:] + x[:-1]) - config.a * time
        n_shift_x = shift_x - numpy.floor(shift_x)
        u = func(n_shift_x)
        return cls(u, dx, time=time)
