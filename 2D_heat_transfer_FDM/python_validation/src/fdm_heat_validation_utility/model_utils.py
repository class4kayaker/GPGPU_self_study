import numpy
import h5py


class FDM_Problem_Config:
    def __init__(self, ndx, ndy, dx, dy):
        self.ndx = ndx
        self.ndy = ndy
        self.dx = dx
        self.dy = dy

        full_ndx = ndx + 1
        full_ndy = ndy + 1
        full_bnd = 2 * (ndx + ndy)

        self.k = numpy.ndarray((full_ndx, full_ndy))
        self.heat_source = numpy.ndarray((full_ndx, full_ndy))
        self.temperature_bnd = numpy.ndarray((full_bnd,))

    def to_h5(self, h5fn):
        with h5py.File(h5fn, "w") as f:
            f["dx"] = self.dx
            f["dy"] = self.dy
            f["K"] = self.k
            f["source"] = self.heat_source
            f["bnd_temp"] = self.temperature_bnd

    def get_pos(self):
        x = numpy.linspace(0.0, self.ndx * self.dx, self.ndx + 1)
        y = numpy.linspace(0.0, self.ndy * self.dy, self.ndy + 1)
        return numpy.meshgrid(x, y)

    def get_bnd_pos(self):
        x = numpy.linspace(0.0, self.ndx * self.dx, self.ndx + 1)
        y = numpy.linspace(0.0, self.ndy * self.dy, self.ndy + 1)
        xbnd = numpy.concatenate(
            x,
            x[-1] * numpy.ones((self.ndy - 1,)),
            x[::-1],
            numpy.zeroes((self.ndy - 1,)),
        )
        ybnd = numpy.concatenate(
            y[-1] * numpy.ones((self.ndx - 1,)),
            y,
            numpy.zeroes((self.ndx - 1,)),
            y[::-1],
        )
        return (xbnd, ybnd)


class FDM_State:
    def __init__(self, ndx, ndy, dx, dy):
        self.ndx = ndx
        self.ndy = ndy
        self.dx = dx
        self.dy = dy

        full_ndx = ndx + 1
        full_ndy = ndy + 1
        state_shape = (full_ndx, full_ndy)

        self.k = numpy.ndarray(state_shape)
        self.heat_source = numpy.ndarray(state_shape)
        self.temperature = numpy.ndarray(state_shape)

    def get_pos(self):
        x = numpy.linspace(0.0, self.ndx * self.dx, self.ndx + 1)
        y = numpy.linspace(0.0, self.ndy * self.dy, self.ndy + 1)
        return numpy.meshgrid(x, y)
