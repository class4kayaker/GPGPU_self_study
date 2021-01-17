import numpy
import h5py


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

    def to_h5(self, h5fn):
        with h5py.File(h5fn, "w") as f:
            f["dx"] = self.dx
            f["dy"] = self.dy
            f["K"] = self.k
            f["source"] = self.heat_source
            f["temperature"] = self.temperature

    @classmethod
    def from_h5(cls, h5fn):
        with h5py.File(h5fn, "r") as f:
            dx = f["dx"][()]
            dy = f["dy"][()]
            k = f["K"][:, :]
            ndx, ndy = k.shape
            ret = cls(ndx, ndy, dx, dy)
            ret.k = k
            ret.heat_source = f["source"][:, :]
            ret.temperature = f["temperature"][:, :]
            return ret

    def get_pos(self):
        x = numpy.linspace(0.0, self.ndx * self.dx, self.ndx + 1)
        y = numpy.linspace(0.0, self.ndy * self.dy, self.ndy + 1)
        return numpy.meshgrid(x, y)
