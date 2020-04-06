"""Utilities to validate a FDM implementation based on providing
and checking the error using HDF5 files generated"""

__version__ = "0.0.1"

import sys
import argparse
import numpy
import h5py


class FDM_Problem_Config:
    def __init__(self, ndx, a=3.0, sigma=0.9):
        self.ndx = ndx
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
            dx = f['dx']
            time = f['time']
            u = f['state']
            return cls(u, dx, time)

    def to_h5(self, h5fn):
        with h5py.File(h5fn, "w") as f:
            f['dx'] = self.dx
            f['time'] = self.time
            f['state'] = self.u

    @classmethod
    def sine_init(cls, config, time=0.0):
        ndx = config.ndx
        dx = 1.0/ndx
        x = numpy.linspace(0.0, 1.0, num=ndx+1)
        u = numpy.sin(2.0*numpy.pi*(0.5*(x[1:]-x[:-1])-config.a*time))
        cls(u, dx)


def parse_args(argv):
    pass


def main():
    pass
