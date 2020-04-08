import numpy


class FDM_Error:
    def __init__(self, computed, true, eps=1e-10):
        self.computed = computed
        self.true = true
        assert abs(self.computed.dx - self.true.dx) < eps * abs(
            self.computed.dx
        )
        if abs(self.computed.time - self.true.time) >= eps:
            raise Exception(
                f"Time mismatch {self.computed.time:g}!={self.true.time:g}"
            )

    def L1_err(self):
        diff = self.computed.u - self.true.u
        return self.computed.dx * numpy.sum(numpy.abs(diff))

    def L2_err(self):
        diff = self.computed.u - self.true.u
        return numpy.sqrt(self.computed.dx * numpy.sum(diff * diff))

    def Linf_err(self):
        diff = self.computed.u - self.true.u
        return numpy.amax(numpy.abs(diff))

    def pprint_string(self):
        return (
            f"Errors: "
            f"L1 {self.L1_err():.6e} "
            f"L2 {self.L2_err():.6e} "
            f"Linf {self.Linf_err():.6e}"
        )
