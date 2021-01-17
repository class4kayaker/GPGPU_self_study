import numpy


class FDM_Diff:
    def __init__(self, state1, state2, eps=1e-10):
        self.state1 = state1
        self.state2 = state2
        assert abs(self.state1.dx - self.state2.dx) < eps * abs(
            self.state1.dx
        )
        assert abs(self.state1.dy - self.state2.dy) < eps * abs(
            self.state1.dx
        )

    def L2_diff(self):
        h = 1.0/(self.state1.dx * self.state1.dy)
        diff = self.state1.temperature - self.state2.temperature
        return numpy.sqrt(h * numpy.sum(diff * diff))

    def pprint_string(self):
        return (
            f"Diff: {self.L2_diff():.6e}"
        )
