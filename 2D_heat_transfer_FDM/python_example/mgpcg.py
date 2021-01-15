import time
import math
import numpy as np
import lxml
import lxml.etree
import h5py

# Record base 2D stencil used
# For reference only
stencil2d = np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])

# Set interpolation 'stencil': Defines expansion from coarse grid cell to
# surrounding corresponding 3x3 in the fine grid
iSten2d = np.array([[0.25, 0.5, 0.25], [0.50, 1.0, 0.50], [0.25, 0.5, 0.25]])

# Set full-weight stencil based on transpose of interpolation
# Require that when applied to a constant 3x3, will be that
# constant
fSten2d = iSten2d / iSten2d.sum()


# Class to hold matrix coeff. Stored in directional form to remove conversion as
# a but source
class ALoc:
    # Create 'local' 3x3 matrix storage class
    def __init__(self, n):
        self.n = n
        self.cen = np.zeros((n - 1, n - 1), dtype=np.float64)
        self.xdir = np.zeros((n - 1, n - 2), dtype=np.float64)
        self.ydir = np.zeros((n - 2, n - 1), dtype=np.float64)
        self.xpydir = np.zeros((n - 2, n - 2), dtype=np.float64)
        self.xmydir = np.zeros((n - 2, n - 2), dtype=np.float64)

    # Set based on std stencil
    def setStencil(self):
        self.cen[:, :] = 4.0 * self.n ** 2
        self.xdir[:, :] = -1.0 * self.n ** 2
        self.ydir[:, :] = -1.0 * self.n ** 2
        self.xpydir[:, :] = 0.0
        self.xmydir[:, :] = 0.0

    # Set matrix based on provided 'viscosity' k
    def setFromK(self, k):
        snx, sny = k.shape
        h = 1.0 / (snx - 1)
        denom = 1.0 / (2.0 * h ** 2)
        # Set diag elements
        self.cen[:, :] = 0.0
        self.cen += 4.0 * k[1:-1, 1:-1] * denom
        self.cen += 1.0 * k[1:-1, 0:-2] * denom  # k_x-
        self.cen += 1.0 * k[1:-1, 2:] * denom  # k_x+
        self.cen += 1.0 * k[0:-2, 1:-1] * denom  # k_y-
        self.cen += 1.0 * k[2:, 1:-1] * denom  # k_y+
        # Set x-dir
        self.xdir[:, :] = -(k[1:-1, 1:-2] + k[1:-1, 2:-1]) * denom
        # Set y-dir
        self.ydir[:, :] = -(k[1:-2, 1:-1] + k[2:-1, 1:-1]) * denom
        # All other components are 0
        self.xpydir *= 0.0
        self.xmydir *= 0.0

    # 'Multiply' Au
    def mult(self, u):
        # References given relative to i,j
        ret = np.zeros(u.shape, dtype=np.float64)
        # diag
        ret += self.cen * u
        # X-dir
        ret[:, 1:] += self.xdir * u[:, :-1]  # Contrib from -0
        ret[:, :-1] += self.xdir * u[:, 1:]  # Contrib from +0
        # Y-dir
        ret[1:, :] += self.ydir * u[:-1, :]  # Contrib from 0-
        ret[:-1, :] += self.ydir * u[1:, :]  # Contrib from 0+
        # ++-dir
        ret[1:, 1:] += self.xpydir * u[:-1, :-1]  # Contrib from --
        ret[:-1, :-1] += self.xpydir * u[1:, 1:]  # Contrib from ++
        # +--dir
        ret[:-1, 1:] += self.xmydir * u[1:, :-1]  # Contrib from -+
        ret[1:, :-1] += self.xmydir * u[:-1, 1:]  # Contrib from +-
        return ret

    # Return result of one iteration of weighted jacobi using provided u, f
    def jacobi(self, u, f, weight=1.0):
        offdiag = self.mult(u) - self.cen * u
        return (1.0 - weight) * u + weight * (f - offdiag) / self.cen

    # Apply one iteration of gauss-seidel using  u,f to u
    def gauss(self, u, f):
        for j in range(self.n - 1):
            for i in range(self.n - 1):
                resid = 0.0

                if i > 0:
                    resid += self.xdir[j, i - 1] * u[j, i - 1]

                if i < self.n - 2:
                    resid += self.xdir[j, i] * u[j, i + 1]

                if j > 0:
                    resid += self.ydir[j - 1, i] * u[j - 1, i]

                if j < self.n - 2:
                    resid += self.ydir[j, i] * u[j + 1, i]

                if i > 0 and j > 0:
                    resid += self.xpydir[j - 1, i - 1] * u[j - 1, i - 1]

                if i < self.n - 2 and j < self.n - 2:
                    resid += self.xpydir[j, i] * u[j + 1, i + 1]

                if i > 0 and j < self.n - 2:
                    resid += self.xpydir[j, i - 1] * u[j + 1, i - 1]

                if i < self.n - 2 and j > 0:
                    resid += self.xpydir[j - 1, i] * u[j - 1, i + 1]

                u[j, i] = (f[j, i] - resid) / self.cen[j, i]

    # Apply one iteration of 4 color gauss-seidel using  u,f to u
    def gauss4c(self, u, f):
        uC = np.zeros(u.shape, dtype=np.float64)
        for joff in range(2):
            for ioff in range(self.n):
                uC[:, :] = u
                uC[joff::2, ioff::2] = 0.0
                colorvals = (f - self.mult(uC)) / self.cen
                u[joff::2, ioff::2] = colorvals[joff::2, ioff::2]

    # Do direct solve if only 1 data point in main mesh
    def dirSolve(self, f):
        if self.n == 2:
            return f[:, :] / self.cen[:, :]
        return None

    # Do coarsening based on interpolation and full weighting
    def coarsen(self):
        # Create matrix based on half size mesh
        newN = self.n // 2
        ret = ALoc(newN)
        inVec = np.zeros((self.n - 1, self.n - 1), dtype=np.float64)
        # For each location in a 3x3 block
        for joff in range(3):
            for ioff in range(3):
                # 3x3 coarse blocks are not close enough to interfere with
                # calculation, so set these based on interpolation
                inVec[:, :] = 0.0
                for j in range(joff, newN - 1, 3):
                    jst, jen = 2 * j, 2 * j + 3
                    for i in range(ioff, newN - 1, 3):
                        ist, ien = 2 * i, 2 * i + 3
                        inVec[jst:jen, ist:ien] = iSten2d

                # Multiply produced vector by A
                v = self.mult(inVec)

                # Calculation coefficients for coarsened matrix based on full
                # weighting.
                for j in range(joff, newN - 1, 3):
                    jst, jen = 2 * j, 2 * j + 3
                    for i in range(ioff, newN - 1, 3):
                        ist, ien = 2 * i, 2 * i + 3

                        # Diagonal
                        ret.cen[j, i] = (v[jst:jen, ist:ien] * fSten2d).sum()

                        if i < newN - 2:
                            # X-Dir
                            ret.xdir[j, i] = (
                                v[jst:jen, ist + 2 : ien + 2] * fSten2d
                            ).sum()

                        if j < newN - 2:
                            # Y-Dir
                            ret.ydir[j, i] = (
                                v[jst + 2 : jen + 2, ist:ien] * fSten2d
                            ).sum()

                        if i < newN - 2 and j < newN - 2:
                            # XpY-Dir
                            ret.xpydir[j, i] = (
                                v[jst + 2 : jen + 2, ist + 2 : ien + 2] * fSten2d
                            ).sum()

                        if i < newN - 2 and j > 0:
                            # XmY-Dir
                            ret.xmydir[j - 1, i] = (
                                v[jst - 2 : jen - 2, ist + 2 : ien + 2] * fSten2d
                            ).sum()

        return ret


# Class to store matrix, u, and f for a particular grid level
class Grid2d:
    # Create storage class for nxn mesh
    def __init__(self, n):
        self.nx = self.ny = self.n = n
        self.hx = self.hy = 1.0 / n
        self.A = ALoc(n)
        self.u = np.zeros((n - 1, n - 1), dtype=np.float64)
        self.f = np.zeros((n - 1, n - 1), dtype=np.float64)

    # Calculate residual for grid.
    def resid(self):
        return self.f - self.A.mult(self.u)

    # Do count weighted jacobi iterations
    def jacobi(self, count, weight=1.0):
        n = 0
        while n < count:
            self.u = self.A.jacobi(self.u, self.f, weight)
            n += 1

    # Do count G-S iterations
    def gauss(self, count):
        n = 0
        while n < count:
            self.A.gauss(self.u, self.f)
            n += 1

    # Do count 4 color G-S iterations
    def gauss4c(self, count):
        n = 0
        while n < count:
            self.A.gauss4c(self.u, self.f)
            n += 1

    # Do direct solve
    def dirSolve(self):
        if self.A and self.n == 2:
            self.u[:, :] = self.A.dirSolve(self.f)

    # Set f based on u, useful for testing that methods are converging correctly
    def setFromU(self, u=None):
        if u is None:
            self.f = self.A.mult(self.u)
        else:
            self.f = self.A.mult(u)

    # Convenience method, returns correct 'vector' shape
    def shape(self):
        return self.u.shape


# Class to handle multigrid interaction aspects
class MGGrid2d:
    # Create storage for Multi-Grid grids
    def __init__(self, n):
        self.grid = Grid2d(n)
        self.coarg = None
        self.fineg = None

    # Create coarser grid based on this grid
    def genCoarseGrid(self, count=1):
        if count < 1:
            return
        if self.coarg is None:
            self.coarg = MGGrid2d(self.grid.n // 2)
            self.coarg.fineg = self
        self.coarg.grid.A = self.grid.A.coarsen()
        if count > 1 or self.coarg is not None:
            self.coarg.genCoarseGrid(count - 1)

    # General relaxation selection method, because only jacobi implemented not
    # as useful as it might be
    def sweepdown(self, count=1, swtype="WJ"):
        if swtype == "WJ":
            self.grid.jacobi(count, 2.0 / 3.0)
        elif swtype == "GS":
            self.grid.gauss(count)
        elif swtype == "4CGS":
            self.grid.gauss4c(count)

    # Do v-cycle based on generated grids
    def vcycle(self, nu1=2, nu2=1, swtype="WJ"):
        if self.coarg is None:
            if self.grid.n == 2:
                self.grid.dirSolve()
            else:
                self.sweepdown(nu1 + nu2, swtype)
        else:
            self.sweepdown(nu1, swtype)
            self.coarg.grid.u[:, :] = 0.0
            resids = self.grid.resid()
            self.coarg.grid.f = fullWeight2d(resids)
            self.coarg.vcycle(nu1, nu2, swtype)
            self.grid.u = self.grid.u + interpolate2d(self.coarg.grid.u)
            self.sweepdown(nu2, swtype)


def interpolate2d(array):
    """Function to produce a 2n-1 x 2m-1 array from a given nxm array with
    interpolated values"""
    onx, ony = array.shape
    nx, ny = 2 * (onx + 1) - 1, 2 * (onx + 1) - 1
    narray = np.zeros((nx, ny), dtype=np.float64)
    for yi in range(ony):
        for xi in range(onx):
            add = array[yi, xi] * iSten2d
            narray[2 * yi : 2 * yi + 3, 2 * xi : 2 * xi + 3] += add
    return narray


def fullWeight2d(array):
    """Function to produce a n/2+1 x m/2+1 array from a given nxm array with
    values based on full weighting"""
    onx, ony = array.shape
    nx, ny = int((onx + 1) / 2), int((ony + 1) / 2)
    narray = np.zeros((ny - 1, nx - 1), dtype=np.float64)
    for yi in range(ny - 1):
        for xi in range(nx - 1):
            arr = array[2 * yi : 2 * yi + 3, 2 * xi : 2 * xi + 3]
            narray[yi, xi] = (arr * fSten2d).sum()
    return narray


# Calculated h-norm for 'vector'
def normh(a):
    h = 1.0
    if a.ndim == 1:
        (n,) = a.shape
        h = 1.0 / n
    elif a.ndim == 2:
        nx, ny = a.shape
        h = 1.0 / (nx * ny)

    return math.sqrt((h * a * a).sum())


# Calculated 2-norm for 'vector'
def norm2(a):
    return math.sqrt((a * a).sum())


# Old CA3 init, used for intermediate testing
def init_2d(u, f):
    snx, sny = u.shape
    nx, ny = snx + 1, sny + 1
    xh = np.linspace(0.0, 1.0, nx + 1)
    yh = np.linspace(0.0, 1.0, ny + 1)
    for yi in range(ny - 1):
        for xi in range(nx - 1):
            xl = xh[xi + 1]
            yl = yh[yi + 1]
            val = -2.0 * (
                (1.0 - 6.0 * xl ** 2.0) * yl ** 2 * (1.0 - yl ** 2.0)
                + (1.0 - 6.0 * yl ** 2.0) * xl ** 2 * (1.0 - xl ** 2.0)
            )
            tru = (xl ** 2 - xl ** 4) * (yl ** 2 - yl ** 4)
            f[yi, xi] = val
            u[yi, xi] = tru


# Problem 1 init for k, f
def initP1(k, f):
    snx, sny = f.shape
    nx = snx + 1
    xh = np.linspace(0.0, 1.0, nx + 1)
    k[:, :] = 1.0
    f[:, :] = 0.0
    f[-1:, :] = k[-1:, 1:-1] * 3.0 * xh[1:-1] * (1.0 - xh[1:-1])


# Problem 2 init for k, f
def initP2(k, f):
    snx, sny = f.shape
    n = snx + 1
    nd2 = n / 2
    nd8 = n / 8
    k[:, :] = 1.0
    k[5 * nd8 + 1 : n - nd8, nd8 + 1 : n - nd8] = 100.0
    k[nd8 + 1 : 5 * nd8 + 1, 3 * nd8 + 1 : 5 * nd8] = 100.0
    f[:, :] = -80
    f[: nd2 - 1, : nd2 - 1] = 80
    f[nd2 - 1, :] = 0
    f[:, nd2 - 1] = 0
    f[nd2:, nd2:] = 80


# Run MG iterations until convergence criterion met
def mgrun(mgg, u, f, mxiters=30, eps=1e-10, nu1=2, nu2=1, swtype="WJ"):
    mgg.grid.u = u
    mgg.grid.f = f
    normR = normh(mgg.grid.resid())
    origNormR = eps
    n = 0
    stime = time.time()
    while normR > eps and normR / origNormR > eps and n < mxiters:
        mgg.vcycle(nu1, nu2, swtype)
        n += 1
        normR = normh(mgg.grid.resid())

    etime = time.time()
    u[:, :] = mgg.grid.u
    return (n, normR, etime - stime)


# Run MGCG iterations until convergence criterion met
def mgcg(mgg, u, f, mxiters=30, eps=1e-10, nu1=2, nu2=1, swtype="WJ"):
    resid = f - mgg.grid.A.mult(u)
    mgg.grid.u[:, :] = 0.0
    mgg.grid.f = resid
    mgg.vcycle(nu1, nu2, swtype)
    residt = mgg.grid.u
    dirvec = residt
    normR = normh(resid)
    origNormR = eps
    anormR = (residt * resid).sum()
    n = 0
    stime = time.time()
    while normR > eps and normR / origNormR > eps and n < mxiters:
        Adirvec = mgg.grid.A.mult(dirvec)
        alpha = anormR / (dirvec * Adirvec).sum()
        u += alpha * dirvec
        resid -= alpha * Adirvec
        normR = normh(resid)
        n += 1
        if normR < eps:
            break
        mgg.grid.u[:, :] = 0.0
        mgg.grid.f = resid
        mgg.vcycle(nu1, nu2, swtype)
        residt = mgg.grid.u
        anormRold = anormR
        anormR = (residt * resid).sum()
        beta = anormR / anormRold
        dirvec = residt + beta * dirvec

    etime = time.time()
    return (n, normR, etime - stime)


# Convenience method to write results to nice CSV file
def writeResults(results, methods, outfile):
    fStrCSVh = ",nit{0:},fnR{0:},t{0:}"
    fStrCSVd = ",{:n},{:e},{:f}"
    fout = open(outfile, "w")
    # Header
    fout.write("size")
    for method in methods:
        fout.write(fStrCSVh.format(method))

    fout.write("\n")

    # Data
    sizes = results.keys()
    sizes.sort()
    for size in sizes:
        fout.write("{:n}".format(size))
        for method in methods:
            niter, normR, time = results[size][method]
            fout.write(fStrCSVd.format(niter, normR, time))

        fout.write("\n")

    fout.close()


# Write XDMF file
def writeSolution(k, u, f, fstem):
    # Create filenames
    xmf_fn = fstem + ".xmf"
    h5_fn = fstem + ".h5"

    # Get required numbers
    onx, ony = k.shape
    nx, ny = onx - 1, ony - 1
    hx, hy = 1.0 / nx, 1.0 / ny
    ufull = np.zeros((onx, ony))
    ffull = np.zeros((onx, ony))
    ufull[1:-1, 1:-1] = u
    ffull[1:-1, 1:-1] = f
    attribdict = {}
    attribdict["Format"] = "HDF"
    attribdict["NumberType"] = "Float"
    attribdict["Precision"] = "8"
    attribdict["Dimensions"] = "{} {}".format(ony, onx)

    # Open H5 file
    h5f = h5py.File(h5_fn, "w")
    # Build XMF file
    root = lxml.etree.Element("Xdmf")
    root.set("Version", "2.0")
    dom = lxml.etree.SubElement(root, "Domain")
    # Describe grid
    grid = lxml.etree.SubElement(dom, "Grid")
    topo = lxml.etree.SubElement(grid, "Topology")
    topo.set("TopologyType", "2DCoRectMesh")
    topo.set("Dimensions", "{} {}".format(ony, onx))
    geom = lxml.etree.SubElement(grid, "Geometry")
    geom.set("GeometryType", "Origin_DxDy")
    orig = lxml.etree.SubElement(geom, "DataItem")
    orig.set("Format", "XML")
    orig.set("Dimensions", "2")
    orig.text = "{} {}".format(0.0, 0.0)
    delt = lxml.etree.SubElement(geom, "DataItem")
    delt.set("Format", "XML")
    delt.set("Dimensions", "2")
    delt.text = "{} {}".format(hy, hx)

    # Write attributes
    elem = lxml.etree.SubElement(grid, "Attribute")
    elem.set("Type", "Scalar")
    elem.set("Name", "K")
    elem.set("Center", "Node")
    dat = lxml.etree.SubElement(elem, "DataItem", attrib=attribdict)
    dat.text = "{}:/K".format(h5_fn)
    h5f.create_dataset("K", data=k)

    elem = lxml.etree.SubElement(grid, "Attribute")
    elem.set("Type", "Scalar")
    elem.set("Name", "U")
    elem.set("Center", "Node")
    dat = lxml.etree.SubElement(elem, "DataItem", attrib=attribdict)
    dat.text = "{}:/U".format(h5_fn)
    h5f.create_dataset("U", data=ufull)

    elem = lxml.etree.SubElement(grid, "Attribute")
    elem.set("Type", "Scalar")
    elem.set("Name", "F")
    elem.set("Center", "Node")
    dat = lxml.etree.SubElement(elem, "DataItem", attrib=attribdict)
    dat.text = "{}:/F".format(h5_fn)
    h5f.create_dataset("F", data=ffull)

    tree = lxml.etree.ElementTree(root)
    tree.write(xmf_fn, encoding="UTF-8", xml_declaration=True, pretty_print=True)
    h5f.close()


if __name__ == "__main__":
    # Build storage dictionary for results
    llist = [4, 5, 6, 7, 8, 9, 10]
    probs = ["P1", "P2"]
    meths = ["MGCG", "MG"]
    results = {}
    for p in probs:
        results[p] = {}
        for l in llist:
            results[p][2 ** l] = {}

    # explicitly set convergence criterion
    mxiters = 300
    tol = 1e-8
    swtype = "WJ"
    print("Method {} tol = {:e}".format(swtype, tol))
    fS = "{:2} {:5} n={:5} niter={:5} final ||r||_h={:12e} time={:12f} sec"
    fnS = "soln{:2}{:}{:04d}"
    for l in llist:
        n = 2 ** l
        k = np.ones((n + 1, n + 1))
        u = np.ones((n - 1, n - 1))
        f = np.ones((n - 1, n - 1))
        mgg = MGGrid2d(n)
        baseA = ALoc(n)
        mgg.grid.A = baseA
        for prob in probs:
            if prob == "P1":
                initP1(k, f)
                mgg.grid.A.setFromK(k)
                mgg.genCoarseGrid(l - 1)
            elif prob == "P2":
                initP2(k, f)
                mgg.grid.A.setFromK(k)
                mgg.genCoarseGrid(l - 1)
            for meth in meths:
                u[:, :] = 0.0
                if meth == "MGCG":
                    niter, normR, ctime = mgcg(mgg, u, f, mxiters, tol, swtype=swtype)
                elif meth == "MG":
                    niter, normR, ctime = mgrun(mgg, u, f, mxiters, tol, swtype=swtype)

                print(fS.format(prob, meth, n, niter, normR, ctime))
                results[prob][n][meth] = (niter, normR, ctime)
                writeSolution(k, u, f, fnS.format(prob, meth, n))

    # Write results to csv file
    methods = ["MGCG", "MG"]
    for prob in results:
        outfile = "res{}.csv".format(prob)
        writeResults(results[prob], methods, outfile)
