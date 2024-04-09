import numpy as np
import scipy as sp


class ReservoirSim:
    """A small Reservoir Simulator.


    """

    def __init__(self, perm_field, phi=0.2, c_f=1e-5, p0=1, rho0=1, mu_w=1,
                 rw=0.15, pres_ini=150, wells=None, dxy=50, dz=10):
        """Initialize a Simulation instance.

        Parameters
        ----------
        perm_field : 2D array
            Permeabilities ny-by-nx (?)
        phi : float
            Porosity (-)
        c_f : float
            Formation compressibility (1/kPa)
        p0 : float
            Initial pressure (bar or kPa?)
        rho0 : float
            Fixed density (kg/m3)
        mu_w : float
            Viscosity (cP - Pa s)
        rw : float
            Well radius (m)
        pres_ini : initial pressure [?]
        wells : location and pressure of wells [?]
        dxy, dz : floats
            Cell dimensions (m); dx=dy=dxy.

        """

        self.size = perm_field.size
        self.shape = perm_field.shape
        self.nx, self.ny = perm_field.shape
        self.perm_field = perm_field.ravel('F')

        self.phi = phi
        self.c_f = c_f
        self.p0 = p0
        self.rho0 = rho0
        self.mu_w = mu_w
        self.rw = rw
        self.dxy = dxy
        self.dx = self.dxy
        self.dy = self.dxy
        self.dz = dz
        self.pres_ini = pres_ini

        # Store volumes (needs adjustment for arbitrary cell volumes)
        self.volumes = np.ones(self.size) * self.dx * self.dy * self.dz

        if wells is None:
            self.wells = np.array([[0, 0, 180], [self.nx-1, self.ny-1, 120]])
        else:
            self.wells = np.array(wells)

        # Get well locations
        self.locs = self.wells[:, 1]*self.nx + self.wells[:, 0]

        # Get well terms (formula will need adjustment for dx!=dy).
        wi = 2 * np.pi * self.perm_field[self.locs] * self.dz
        wi /= self.mu_w * np.log(0.208 * self.dxy / self.rw)

        # Add wells
        self._add_wells_f = self.wells[:, 2] * wi
        self._add_wells_d = wi

    def solve(self, pressure, dt):
        """Construct K-matrix."""

        # Mobility ratio without permeability
        phi = self.rho0 * (1 + self.c_f * (pressure - self.p0)) / self.mu_w

        # Compr. and right-hand side f
        compr = self.volumes * self.phi * self.c_f / dt
        f = compr * pressure

        # Pre-allocate diagonals.
        mn = np.zeros(self.size)
        m1 = np.zeros(self.size)
        d = compr
        p1 = np.zeros(self.size)
        pn = np.zeros(self.size)

        t1 = self.dy * self.perm_field[:-1] * self.perm_field[1:]
        t1 /= self.perm_field[:-1] + self.perm_field[1:]
        t1 *= (phi[:-1] + phi[1:]) / 2
        t1[self.nx-1::self.nx] = 0.0
        d[:-1] += t1
        d[1:] += t1
        m1[:-1] -= t1
        p1[1:] -= t1

        t2 = self.dx * self.perm_field[:-self.nx] * self.perm_field[self.nx:]
        t2 /= self.perm_field[:-self.nx] + self.perm_field[self.nx:]
        t2 *= (phi[:-self.nx] + phi[self.nx:]) / 2
        d[:-self.nx] += t2
        d[self.nx:] += t2
        mn[:-self.nx] -= t2
        pn[self.nx:] -= t2

        # Add wells.
        f[self.locs] += self._add_wells_f
        d[self.locs] += self._add_wells_d

        # Bring to sparse matrix
        offsets = np.array([-self.nx, -1, 0, 1, self.nx])
        data = np.array([mn, m1, d, p1, pn])
        K = sp.sparse.dia_array((data, offsets), shape=(self.size, self.size))

        # Solve the system
        return sp.sparse.linalg.spsolve(K.tocsc(), f, use_umfpack=False)

    def __call__(self, dt=np.ones(10)*0.0001):
        """Run simulator.

        Parameters
        ----------
        dt : array
            Time steps.

        """

        pressure = np.ones((dt.size+1, self.size)) * self.pres_ini
        for i, d in enumerate(dt):
            pressure[i+1, :] = self.solve(pressure[i, :], d)

        return pressure.reshape((dt.size+1, *self.shape), order='F')


def index2ij(index, nx, ny):
    """Convert index numeration to ij-index."""
    return ((index % nx) + 1, (index // nx) + 1)


def ij2index(i, j, nx, ny):
    """Convert ij numeration to index."""
    return (i-1) + (j-1)*nx


# TODO 0: Also implement Sphere function
# TODO 1: Ensure it is the same as before
# TODO 2: It could be further speedup:
#         the first loop is only necessary for i=1
def build_perm_cov_matrix(nx, ny, length, theta, sigma_pr2):
    cost = np.cos(theta)
    sint = np.sin(theta)
    cov = np.zeros([nx*ny, nx*ny])
    xx = [((i % nx) + 1, (i // nx) + 1) for i in range(nx*ny)]
    for i in range(nx):
        x0 = xx[i]
        for j in range(nx*ny):
            x1 = xx[j]
            d0 = x1[0]-x0[0]
            d1 = x1[1]-x0[1]
            rot0 = cost*d0 - sint*d1
            rot1 = sint*d0 + cost*d1

            # Gaspari Cohn TODO get powers of, w\o sqrt
            hl = np.sqrt((rot0/length[0])**2 +
                         (rot1/length[1])**2)

            if hl < 1:
                cov[i, j] = (-(hl**5)/4 + (hl**4)/2 + (hl**3)*5/8 -
                             (hl**2)*5/3 + 1)
            elif hl >= 1 and hl < 2:
                cov[i, j] = ((hl**5)/12 - (hl**4)/2 + (hl**3)*5/8 +
                             (hl**2)*5/3 - hl*5 + 4 - (1/hl)*2/3)
    for j in range(1, ny):
        cov[nx*j:nx*(j+1), nx*j:] = cov[:nx, :-nx*j]
        for i in range(j):
            cov[nx*j:nx*(j+1), nx*(j-i-1):nx*(j-i)] = cov[:nx, nx*i:nx*(i+1)]

    return cov
