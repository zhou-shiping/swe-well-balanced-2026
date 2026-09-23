"""Section 5.5.2 data; x,y are horizontal and z is physical vertical."""
import numpy as np

LENGTH_M = 100.0
CENTER_M = 50.0
DAM_RADIUS_M = 15.0
INNER_DEPTH_M = 1.5
OUTER_DEPTH_M = 1.0
SHEAR_RADIUS_M = 12.0
PEAK_MEAN_SPEED_M_S = 0.4
GRAVITY_M_S2 = 9.81
OUTPUT_TIMES_S = (0.0, 1.0, 2.0, 3.0)
PROFILE_POINTS_M = ((55.0, 50.0), (60.0, 50.0), (65.0, 50.0))
# (2j+1) integral_0^1 f(s) phi_j(s) ds, for phi_j(s)=P_j(1-2s).
MOMENT_FACTORS = (3.6/np.pi**2, 3.0/np.pi**2, 50.4*(np.pi**2-10)/np.pi**4)
# Physical specification copied from the independently prepared OpenFOAM case.
OPENFOAM_PROBLEM_SHA256 = "366b8811c018914c946c80fe142f2f8958babbd9415c41b9460a4e93a37f2c12"


def moment_factors(order):
    """Project the fixed cosine profile; preserve the legacy first three modes."""
    if not isinstance(order, (int, np.integer)) or order < 0:
        raise ValueError("moment order must be a nonnegative integer")
    if order <= 3:
        return np.array(MOMENT_FACTORS[:order])
    nodes, weights = np.polynomial.legendre.leggauss(max(64, order+8))
    s = (nodes+1)/2
    shape = 1+.3*np.cos(np.pi*s)+.2*np.cos(2*np.pi*s)
    basis = np.polynomial.legendre.legvander(-nodes, order)
    factors = (basis[:, 1:].T @ (weights*shape/2))*(2*np.arange(1, order+1)+1)
    factors[:3] = MOMENT_FACTORS
    return factors


def mean_velocity(x, y):
    X, Y = np.asarray(x)-CENTER_M, np.asarray(y)-CENTER_M
    cutoff = np.maximum(1-(X*X+Y*Y)/SHEAR_RADIUS_M**2, 0)**3
    omega = PEAK_MEAN_SPEED_M_S*np.sqrt(7)*(7/6)**3/SHEAR_RADIUS_M
    return -omega*Y*cutoff, omega*X*cutoff


def pointwise_state(x, y, order):
    factors = moment_factors(order)
    h = np.where((x-CENTER_M)**2+(y-CENTER_M)**2 <= DAM_RADIUS_M**2,
                 INNER_DEPTH_M, OUTER_DEPTH_M)
    u, v = mean_velocity(x, y)
    fields = [h, h*u, h*v]
    for factor in factors:
        fields.extend((factor*h*u, factor*h*v))
    return np.stack(np.broadcast_arrays(*fields), axis=-1)


def disk_primitive(x, y):
    """Signed area between the axes and (x,y) inside the radius-R disk.

    The rectangle integral follows by inclusion-exclusion. This analytically
    averages the discontinuous depth, without sampling the circular jump.
    """
    R = DAM_RADIUS_M
    a, b = np.minimum(abs(x), R), np.minimum(abs(y), R)
    cut = np.minimum(a, np.sqrt(np.maximum(R*R-b*b, 0)))
    def integral(t):
        return 0.5*(t*np.sqrt(np.maximum(R*R-t*t, 0))+R*R*np.arcsin(t/R))
    return np.sign(x)*np.sign(y)*(b*cut+integral(a)-integral(cut))


def initial_state(nx, ny, order, quadrature=8):
    """Conservative cell averages; analytic disk area and tensor Gauss velocity."""
    factors = moment_factors(order)
    if nx < 4 or ny < 4 or quadrature < 2:
        raise ValueError("need nx,ny>=4 and quadrature>=2")
    dx, dy = LENGTH_M/nx, LENGTH_M/ny
    x, y = (np.arange(nx)+0.5)*dx, (np.arange(ny)+0.5)*dy
    edges_x = np.linspace(-CENTER_M, LENGTH_M-CENTER_M, nx+1)
    edges_y = np.linspace(-CENTER_M, LENGTH_M-CENTER_M, ny+1)
    area = disk_primitive(edges_x[:, None], edges_y[None, :])
    fraction = np.clip(np.diff(np.diff(area, axis=0), axis=1)/(dx*dy), 0, 1)
    state = np.zeros((nx, ny, 3+2*order))
    state[..., 0] = OUTER_DEPTH_M+(INNER_DEPTH_M-OUTER_DEPTH_M)*fraction
    nodes, weights = np.polynomial.legendre.leggauss(quadrature)
    # The velocity support r<12 lies strictly inside the depth plateau r<15.
    for xi, wi in zip(nodes, weights):
        for eta, wj in zip(nodes, weights):
            u, v = mean_velocity(x[:, None]+xi*dx/2, y[None, :]+eta*dy/2)
            state[..., 1] += INNER_DEPTH_M*wi*wj*u/4
            state[..., 2] += INNER_DEPTH_M*wi*wj*v/4
    for j, factor in enumerate(factors):
        state[..., 3+2*j] = factor*state[..., 1]
        state[..., 4+2*j] = factor*state[..., 2]
    return x, y, state
