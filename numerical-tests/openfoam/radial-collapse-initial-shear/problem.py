"""Physical initial data for Section 5.5.2; coordinates here use OpenFOAM order."""
import numpy as np

DOMAIN_M = (100.0, 2.0, 100.0)
CENTER_M = 50.0
DAM_RADIUS_M = 15.0
INNER_DEPTH_M = 1.5
OUTER_DEPTH_M = 1.0
SHEAR_RADIUS_M = 12.0
PEAK_MEAN_SPEED_M_S = 0.4
COSINE_AMPLITUDES = (0.3, 0.2)
GRAVITY_M_S2 = 9.81
RHO_WATER = 1000.0
RHO_AIR = 1.0
VERTICAL_CELLS = (200, 400, 800)
OUTPUT_TIMES_S = (0.0, 1.0, 2.0, 3.0)
PROFILE_POINTS_M = ((55.0, 50.0), (60.0, 50.0), (65.0, 50.0))


def depth(x, z):
    return np.where((x-CENTER_M)**2 + (z-CENTER_M)**2 <= DAM_RADIUS_M**2,
                    INNER_DEPTH_M, OUTER_DEPTH_M)


def mean_velocity(x, z):
    """Compact counterclockwise swirl, with exactly the specified peak speed."""
    X, Z = np.asarray(x)-CENTER_M, np.asarray(z)-CENTER_M
    cutoff = np.maximum(1.0-(X*X+Z*Z)/SHEAR_RADIUS_M**2, 0.0)**3
    omega = PEAK_MEAN_SPEED_M_S * np.sqrt(7.0)*(7.0/6.0)**3 / SHEAR_RADIUS_M
    return -omega*Z*cutoff, omega*X*cutoff


def vertical_shape(s):
    a, b = COSINE_AMPLITUDES
    return 1.0 + a*np.cos(np.pi*s) + b*np.cos(2.0*np.pi*s)


def initial_velocity(x, y, z):
    """Water cosine shear, extended C1 into air and tapered to zero at the top."""
    h = depth(x, z)
    eta = np.clip((y-h)/(DOMAIN_M[1]-h), 0.0, 1.0)
    air_shape = vertical_shape(1.0)*(1.0-3.0*eta**2+2.0*eta**3)
    shape = np.where(y <= h, vertical_shape(y/h), air_shape)
    u, v = mean_velocity(x, z)
    u, v = np.broadcast_arrays(u*shape, v*shape)
    return np.stack((u, np.zeros_like(u), v), axis=-1)


def initial_model_state(x, y, order):
    """Pointwise conservative G-SWLME data in manuscript horizontal (x,y).

    For finite-volume initialization, quadrature-average this common continuous
    state over each horizontal cell; these are not precomputed cell averages.
    """
    if order not in (0, 1, 2):
        raise ValueError("this test compares moment orders 0, 1, 2")
    h = depth(x, y)
    u, v = mean_velocity(x, y)
    fields = [h, h*u, h*v]
    a, b = COSINE_AMPLITUDES
    for factor in (12.0*a/np.pi**2, 15.0*b/np.pi**2)[:order]:
        fields.extend((h*factor*u, h*factor*v))
    return np.stack(np.broadcast_arrays(*fields), axis=-1)


def case_settings(nvert, nx=400, nz=400):
    return {
        "created": "2026-09-09", "status": "prepared; not run",
        "mesh_cells_openfoam_xyz": [nx, nvert, nz],
        "domain_m": list(DOMAIN_M), "center_m": CENTER_M,
        "dam_radius_m": DAM_RADIUS_M,
        "inner_depth_m": INNER_DEPTH_M, "outer_depth_m": OUTER_DEPTH_M,
        "shear_radius_m": SHEAR_RADIUS_M,
        "peak_mean_speed_m_s": PEAK_MEAN_SPEED_M_S,
        "cosine_amplitudes": list(COSINE_AMPLITUDES),
        "gravity_m_s2": GRAVITY_M_S2, "rho_water": RHO_WATER,
        "rho_air": RHO_AIR, "nu_water_m2_s": 1e-6,
        "nu_air_m2_s": 1.48e-5, "surface_tension_n_m": 0.07,
        "max_co": 0.25, "max_alpha_co": 0.25, "write_precision": 12,
        "initial_dt_s": 0.001, "end_time_s": 3.0,
        "write_interval_s": 0.5,
    }
