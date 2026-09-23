"""Unsplit path-conservative MUSCL/LLF + SSP-RK2 for frictionless 2D G-SWLME.

State: h,hu,hv,h*alpha_1,h*beta_1,...; array axes: x,y,component.
Fluxes follow manuscript F_L,G_L; Cx=P_1+Delta A and Cy=Q_1+Delta B.
Straight paths in conservative state are integrated with three-point Gauss.
Both face jumps and within-cell nonconservative integrals are retained.
"""
import numpy as np
from problem import GRAVITY_M_S2

PATH_NODES = ((1-np.sqrt(3/5))/2, 0.5, (1+np.sqrt(3/5))/2)
PATH_WEIGHTS = (5/18, 4/9, 5/18)


def validate(state):
    if not np.all(np.isfinite(state)) or np.min(state[..., 0]) <= 0:
        raise FloatingPointError("nonfinite state or nonpositive depth; no clipping applied")
    return state


def flux(state, axis, gravity=GRAVITY_M_S2):
    h = state[..., 0]
    velocity = state[..., 1:]/h[..., None]
    u, v = velocity[..., 0], velocity[..., 1]
    a, b = velocity[..., 2::2], velocity[..., 3::2]
    normal = u if axis == 0 else v
    modes = a if axis == 0 else b
    denominator = 2*np.arange(a.shape[-1])+3
    result = np.zeros_like(state)
    result[..., 0] = state[..., 1+axis]
    result[..., 1] = h*(normal*u+np.sum(modes*a/denominator, axis=-1))
    result[..., 2] = h*(normal*v+np.sum(modes*b/denominator, axis=-1))
    result[..., 1+axis] += 0.5*gravity*h*h
    result[..., 3::2] = h[..., None]*(normal[..., None]*a+u[..., None]*modes)
    result[..., 4::2] = h[..., None]*(normal[..., None]*b+v[..., None]*modes)
    return result


def speed(state, axis, gravity=GRAVITY_M_S2):
    h = state[..., 0]
    normal = state[..., 1+axis]/h
    modes = state[..., 3+axis::2]/h[..., None]
    energy = np.sum(modes*modes/(2*np.arange(modes.shape[-1])+3), axis=-1)
    return abs(normal)+np.sqrt(gravity*h+3*energy)


def nonconservative_action(velocity, jump, axis):
    """C_axis(U) dU; velocity contains u,v,alpha_1,beta_1,... ."""
    result = np.zeros_like(jump)
    u, v = velocity[..., 0], velocity[..., 1]
    a, b = velocity[..., 2::2], velocity[..., 3::2]
    da, db = jump[..., 3::2], jump[..., 4::2]
    denominator = 2*np.arange(a.shape[-1])+3
    if axis == 0:
        result[..., 2] = np.sum((-b*da+a*db)/denominator, axis=-1)
        normal_jump = da
    else:
        result[..., 1] = np.sum((b*da-a*db)/denominator, axis=-1)
        normal_jump = db
    result[..., 3::2] = -u[..., None]*normal_jump
    result[..., 4::2] = -v[..., None]*normal_jump
    return result


def path_integral(left, right, axis):
    jump = right-left
    if jump.shape[-1] == 3:
        return np.zeros_like(jump)
    # C is linear in primitive velocities, so average them before its action.
    average = np.zeros_like(left[..., 1:])
    for node, weight in zip(PATH_NODES, PATH_WEIGHTS):
        state = left+node*jump
        average += weight*state[..., 1:]/state[..., :1]
    return nonconservative_action(average, jump, axis)


def minmod(a, b, c):
    same_sign = (np.sign(a) == np.sign(b)) & (np.sign(a) == np.sign(c))
    return np.where(same_sign, np.sign(a)*np.minimum(abs(a), np.minimum(abs(b), abs(c))), 0)


def reconstruct(state, axis, spatial_order=2, boundary="wall"):
    """Return face left/right states and each cell's two reconstructed ends."""
    if boundary == "periodic":
        padded = np.concatenate((np.take(state, [-1], axis=axis), state,
                                 np.take(state, [0], axis=axis)), axis=axis)
    elif boundary == "wall":
        left, right = np.take(state, [0], axis=axis).copy(), np.take(state, [-1], axis=axis).copy()
        left[..., 1+axis::2] *= -1
        right[..., 1+axis::2] *= -1
        padded = np.concatenate((left, state, right), axis=axis)
    else:
        raise ValueError("boundary must be wall or periodic")
    # Put the normal grid direction first for the same slices in x and y.
    padded = np.swapaxes(padded, 0, axis)
    slopes = np.zeros_like(padded)
    if spatial_order == 2:
        back = padded[1:-1]-padded[:-2]
        forward = padded[2:]-padded[1:-1]
        slopes[1:-1] = minmod(1.5*back, (back+forward)/2, 1.5*forward)
        if boundary == "periodic":
            slopes[0], slopes[-1] = slopes[-2], slopes[1]
        else:
            slopes[1] = slopes[-2] = 0
    elif spatial_order != 1:
        raise ValueError("spatial_order must be 1 or 2")
    minus, plus = padded-slopes/2, padded+slopes/2
    if min(np.min(minus[..., 0]), np.min(plus[..., 0])) <= 0:
        raise FloatingPointError("nonpositive reconstructed depth")
    return tuple(np.swapaxes(q, 0, axis) for q in
                 (plus[:-1], minus[1:], minus[1:-1], plus[1:-1]))


def directional_operator(state, spacing, axis, spatial_order=2, boundary="wall", gravity=GRAVITY_M_S2):
    left, right, cell_minus, cell_plus = reconstruct(state, axis, spatial_order, boundary)
    speed_bound = np.maximum(speed(left, axis, gravity), speed(right, axis, gravity))
    numerical_flux = (flux(left, axis, gravity)+flux(right, axis, gravity)
                      - speed_bound[..., None]*(right-left))/2
    path = path_integral(left, right, axis)
    flux_normal = np.swapaxes(numerical_flux, 0, axis)
    path_normal = np.swapaxes(path, 0, axis)
    derivative = (flux_normal[1:]-flux_normal[:-1]+(path_normal[1:]+path_normal[:-1])/2)
    derivative = np.swapaxes(derivative, 0, axis)+path_integral(cell_minus, cell_plus, axis)
    return -derivative/spacing


def spatial_operator(state, dx, dy, spatial_order=2, boundary="wall", gravity=GRAVITY_M_S2):
    return (directional_operator(state, dx, 0, spatial_order, boundary, gravity)
            + directional_operator(state, dy, 1, spatial_order, boundary, gravity))


def stable_timestep(state, dx, dy, cfl, spatial_order=2, boundary="wall", gravity=GRAVITY_M_S2):
    rates = []
    for axis, spacing in enumerate((dx, dy)):
        left, right, _, _ = reconstruct(state, axis, spatial_order, boundary)
        rates.append(max(float(speed(left, axis, gravity).max()),
                         float(speed(right, axis, gravity).max()))/spacing)
    return cfl/sum(rates)


def step(state, dt, dx, dy, spatial_order=2, boundary="wall", gravity=GRAVITY_M_S2):
    stage = validate(state+dt*spatial_operator(state, dx, dy, spatial_order, boundary, gravity))
    return validate((state+stage+dt*spatial_operator(stage, dx, dy, spatial_order, boundary, gravity))/2)
