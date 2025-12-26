import numpy as np
import cvxpy as cp


def cbf_row_for_obstacles(p, v_nominal, obs, params):

    c = np.asarray(obs["c"], dtype=float)
    c_dot = np.asarray(obs.get("c_dot", np.zeros(3)), dtype=float)  # <-- fixed
    r_obs = float(obs["r_obs"])

    drone_radius   = float(params.get("drone_radius", 0.2))
    safety_margin  = float(params.get("safety_margin", 0.0))
    gamma          = float(params.get("gamma", 1.0))

    # Effective safety radius
    r_safe = r_obs + drone_radius + safety_margin

    # Relative position and relative velocity
    p_rel    = c - p
    v_rel_nom = c_dot - v_nominal   # (obstacle vel - drone vel)

    d     = np.linalg.norm(p_rel)
    v_mag = np.linalg.norm(v_rel_nom)

    # If obstacle is very far or relative motion tiny, skip
    if d < 1e-4 or v_mag < 1e-4:
        return None, None

    sin_phi = min(r_safe / max(d, 1e-4), 0.999)
    cos_phi = np.sqrt(1.0 - sin_phi**2)

    # Defining CBF: h(p) = <p_rel, v_rel_nom> + ||p_rel|| ||v_rel_nom|| cos(phi)
    h = float(p_rel @ v_rel_nom + d * v_mag * cos_phi)

    # Gradient wrt p (note p_rel = c - p ⇒ ∂p_rel/∂p = -I)
    grad_h_p = -(v_rel_nom + v_mag * cos_phi * (p_rel / d))

    # CBF condition: grad_h_p^T u + gamma*h >= 0
    # Convert to A u <= b form:  -(grad_h_p)^T u <= gamma*h
    A_i = -grad_h_p.reshape(1, 3)
    b_i = gamma * h

    return A_i, b_i


def cbf_safe_velocity(p, v_nominal, obstacles, params):

    p         = np.asarray(p, dtype=float).reshape(3,)
    v_nominal = np.asarray(v_nominal, dtype=float).reshape(3,)

    v_max = float(params.get("v_max", 2.0))

    # If no obstacle then just saturate nominal velocity and return
    if obstacles is None or len(obstacles) == 0:
        speed = np.linalg.norm(v_nominal)
        if speed > v_max:
            return v_nominal * (v_max / speed)
        else:
            return v_nominal

    A_list = []
    b_list = []

    for obs in obstacles:
        A_i, b_i = cbf_row_for_obstacles(p, v_nominal, obs, params)
        if A_i is not None:
            A_list.append(A_i)
            b_list.append(b_i)

    # If no active constraints, just saturate nominal
    if len(A_list) == 0:
        speed = np.linalg.norm(v_nominal)
        if speed > v_max:
            return v_nominal * (v_max / speed)
        else:
            return v_nominal

    A = np.vstack(A_list)
    b = np.array(b_list).reshape(-1)

    # QP variable
    u = cp.Variable(3)

    # Objective: stay close to nominal velocity
    obj = cp.Minimize(cp.sum_squares(u - v_nominal))

    # Constraints:
    constraints = []
    constraints.append(A @ u <= b)

    # Velocity bounds
    constraints += [
        u[0] <=  v_max,
        u[0] >= -v_max,
        u[1] <=  v_max,
        u[1] >= -v_max,
        u[2] <=  v_max,
        u[2] >= -v_max,
    ]

    problem = cp.Problem(obj, constraints)

    try:
        problem.solve(solver=cp.OSQP, warm_start=True)
    except Exception:
        # If solver fails, fall back to nominal
        speed = np.linalg.norm(v_nominal)
        if speed > v_max:
            return v_nominal * (v_max / speed)
        else:
            return v_nominal

    if u.value is None:
        # Infeasible or no solution: fall back
        speed = np.linalg.norm(v_nominal)
        if speed > v_max:
            return v_nominal * (v_max / speed)
        else:
            return v_nominal

    v_safe = np.array(u.value).reshape(3,)

    return v_safe
