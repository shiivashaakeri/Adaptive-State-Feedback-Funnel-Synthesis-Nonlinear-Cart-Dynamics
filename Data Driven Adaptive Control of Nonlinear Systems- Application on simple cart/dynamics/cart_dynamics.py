import numpy as np
from scipy.integrate import solve_ivp


class CartDynamics:
    def __init__(self, params):
        self.params = params

    def nonlinear_dynamics(self, t, state, control_current, control_slope, t_start, prop):
        """
        Compute the nonlinear dynamics of the cart.
        State: [x, x_dot]
        Dynamics:
            m * (1 + epsilon*x_dot^2) * x_ddot = F - b*x_dot
        =>  x_ddot = (F - b*x_dot) / (m*(1 + epsilon*x_dot^2))
        """
        if control_slope is None:
            control_slope = np.zeros_like(control_current)
        # If FOH is desired and dis_exact flag is set, linearly interpolate control:
        if self.params.get("dis_type", None) == "FOH" and self.params.get("dis_exact", False) and not prop:
            control = control_current + control_slope * (t - t_start)
        else:
            control = control_current
        F = np.array(control).flatten()[0]
        x, x_dot = state[0], state[1]
        m = self.params["m"]
        b = self.params["b"]
        epsilon = self.params["epsilon"]
        x_ddot = (F - b * x_dot) / (m * (1 + epsilon * x_dot**2))
        return np.array([x_dot, x_ddot])

    def simulate_nonlinear_step(self, state, control_current, control_slope=None, t_start=0.0, prop=True):
        """
        Simulate the nonlinear dynamics over one time step using a high-accuracy integrator.
        Here we use the DOP853 method.
        """
        dt = self.params["dt"]
        sol = solve_ivp(
            lambda t, s: self.nonlinear_dynamics(t, s, control_current, control_slope, t_start, prop),
            [0, dt],
            state,
            method="DOP853",  # Higher-order accurate integrator
        )
        return sol.y[:, -1]

    def simulate_linear_step(self, state, control_current, A_d, B_d, control_slope=None, B_d_slope=None):
        """
        Propagate the state using the discrete-time linear model.
        For ZOH:
            x_next = A_d * state + B_d * control_current
        For FOH:
            x_next = A_d * state + B_d * control_current + B_d_slope * (dt * control_slope)
        """
        if B_d_slope is not None:
            if control_slope is None:
                raise ValueError("For FOH discretization, a control_slope must be provided.")
            dt = self.params["dt"]
            return (
                A_d @ state
                + B_d @ np.atleast_1d(control_current).flatten()
                + B_d_slope @ (dt * np.atleast_1d(control_slope).flatten())
            )
        else:
            return A_d @ state + B_d @ np.atleast_1d(control_current).flatten()
