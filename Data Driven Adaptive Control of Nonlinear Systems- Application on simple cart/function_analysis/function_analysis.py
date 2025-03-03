import numpy as np

from dynamics.discretization import Discretization
from dynamics.linearization import Linearization


class FunctionAnalysis:
    def __init__(self, params):
        self.params = params

    def compute_time_varying_jacobians(self, states, controls):
        """
        For a given nominal trajectory (states and controls), compute the discrete-time
        Jacobians A(k) and B(k) at a specified linearization interval T.

        If T == dt, linearize at every step. Otherwise, linearize every T seconds and
        reuse the last computed matrices for the intermediate steps.

        Inputs:
        - states: array of nominal states (each state is [position, velocity])
        - controls: array of nominal control inputs (each control is [force])
        - params: dictionary containing system parameters, including:
            'dt': sampling time,
            'T_lin': linearization interval (in seconds)

        Returns:
        - A_list: list of discrete-time state matrices A(k) for each step
        - B_list: list of discrete-time input matrices B(k) for each step
        """
        dt = self.params["dt"]
        T_lin = self.params["dt"]
        steps_per_lin = int(round(T_lin / dt))
        A_list = []
        B_list = []
        last_A = None
        last_B = None
        lin = Linearization(self.params)
        A_func, B_func = lin.analytical_jacobians()

        num_steps = len(states) - 1  # assuming states has length N+1 (for N control steps)
        for k in range(num_steps):
            # At every multiple of steps_per_lin, re-linearize
            if k % steps_per_lin == 0:
                x_nom = states[k]
                u_nom = controls[k] if k < len(controls) else controls[-1]
                # Evaluate the continuous-time Jacobians at (x_nom, u_nom)
                A_cont = A_func(x_nom[0], x_nom[1], u_nom[0])
                B_cont = B_func(x_nom[0], x_nom[1], u_nom[0])
                # Discretize the Jacobians (ZOH)
                disc = Discretization(self.params)
                A_d, B_d = disc.discretize(A_cont, B_cont)
                last_A = A_d
                last_B = B_d
            # Otherwise, reuse the last computed matrices
            A_list.append(last_A)
            B_list.append(last_B)
        return A_list, B_list

    def compute_lipschitz_constant(self, matrices, time_steps):
        # Ensure inputs are valid
        if len(matrices) != len(time_steps):
            raise ValueError("Length of matrices list and time_steps list must be equal.")
        if len(matrices) < 2:
            # With fewer than 2 points, return 0 (no change over time)
            return 0.0

        # Initialize L to 0
        L = 0.0
        # Compare each pair of matrices
        for i in range(len(matrices)):
            for j in range(i + 1, len(matrices)):
                # Compute Frobenius norm of difference
                diff_norm = np.linalg.norm(matrices[i] - matrices[j], ord="fro")
                # Compute time difference
                time_diff = abs(time_steps[i] - time_steps[j])
                if time_diff == 0:
                    continue  # skip pairs with zero time difference to avoid division by zero
                # Compute the ratio (difference norm per time)
                ratio = diff_norm / time_diff
                # Update L if this ratio is larger
                L = max(L, ratio)
        return L

    def compute_residuals(self, nominal_states, nominal_controls, actual_states, actual_controls):
        """
        Compute the residual error at each dt step over the trajectory.

        Given:
          - nominal_states: array of shape (N+1, state_dim)
          - nominal_controls: array of shape (N, control_dim)
          - actual_states: array of shape (N+1, state_dim)
          - actual_controls: array of shape (N, control_dim)
          - t_vals: time vector for the full dt grid (length N+1)

        The residual at each step k (for k=0,...,N-1) is computed as:
        r(k) = [x_actual(k+1) - x_nominal(k+1)] - [A(k) (x_actual(k)- x_nominal(k)) + B(k) (u_actual(k)- u_nominal(k))]

        Returns:
          - r_list: a list of residual errors (each as a column vector) at each dt step.
        """
        # Compute the time-varying Jacobians on the full dt grid from the nominal trajectory.
        A_list, B_list = self.compute_time_varying_jacobians(nominal_states, nominal_controls)

        r_list = []
        num_steps = len(nominal_states) - 1  # because we use k and k+1

        for k in range(num_steps):
            # Compute the deviation at time k and k+1
            delta_x_k = actual_states[k] - nominal_states[k]
            delta_x_next = actual_states[k + 1] - nominal_states[k + 1]
            # Compute control deviation at time k (ensuring it's a vector)
            delta_u_k = np.atleast_1d(actual_controls[k]) - np.atleast_1d(nominal_controls[k])

            # Use the linearization at time k to predict the evolution of the deviation over one dt step.
            predicted_delta = A_list[k] @ delta_x_k + B_list[k] @ delta_u_k

            # Residual error is the difference between the actual deviation at k+1 and the predicted deviation.
            r_k = delta_x_next - predicted_delta
            r_list.append(r_k.reshape(-1, 1))  # reshape as a column vector for consistency

        return r_list
