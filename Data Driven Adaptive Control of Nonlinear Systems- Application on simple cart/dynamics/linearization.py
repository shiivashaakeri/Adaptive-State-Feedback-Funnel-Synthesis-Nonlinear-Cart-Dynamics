from sympy import Matrix, lambdify, symbols


class Linearization:
    def __init__(self, params):
        self.params = params

    def analytical_jacobians(self):
        """
        Computes the analytical Jacobians of the inverted pendulum dynamics.

        The dynamics are defined by:
            theta_dot = omega
            omega_dot = (g/L)* sin(theta) + (1/(m*L**2))* u

        Returns:
        A, B: Functions that compute the Jacobians of the system dynamics with respect
              to the state [theta, omega] and the control input [u].
        """
        # Define symbols for the state and control input
        x, x_dot, F = symbols('x x_dot F')
        m = self.params["m"]
        b = self.params["b"]
        epsilon = self.params.get("epsilon", 0)

        # Define the cart dynamics:
        f1 = x_dot
        f2 = (F - b * x_dot) / (m * (1 + epsilon * x_dot**2))

        # Construct state_dot vector
        state_dot = Matrix([f1, f2])
        # Construct the state vector and control vector
        state_vec = Matrix([x, x_dot])
        control_vec = Matrix([F])

        # Compute the Jacobians with respect to the state and control input
        A_expr = state_dot.jacobian(state_vec)
        B_expr = state_dot.jacobian(control_vec)

        # Convert the Jacobians to functions that can be evaluated numerically
        A = lambdify((x, x_dot, F), A_expr, "numpy")
        B = lambdify((x, x_dot, F), B_expr, "numpy")

        return A, B
