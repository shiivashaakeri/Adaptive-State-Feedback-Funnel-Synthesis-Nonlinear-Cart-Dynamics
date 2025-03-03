import numpy as np
from scipy.linalg import expm


class Discretization:
    def __init__(self, params):
        """
        Initialize the Discretization class with system parameters.

        Parameters:
        params (dict): Parameters of the system, including the number of states,
                       controls, and other discretization-related parameters.
        """
        self.params = params

    def discretize(self, A, B):
        """
        Discretize the continuous-time system using either ZOH or FOH.
        Returns:
            For ZOH: A_d, B_d
            For FOH: A_d, B_d0, B_d1
        """
        dt = self.params["dt"]
        n = A.shape[0]
        B = np.atleast_2d(B)
        m = B.shape[1]
        if self.params["dis_type"] == "FOH":
            # FOH discretization using an augmented matrix of size (n+2*m) x (n+2*m)
            # Construct the augmented matrix:
            #   [ A    B      0 ]
            #   [ 0    0     I_m ]
            #   [ 0    0      0  ]
            M_FOH = np.block([
                [A,                B,                  np.zeros((n, m))],
                [np.zeros((m, n)), np.zeros((m, m)),    np.eye(m)],
                [np.zeros((m, n+2*m))]
            ])
            Mexp = expm(M_FOH * dt)
            A_d = Mexp[:n, :n]
            B_d0 = Mexp[:n, n:n+m]
            B_d1 = Mexp[:n, n+m:n+2*m]
            return A_d, B_d0, B_d1
        else:
            # ZOH discretization
            M = np.block([
                [A, B],
                [np.zeros((m, n+m))]
            ])
            Mexp = expm(M * dt)
            A_d = Mexp[:n, :n]
            B_d = Mexp[:n, n:]
            return A_d, B_d
