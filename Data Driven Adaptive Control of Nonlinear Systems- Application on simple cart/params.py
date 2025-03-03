import numpy as np


def cart_params():

    # System and obstacle parameters
    n_obs = 0  # Number of obstacles
    n_states = 2 + n_obs  # Number of states including augmentation
    n_controls = 1  # Number of controls

    # Initial and final states
    initial_state = np.array([5.0, 0])
    final_state = np.array([0.0, 0])
    initial_control = np.array([0.0])

    # State bounds
    max_state = np.array([10, 5])
    min_state = np.array([-10, -5])

    # Control bounds
    max_control = np.array([45])
    min_control = np.array([-45])

    # Simulation parameters
    total_time = 10

    L_constant = 5.8

    # Return parameter dictionary
    return dict(  # noqa: C408
        g=9.81,  # Gravity
        m=1.0,  # Mass
        b=0.5,
        epsilon= 0.1,
        dt= 0.001,
        total_time=total_time,
        T_lin = 0.2,
        n_states=n_states,
        n_controls=n_controls,
        n_obs=n_obs,
        dis_type="ZOH",  # Discretization type: ZOH or FOH
        dis_exact=True,  # Whether to use exact discretization
        initial_state=initial_state,
        initial_control=initial_control,
        final_state=final_state,
        max_state=max_state,
        min_state=min_state,
        max_control=max_control,
        min_control=min_control,
        L_constant = L_constant,
    )


