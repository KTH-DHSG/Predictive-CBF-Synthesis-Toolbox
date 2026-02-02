"""

    Implementation of a safe controller that computes a safe control input based on the given controller settings and current state. The safe control input is computed by solving an optimization problem that minimizes the distance to the baseline control input while satisfying the control barrier function (CBF) and lambda conditions. If the point is within the domain, a safe control input is computed. Otherwise, the baseline control input is returned, as the system can be assumed to be sufficiently remote to the boundary of the safe set.

    (c) Adrian Wiltz, 2025

"""

import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator
from Dynamics.DynamicSystem import DynamicSystem
from Auxiliaries.auxiliary_math import approximate_directional_gradient, approximate_gradient, isPointWithinDomain
from Auxiliaries.colored_warnings import colorWarning
import casadi as ca

def compute_safe_input(controller_settings,t,x,u_baseline,P=None, method='trust-constr'):
    """
    Compute a safe control input based on the given controller settings and current state. The safe control input is computed by solving an optimization problem that minimizes the distance to the baseline control input while satisfying the control barrier function (CBF) and lambda conditions. If the point is within the domain, a safe control input is computed. Otherwise, the baseline control input is returned, as the system can be assumed to be sufficiently remote to the boundary of the safe set. 

    Parameters:
        controller_settings (dict): A dictionary containing the settings for the controller. Expected keys are:
            - 'cbf_grid_points': Grid points of the state space of the control barrier function (CBF). Used to check if the point x is within the domain. If the point is within the domain, a safe control input is computed. Otherwise, the baseline control input is returned.
            - One of the following:
                * 'cbf_function': Function to compute the CBF value with interface cbf_function(t,x) where t is a scalar representing time and x an NumPy arrary representing the system state.
                * 'cbf_interpolator': Interpolator function for the time-invariant CBF. The interpolator should be a callable function that takes a NumPy array as input (state) and returns a NumPy array with one element representing the CBF value. It provides the possibility to 
            - 'alpha': Function to compute alpha based on the CBF value.
            - 'dynamics': Instance of a DynamicSystem class of the dynamic system, expected to have a method f(x, u) that computes the system dynamics x_dot = f(x, u).
            - 'lambda_fun' (optional): Function to compute lambda at a given time. Default is a constant zero function.
            - 'dt' (optional): Time step for numerical differentiation. Default is 0.01.
            - 'step_size' (optional): Step size for numerical gradient approximation. Default is 0.1.
            - 'alpha_offset' (optional): Offset for the alpha value. Default is 0.01 A positive offset can be used to increase the robustness of the controller against disturbances.
        t (float): Current time.
        x (array-like): Current state of the system.
        u_baseline (array-like): Baseline control input.
        P (array-like, optional): Positive definite weight matrix in the objective. Default is the identity matrix.

    Returns:
        NumPy-array: Safe control input. If the point is within the domain, a safe control input is computed. Otherwise, the baseline control input is returned.
    """

    if P is None:
        P = np.eye(len(u_baseline))

    if 'cbf_grid_points' in controller_settings:
        cbf_grid_points = controller_settings['cbf_grid_points']
    else:
        cbf_grid_points = None

    # ensure that state x is a one-dimensional numpy array
    if isinstance(x, ca.DM):
        x = np.array(x.full()).flatten()

    # Check if the point is within the domain. If yes, generate a safe control input. Otherwise, use the baseline controller as the system can be assumed to be sufficiently remote to the boundary of the safe set.
    if (cbf_grid_points is None) or isPointWithinDomain(cbf_grid_points,x):

        # Extract the controller settings
        alpha = controller_settings['alpha']
        dynamics = controller_settings['dynamics']
        bounds = [(dynamics.u_min[i],dynamics.u_max[i]) for i in range(dynamics.u_dim)]
        if 'lambda_fun' in controller_settings:
            lambda_fun = controller_settings['lambda_fun']
        else:
            lambda_fun = lambda t: 0
        if 'dt' in controller_settings:
            dt = controller_settings['dt']
        else:
            dt = 0.01
        if 'step_size' in controller_settings:
            step_size = controller_settings['step_size']
        else:
            step_size = 0.1
        if 'alpha_offset' in controller_settings:
            alpha_offset = controller_settings['alpha_offset']
        else:
            alpha_offset = 0

        if 'cbf_interpolator' in controller_settings:
            cbf_interpolator = controller_settings['cbf_interpolator']

            # Interpolate the CBF and lambda functions
            lambda_val = lambda_fun(t)
            cbf_val = cbf_interpolator(x)
            b_val = cbf_val + lambda_val

            # compute the CBF gradient
            cbf_gradient = approximate_gradient(cbf_interpolator,x,h=step_size)
            
            # Compute the time-gradient of the CBF and lambda functions
            lambda_val_dt = (lambda_fun(t+dt)-lambda_val)/dt
            b_dt_val = lambda_val_dt

        elif 'cbf_function' in controller_settings:
            cbf_function = controller_settings['cbf_function']
            
            # compute the CBF value
            b_val = cbf_function(t,x)
            cbf_gradient = approximate_gradient(lambda x: cbf_function(t,x),x,h=step_size)
            b_dt_val = (cbf_function(t+dt,x) - cbf_function(t,x))/dt

        else:
            raise ValueError("No CBF as function or interpolator provided.")

        alpha_val = alpha(b_val)

        # Compute the safe input
        obj = lambda u: np.dot(P@(u-u_baseline),u-u_baseline)
        const = {'type':'ineq', 'fun': lambda u_tmp: np.dot(cbf_gradient, dynamics.f(x,u_tmp)) + b_dt_val + alpha_val - alpha_offset}


        try:
            # Safe input computation with fallback to baseline controller
            # result = minimize(obj,u_baseline,constraints=const,bounds=bounds,method=method)
            result = minimize(obj,0*u_baseline,constraints=const,bounds=bounds)
            u_safe = result.x
        except:
            # Safe input computation failed, use baseline controller instead
            u_safe = u_baseline
            colorWarning(f"Safe controller optimization failed at t={t} and x={x}. Using baseline controller instead.")

    else:
        u_safe = u_baseline

    return np.array(u_safe)

def compute_safe_input_multiconstrained(controller_settings,t,x,u_baseline,P=None, method='trust-constr'):
    """
    Compute a safe control input based on the given controller settings and current state. The safe control input is computed by solving an optimization problem that minimizes the distance to the baseline control input while satisfying the control barrier function (CBF) and lambda conditions. If the point is within the domain, a safe control input is computed. Otherwise, the baseline control input is returned, as the system can be assumed to be sufficiently remote to the boundary of the safe set. 

    Parameters:
        controller_settings (dict): A dictionary containing the settings for the controller. Expected keys are:
            - 'cbf_grid_points': Grid points of the state space of the control barrier function (CBF). Used to check if the point x is within the domain. If the point is within the domain, a safe control input is computed. Otherwise, the baseline control input is returned.
            - Defining the constraints imposed on the system
                * 'cbf_function_array': Array of functions to compute the CBF value with interface cbf_function(t,x) where t is a scalar representing time and x an NumPy arrary representing the system state.
            - 'alpha_array': Array of functions to compute alpha based on the CBF value.
            - 'dynamics': Instance of a DynamicSystem class of the dynamic system, expected to have a method f(x, u) that computes the system dynamics x_dot = f(x, u).
            - 'lambda_fun_array' (optional): Array of functions to compute lambda at a given time. Default is a constant zero function.
            - 'dt' (optional): Time step for numerical differentiation. Default is 0.01.
            - 'step_size_array' (optional): Array of step sizese for numerical gradient approximation. Default is 0.1.
            - 'alpha_offset_array' (optional): Array of offsets for the alpha value. Default is 0.01 A positive offset can be used to increase the robustness of the controller against disturbances.
        t (float): Current time.
        x (array-like): Current state of the system.
        u_baseline (array-like): Baseline control input.
        P (array-like, optional): Positive definite weight matrix in the objective. Default is the identity matrix.

    Returns:
        NumPy-array: Safe control input. If the point is within the domain, a safe control input is computed. Otherwise, the baseline control input is returned.
    """

    if P is None:
        P = np.eye(len(u_baseline))

    if 'cbf_grid_points' in controller_settings:
        cbf_grid_points = controller_settings['cbf_grid_points']
    else:
        cbf_grid_points = None

    # ensure that state x is a one-dimensional numpy array
    if isinstance(x, ca.DM):
        x = np.array(x.full()).flatten()

    # Check if the point is within the domain. If yes, generate a safe control input. Otherwise, use the baseline controller as the system can be assumed to be sufficiently remote to the boundary of the safe set.
    if (cbf_grid_points is None) or isPointWithinDomain(cbf_grid_points,x):

        # Extract the controller settings
        alpha_array = controller_settings['alpha_array']
        dynamics = controller_settings['dynamics']
        bounds = [(dynamics.u_min[i],dynamics.u_max[i]) for i in range(dynamics.u_dim)]

        num_constraints = controller_settings['cbf_function_array'].__len__()
        
        if 'lambda_fun_array' in controller_settings:
            lambda_fun_array = controller_settings['lambda_fun_array']
        else:
            lambda_fun_array = [lambda t: 0 for _ in range(num_constraints)]
        if 'dt' in controller_settings:
            dt = controller_settings['dt']
        else:
            dt = 0.01
        if 'step_size_array' in controller_settings:
            step_size_array = controller_settings['step_size_array']
        else:
            step_size_array = [0.1] * num_constraints
        if 'alpha_offset_array' in controller_settings:
            alpha_offset_array = controller_settings['alpha_offset_array']
        else:
            alpha_offset_array = [0] * num_constraints

        if 'cbf_function_array' in controller_settings:
            cbf_function_array = controller_settings['cbf_function_array']

            # compute the CBF value
            b_val_array = np.zeros(num_constraints)
            cbf_gradient_array = np.zeros((num_constraints, len(x)))
            b_dt_val_array = np.zeros(num_constraints)

            for i in range(num_constraints):
                lambda_val = lambda_fun_array[i](t)
                cbf_val = cbf_function_array[i](t,x)
                b_val_array[i] = cbf_val + lambda_val

                # compute the CBF gradient
                cbf_gradient_array[i,:] = approximate_gradient(lambda x: cbf_function_array[i](t,x),x,h=step_size_array[i])
                
                # Compute the time-gradient of the CBF and lambda functions
                lambda_val_dt = (lambda_fun_array[i](t+dt)-lambda_val)/dt
                b_dt_val_array[i] = lambda_val_dt

        else:
            raise ValueError("No CBF as function or interpolator provided.")

        alpha_val_array = np.array([alpha_array[i](b_val_array[i]) for i in range(num_constraints)])

        # Compute the safe input
        obj = lambda u: np.dot(P@(u-u_baseline),u-u_baseline)

        def cbf_constraint(u_tmp):
            # Compute system dynamics f(x, u)
            f_val = dynamics.f(x, u_tmp)  # shape (n_states,)
            
            # Compute the Lie derivative term ∇h * f
            lie_derivative = cbf_gradient_array @ f_val  # shape (num_constraints,)
            
            # Compute the final CBF constraint values
            constraint_vals = lie_derivative + b_dt_val_array + alpha_val_array - alpha_offset_array
            
            # Return as 1D array
            return np.array(constraint_vals).flatten()

        const = {'type': 'ineq', 'fun': cbf_constraint}

        try:
            # Safe input computation with fallback to baseline controller
            # result = minimize(obj,u_baseline,constraints=const,bounds=bounds,method=method)
            result = minimize(obj,0*u_baseline,constraints=const,bounds=bounds)
            u_safe = result.x
        except:
            # Safe input computation failed, use baseline controller instead
            u_safe = u_baseline
            colorWarning(f"Safe controller optimization failed at t={t} and x={x}. Using baseline controller instead.")

    else:
        u_safe = u_baseline

    return np.array(u_safe)

