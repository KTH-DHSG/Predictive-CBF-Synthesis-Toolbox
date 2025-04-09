"""

    CBF computation module for parallelized computation of Control Barrier Function (CBF) values.

    This module provides functions to compute CBF values in parallel using Dask, a parallel computing library for Python. 
    It allows for the computation of CBF values over a domain of points, distributing the workload across multiple processes.
    It also includes functions to initialize the CBF computation, compute CBF values at specific points, and handle optimization problems using CasADi.

    (c) Adrian Wiltz, 2025

"""

import sys                                              # import sys module                                   
import os                                               # import os module

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np                                      # import NumPy
import Auxiliaries.auxiliary as aux                               # import Auxiliaries module
from Auxiliaries.colored_warnings import colorWarning   # colored warnings
import dask                                             # parallelization
from dask import delayed                                # delayed computation
from dask.distributed import Client, as_completed       # dask client
from Dynamics.GenericDynamicSystem import GenericDynamicSystem # import the generic dynamic system  
from tqdm import tqdm                                   # progress bar
import casadi as ca                                     # import CasADi
import time                                             # measure time
import copy                                             # copy objects  
import webbrowser                                       # open web browser

def computeCbfParallelized(cbfModule, num_of_batches_factor=20, processes=None, timeout_per_sample=30):
    """
    Compute Control Barrier Function (CBF) values in parallel.
    This function parallelizes the computation of CBF values over a domain of points using Dask.
    It splits the domain into batches and distributes the computation across multiple processes.

    Remarks:
    --------
    - If it is displayed that the processes are killed prematurely, please double check that the attributes to the CBF module have been correctly initialized. In particular ensure that the functions h and cf as well as the dynamics have been correctly defined.
    - If the latter has been ensured, try to increase the timeout_per_sample parameter.

    Parameters:
    ----------
    cbfModule (module): The module containing the CBF and its specifications.
    num_of_batches_factor (int, optional): Factor to determine the number of batches. Default is 20. The number of batches is determined as processes * num_of_batches_factor.
    processes (int, optional): Number of processes to use. If None, defaults to the number of CPU cores.
    timeout_per_sample (int, optional): Timeout per sample in seconds. Default is 30. If the computation of a sample takes on average longer than this value, the process is killed.
    
    Returns:
    -------
    None
    """

    # Get optimization specifications
    opt_specs = copy.deepcopy(cbfModule.__getOptSpecs__())

    if opt_specs['p_opts'] is None:
        opt_specs['p_opts'] = {'print_time': False,
            'verbose': False}
    if opt_specs['s_opts'] is None:
        opt_specs['s_opts'] = {'max_iter': 1000,
            'print_level': 0}

    # Get the domain of the CBF as a list of points
    point_list = copy.deepcopy(cbfModule.cbf.getPointList())
    cbf_values = [-np.inf] * len(point_list)
    for i, point in enumerate(point_list):
        point["cbf_value"] = cbf_values[i]

    # Determine the number of batches and processes
    if processes is None:
        processes = os.cpu_count()

    num_of_batches = processes * num_of_batches_factor
    
    # SPlit the list of points into batches
    batch_size = len(point_list)//num_of_batches
    batches = [point_list[i:i + batch_size] for i in range(0, len(point_list), batch_size)]

    # Create a list of optimization specifications for each batch
    opt_specs_list = [opt_specs for _ in range(len(batches))]

    # Initialize the dask client
    timeout = timeout_per_sample * batch_size
    client = Client(processes=True, n_workers=processes, threads_per_worker=1, memory_limit='2GB', death_timeout=180)
    webbrowser.open(client.dashboard_link)
    # futures = client.map(computeCbfForBatch, [opt_specs_list[0]], [batches[0]])
    # client.gather(futures)
    futures = client.map(computeCbfForBatch, opt_specs_list, batches, retries=3)

    for future in tqdm(as_completed(futures, timeout=timeout), total=len(futures), desc="Computing CBF [batches computed/total batches]"):
        result = future.result()
        for batch_element in result:
            index = batch_element["index"]
            cbf_value = batch_element["cbf_value"]
            cbfModule.cbf.cbf_values[index] = cbf_value
    
    print("CBF computation domain completed.")

def computeCbfForBatch(opt_specs, batch):
    """
    Computes Control Barrier Function (CBF) values for a batch of points.

    Parameters:
    opt_specs (dict): A dictionary containing optimization specifications, including:
        - "h" (function): The h function as JSON string (to avoid serialization issues; function is recreated internally).
        - "cf" (function): The terminal constraint function as JSON string (to avoid serialization issues; function is recreated internally).
        - "N" (int): The number of discretization steps.
        - "dt" (float): The time step size.
        - "gamma" (float): The gamma parameter.
        - "h_offset" (float): The offset for the h function.
        - "p_opts" (dict): Solver options for the CasADi solver.
        - "s_opts" (dict): Solver options for the CasADi solver.
        - "eps" (float): A small epsilon value to avoid division by zero.
        - "p_norm" (float): The p-norm value/start value for p_norm.
        - "p_norm_decrement" (float): The p-norm decrement value. p_norm is decremented with this value if no solution is found.
        - "p_norm_min" (float): The minimum p-norm value. If p_norm is smaller than this value, the optimization is not re-evaluated.
        - "warmStartInputTrajectories" (np.ndarray): Initial input trajectories for warm start.
        - "dynamics" (str): JSON string representing the dynamics of the system (to avoid serialization issues; object is recreated internally).
    batch (list): A list of dictionaries, where each dictionary represents a point and contains:
        - "point" (np.ndarray): The point at which to compute the CBF value.
        - "index" (int): The index of the point in the point list.
        - "cbf_value" (float): The computed CBF value for the point.
        
    Returns:
    list: The input batch list with an additional key "cbf_value" in each dictionary, representing the computed CBF value for the corresponding point. If an error occurs during computation, "cbf_value" is set to NaN.
    """

    batch = copy.deepcopy(batch)

    # recreate dynamics object and functions
    h = aux.JSONStringToFunc(opt_specs["h"])
    cf = aux.JSONStringToFunc(opt_specs["cf"])
    generic_dynamics = GenericDynamicSystem()
    generic_dynamics.loadAttributesFromJSON(opt_specs["dynamics"])

    opt_specs_reconstructed = copy.deepcopy(opt_specs)
    opt_specs_reconstructed["h"] = h
    opt_specs_reconstructed["cf"] = cf
    opt_specs_reconstructed["dynamics"] = generic_dynamics
    opt_specs_reconstructed["warmStartInputTrajectories"] = np.array(opt_specs["warmStartInputTrajectories"])

    warmStartInputTrajectories = opt_specs_reconstructed["warmStartInputTrajectories"]

    opti_object = initializeCbfComputation(opt_specs_reconstructed)

    u_opt = None
    for batch_element in batch:
        if u_opt is not None:
            warmStartInputTrajectories_tmp = np.append(warmStartInputTrajectories,np.array([u_opt]),axis=0)
        else:
            warmStartInputTrajectories_tmp = warmStartInputTrajectories
        
        current_point = batch_element["point"]

        try:
            cbf_value, u_opt = computeCbfAtPoint(opti_object, current_point, warmStartInputTrajectories_tmp)
            batch_element["cbf_value"] = cbf_value
        except Exception as e:
            colorWarning(f"An error occurred while computing the CBF value at {current_point}: {e}")
            batch_element["cbf_value"] = np.nan
            u_opt = None

    return batch

def initializeCbfComputation(opt_specs_with_dynamics):
    """Implementation of the optimization problem
    
                                ||[ 1 / (h(x_N[0]) - 0 * gamma * Δt + h_tilde + epsilon) ]||
                                ||[ ...                                                  ]||             
        u*_N = argmin_{u_N}  ||[ 1 / (h(x_N[k]) - k * gamma * Δt + h_tilde + epsilon) ]||
                                ||[ ...                                                  ]||
                                ||[ 1 / (h(x_N[N]) - N * gamma * Δt + h_tilde + epsilon) ]||

        s.t.    x_N[0] = x0_param
                x_N[k+1] = f(x_N[k],u_N[k]) for k = 0,...,N
                u_min <= u_{N-1}[k] <= u_max for k = 0,...,N
                cf(x_N[N]) >= 0

    Parameters
    ----------
    opt_specs_with_dynamics : dict
        Dictionary containing the specifications for the optimization problem.
        - h: function
            The function h(x) used in the optimization problem.
        - cf: function
            The terminal constraint function cf(x).
        - dynamics: object
            The dynamics of the system (as DynamicSystem object or an object of one of its subclasses).
        - N: int
            The number of discretization steps.
        - dt: float
            The time step size.
        - gamma: float
            The gamma parameter.
        - h_offset: float
            The offset for the h function.
        - p_opts: dict, optional
            Solver options for the CasADi solver. Default is used when None is provided as a value. The default is {'print_time': False, 'verbose': False}.
        - s_opts: dict, optional
            Solver options for the CasADi solver. Default is used when None is provided as a value. The default is {'max_iter': 1000, 'print_level': 0}.
        - eps: float
            A small epsilon value to avoid division by zero.
        - warmStartInputTrajectories: list
            List of warm start input trajectories for the optimization.

    Returns
    -------
    opti_object : dict
        Dictionary containing the optimization problem and its parameters.
        - h: function
            The function h(x) used in the optimization problem.
        - x0_param: CasADi parameter
            The initial state parameter for the optimization problem.
        - p_norm_param: CasADi parameter
            The p-norm parameter for the objective function.
        - cbfOpti: CasADi Opti object
            The optimization problem object.
        - X: CasADi variable
            The state trajectory variable.
        - U: CasADi variable
            The control input trajectory variable.
        - p_norm: float
            The p-norm value/start value for p_norm.
        - p_norm_decrement: float
            The p-norm decrement value. p_norm is decremented with this value if no solution is found.
        - p_norm_min: float
            The minimum p-norm value. If p_norm is smaller than this value, the optimization is not re-evaluated.
    """
    # Get optimization specifications
    h = opt_specs_with_dynamics["h"]                    # h function
    cf = opt_specs_with_dynamics["cf"]                  # terminal constraint function
    dynamics = opt_specs_with_dynamics["dynamics"]      # dynamics of the system
    N = opt_specs_with_dynamics["N"]                    # number of discretization steps
    dt = opt_specs_with_dynamics["dt"]                  # time step size
    gamma = opt_specs_with_dynamics["gamma"]            # gamma parameter
    h_offset = opt_specs_with_dynamics["h_offset"]      # offset for the h function
    p_opts = opt_specs_with_dynamics["p_opts"]          # solver options for the CasADi solver
    s_opts = opt_specs_with_dynamics["s_opts"]          # solver options for the CasADi solver
    eps = opt_specs_with_dynamics["eps"]                # small epsilon value to avoid division by zero
    p_norm = opt_specs_with_dynamics["p_norm"]          # p-norm value/start value for p_norm
    p_norm_decrement = opt_specs_with_dynamics["p_norm_decrement"] # p-norm decrement value
    p_norm_min = opt_specs_with_dynamics["p_norm_min"]  # minimum p-norm value

    opti_object = { 'h': h,
                    'p_norm': p_norm,
                    'p_norm_decrement': p_norm_decrement,
                    'p_norm_min': p_norm_min,
                    'dynamics': dynamics,
                    'dt': dt,
                    'N': N}
    
    if p_opts is None:
        p_opts = {'print_time': False,
            'verbose': False}
    if s_opts is None:
        s_opts = {'max_iter': 1000,
            'print_level': 0}

    # determine dimensions of state and input of the dynamic system
    x_dim = dynamics.x_dim
    u_dim = dynamics.u_dim

    # Create a CasADi function for the dynamics
    state = ca.MX.sym('state', x_dim)   
    control = ca.MX.sym('control', u_dim)
    dynamics_ca = ca.Function('dynamics', [state, control], [dynamics.f(state, control)])

    # Use CasADi's built-in integrator for discretization
    ode = {'x': state,
            'p': control,
            'ode': dynamics_ca(state, control)}       # Dynamics: dx/dt = f(x, u)
    opts = {'simplify': True,               # Simplify the integrator model
            'number_of_finite_elements': 4} # Number of finite elements for the integrator
    integrator = ca.integrator('integrator', 'rk', ode, 0, dt, opts)

    x_next = integrator(x0=state, p=control)['xf']
    dynamics_discretized = ca.Function('dynamics_discretized', [state, control], [x_next], ['x', 'u'], ['x_next'])

    # Define the optimization problem
    cbfOpti = ca.Opti()    
        
    # Decision variables: states and controls
    X = cbfOpti.variable(x_dim, N+1)      # State trajectory: [x0, x1, ..., xN]
    U = cbfOpti.variable(u_dim, N)        # Control input trajectory: [u0, u1, ..., uN-1]
    opti_object['X'] = X
    opti_object['U'] = U

    # Initial state as a parameter to the optimization problem
    x0_param = cbfOpti.parameter(x_dim)
    opti_object['x0_param'] = x0_param
    cbfOpti.subject_to(X[:, 0] == x0_param)    # Initial state

    # Dynamics constraints
    for k in range(N):
        # Integrate dynamics
        x_next = dynamics_discretized(X[:, k], U[:, k])
        cbfOpti.subject_to(X[:, k+1] == x_next)

        # Input constraints
        cbfOpti.subject_to(cbfOpti.bounded(dynamics.u_min, U[:, k], dynamics.u_max))  # Control input: u_min <= u <= u_max

    # Terminal constraint
    cbfOpti.subject_to(cf(X[:, -1]) >= 0)  # Terminal constraint: cf(xN) >= 0

    # Objective function
    h_values = [h(X[:, k]) for k in range(N+1)]       # Compute h(x_N[k]) for k = 0,...,N

    p_norm_param = cbfOpti.parameter()
    opti_object['p_norm_param'] = p_norm_param
    
    h_values_inv = 1 / (ca.vertcat(*h_values) - np.arange(N+1) * gamma * dt + h_offset + eps) # Vectorized computation of 1 / (h(x_N[k]) - k * gamma * Δt + h_tilde + epsilon) for k = 0,...,N
    cost = ca.power(ca.sum1(ca.power(ca.fabs(h_values_inv), p_norm_param)), 1/p_norm_param)            # Compute the p-norm of the vector h_values_inv 
    cbfOpti.minimize(cost)                                      # Objective: minimize the p-norm of the vector h_values_inv

    # Solver options
    cbfOpti.solver('ipopt', p_opts, s_opts)

    # Store the optimization problem as an arrtibute of the CBF module
    opti_object['cbfOpti'] = cbfOpti

    return opti_object

def computeCbfAtPoint(opti_object, point, warmStartInputTrajectories):
    """
    Computes the Control Barrier Function (CBF) value at a given point.

    Parameters:
    ----------
    opti_object (dict): A dictionary containing the optimization problem and related parameters.
        - h (function): The h function. 
        - cbfOpti (object): The optimization problem.
        - p_norm (float): The p-norm value.
        - p_norm_decrement (float): The p-norm decrement value.
        - p_norm_min (float): The minimum p-norm value.
        - x0_param (object): The initial state parameter.
        - p_norm_param (object): The p-norm parameter.
        - X (object): The state trajectory variable.
        - U (object): The control input trajectory variable.
        - dynamics (object): The dynamics of the system.
        - dt (float): The time step size.
        - N (int): The number of discretization steps.
    point (array-like): The point at which the CBF value is to be computed.
    warmStartInputTrajectories (array-like): The warm start input trajectories.

    Returns:
    -------
    tuple: A tuple containing:
        - cbfValue (float): The computed CBF value at the given point.
        - u_opt (array-like): The optimal control input trajectory.
    """

    
    # Read out opti_object
    h = opti_object['h']                            # h function
    cbfOpti = opti_object['cbfOpti']                # Optimization problem
    p_norm = opti_object['p_norm']                  # p-norm value
    p_norm_decrement = opti_object['p_norm_decrement'] # p-norm decrement value
    p_norm_min = opti_object['p_norm_min']          # minimum p-norm value
    x0_param = opti_object['x0_param']              # initial state parameter
    p_norm_param = opti_object['p_norm_param']      # p-norm parameter
    X = opti_object['X']                            # state trajectory variable
    U = opti_object['U']                            # control input trajectory variable
    dynamics = opti_object['dynamics']              # dynamics of the system
    dt = opti_object['dt']                          # time step size
    N = opti_object['N']                            # number of discretization steps

    warmStartStateTrajectories = np.zeros((warmStartInputTrajectories.shape[0],dynamics.x_dim,N+1))
    for i in range(warmStartInputTrajectories.shape[0]):
        _, warmStartStateTrajectories[i] = dynamics.simulateOverHorizon(x0=point,u=warmStartInputTrajectories[i],dt=dt)
    
    # Set starting point for optimization
    cbfOpti.set_value(x0_param, point)

    # Initialize the CBF value array for each warm start state trajectory
    cbfValues = np.nan * np.ones(warmStartInputTrajectories.shape[0])
    
    # Initialize solution variables 
    cbfValue = -np.inf
    u_opt = None

    for i in range(warmStartInputTrajectories.shape[0]):
        
        solution_found = False
        exit = False
        p_norm_tmp = p_norm

        # Set the warm start trajectories
        cbfOpti.set_initial(U, warmStartInputTrajectories[i])
        cbfOpti.set_initial(X, warmStartStateTrajectories[i])

        cbfOptiSolution = None

        while not solution_found and not exit:
            # Solve optimization problem and handle possible problems. Extraction of the CBF value if optimization was successfull, otherwise decrease p_norm value and try again.

            # Set the p_norm parameter
            cbfOpti.set_value(p_norm_param, p_norm_tmp)

            try:
                # solve optimization problem
                cbfOptiSolution_tmp = cbfOpti.solve()

                # Extract solution if optimization was successful
                isOptimizationSuccessful = cbfOptiSolution_tmp.stats()['return_status']
                if isOptimizationSuccessful == 'Solve_Succeeded':
                    cbfOptiSolution = cbfOptiSolution_tmp
                    solution_found = True
                elif isOptimizationSuccessful == 'Solved_To_Acceptable_Level':
                    cbfOptiSolution = cbfOptiSolution_tmp
                    print(f"At {point} the optimization was solved to an acceptable level. The search continues to find a better solution. A decreased p-norm value is used.")
                else:
                    print(f"At {point} the optimization was unsuccessful. The search is with a decreased p-norm value.")

                # Decrease p_norm value if no solution was found
                if p_norm_tmp > p_norm_min:
                    p_norm_tmp -= p_norm_decrement
                else:
                    exit = True

            except Exception as e:
                # error handling
                if p_norm_tmp > p_norm_min:
                    # Decrease p_norm value if an error occurred during optimization
                    p_norm_tmp -= p_norm_decrement
                    colorWarning(f"An error occurred during optimization for point {point}: {e}. The optimization will be repeated with a decreased p-norm value.")
                elif ~np.isnan(cbfValues[i]):
                    # If an approximation of the CBF value has been found, and no better solution could be obtained with further iterations. In this case, the last found approximation is returned.
                    solution_found = True
                else:
                    colorWarning(f"An error occurred during optimization for point {point}: {e}")
                    exit = True

        # post processing: find the actual CBF value
        if cbfOptiSolution is not None:
            # 1. extract solution from cbfOptiSolution
            u_opt_tmp = cbfOptiSolution.value(U)
            # 2. simulate the system with the optimal control input
            _, x_opt_tmp = dynamics.simulateOverHorizon(x0=point,u=u_opt_tmp,dt=dt)
            # 3. compute the CBF value
            h_values_opt = [h(x_opt_tmp[:, k]) for k in range(N+1)]       # Compute h(x_opt[k]) for k = 0,...,N
            # 4. compute the cbf value candidate as the smallest value of h_values_opt
            cbfValues[i] = np.min(h_values_opt)
            # 5. store the optimal control input and state trajectory

            if cbfValues[i] > cbfValue:
                cbfValue = cbfValues[i]
                u_opt = u_opt_tmp
                x_opt = x_opt_tmp

        else:
            cbfValues[i] = np.nan

    # FInd the minimum value of the CBF value array
    if cbfValue > -np.inf:
        print(f"CBF value at {point} has been found: cbfValue = {cbfValue}. List of values from which the CBF value has been selected from: {cbfValues}")
    else:   
        cbfValue = np.nan
        print(f"CBF value at {point} could not be found. The CBF value is set to nan.")

    return cbfValue, u_opt

