"""

    The CBFmodule handles all specifications required for the computation of a CBF. Most importantly, it constains the CBF itself, the dynamic system, the state constraint, the terminal constraint, the prediction horizon and the CBF design parameter. Its functions are used to set the warm start input trajectories, compute the offset of the CBF, save and load the CBFmodule to/from a file and generate the optimization specifications. The optimization specifications are used to initialize the CBF computation. The CBFmodule is used in the CBF computation and in the SafeController class. The module also includes a function for serialization that allows for using the module in parallelized computations.

    (c) Adrian Wiltz, 2025

"""

import sys                                              # import sys module                                   
import os                                               # import os module

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np                                      # import NumPy
import casadi as ca                                     # import CasADi
from CBF.CBF import CBF                                 # import the CBF class
from Dynamics.DynamicSystem import DynamicSystem        # import the dynamic system class
from Dynamics.GenericDynamicSystem import GenericDynamicSystem # import the generic dynamic system class
from datetime import datetime                           # timestamp
import Auxiliaries.auxiliary as aux                     # auxiliary functions
from Auxiliaries.colored_warnings import colorWarning   # colored warnings
import copy                                             # import copy module
import json                                             # save/load data
import inspect                                          # get source code of functions
import textwrap                                         # wrap text
import types                                            # check type of functions

class CBFmodule:
    """Class that stores all information needed for the computation of a CBF. 
    This includes the CBF itself, the dynamic system, the state constraint, the terminal constraint, 
    the prediction horizon, and the CBF design parameter."""

    def __init__(self,h=None,dynamicSystem=None,cf=None,T=None,N=None,gamma=None,domain_lower_bound=None,domain_upper_bound=None,discretization=None,p_norm=50,p_norm_decrement=10,p_norm_min=20):
        """Summarizes all specifications for the computation of a CBF.

        Args:
            h (lambda function or function): function defining state constraint as its zero super level set
            dynamicSystem (DynamicSystem): dynamic system description
            cf (lambda function or function): terminal constraint condition, returning true or false
            T (float): prediction horizon
            N (int): number of steps into which the prediction horizon is divided
            gamma(float): positive CBF desgin parameter
            domain_lower_bound (NumPy array of length x_dim): lower bound of domain
            domain_upper_bound (NumPy array of length x_dim): upper bound of domain
            discretization (int array of length x_dim): number of grid points in respective dimension
        """
        if not all(value is None for value in [h, dynamicSystem, cf, T, N, gamma, domain_lower_bound, domain_upper_bound, discretization]):
            # initialization with variables
            self.h = h                              # state constraint function
            self.dynamics = copy.deepcopy(dynamicSystem)  # dynamic system description (object of DynamicSystem class) 
            self.terminal_condition = cf            # terminal constraint function
            self.T = T                              # prediction horizon
            self.N = N                              # number of steps into which the prediction horizon is divided
            self.dt = T/N                           # discretization time step
            self.gamma = gamma                      # CBF design parameter
            self.p_norm = p_norm                    # p-norm value used to approximate the maximum of the min-max-problem
            self.p_norm_decrement = p_norm_decrement  # decrement value for p_norm used if the optimization is not successful
            self.p_norm_min = p_norm_min            # minimum value for p-norm used in optimization; if this value is reached, p_norm will not be further decremented

            self.domain_lower_bound = domain_lower_bound    # lower bound of domain
            self.domain_upper_bound = domain_upper_bound    # upper bound of domain
            self.discretization = discretization            # number of grid points in respective dimension

            self.warmStartInputTrajectories = None       # warm start input trajectories

            self.cbf = CBF(domain_lower_bound,domain_upper_bound,discretization)  # CBF object
        else:
            # empty initialization of instance, can be used e.g. for loading data from a file
            pass

    def __getOptSpecs__(self):
        """
        Generates and returns a dictionary of optimization specifications.
        This method ensures that the `warmStartInputTrajectories` attribute is set,
        either by using an existing value or by calling the `setWarmStartInputTrajectories` method.
        It then computes the `h_offset` and constructs a dictionary containing various
        parameters required for optimization.
        Returns:
            dict: A dictionary containing the following keys:
                - 'h': The current value of the barrier function.
                - 'cf': The terminal condition.
                - 'dynamicSystem': The dynamic system being used as string. To use the dynamic system, the string must be converted back to a dynamic system object! Therefore, use a GenericDynamicSystem object to store the dynamic system and call the loadAttributesFromJSON method to convert the string back to a dynamic system object.
                - 'T': The total time horizon.
                - 'N': The number of discretization steps.
                - 'dt': The time step size.
                - 'gamma': The gamma parameter.
                - 'h_offset': The computed offset for the barrier function.
                - 'p_opts': Placeholder for solver options (currently None).
                - 's_opts': Placeholder for solver options (currently None).
                - 'eps': A small epsilon value (default is 10).
                - 'p_norm': The p-norm value.
                - 'p_norm_decrement': The decrement value for the p-norm.
                - 'p_norm_min': The minimum value for the p-norm.
                - 'warmStartInputTrajectories': The warm start input trajectories as multi-dimensional python (not NumPy) array.
        """
        

        if not hasattr(self, 'warmStartInputTrajectories'):
            self.warmStartInputTrajectories = self.setWarmStartInputTrajectories()
        elif self.warmStartInputTrajectories is None:
            self.warmStartInputTrajectories = self.setWarmStartInputTrajectories()

        h_offset = self.computeHoffset()

        opt_specs = {
            'h': aux.funcToJSONString(self.h),
            'cf': aux.funcToJSONString(self.terminal_condition),
            'dynamics': self.dynamics.getAttributesAsJSON(),
            'T': self.T,
            'N': self.N,
            'dt': self.dt,
            'gamma': self.gamma,
            'h_offset':h_offset,
            'p_opts': None,
            's_opts': None,
            'eps': 10,
            'p_norm': self.p_norm, 
            'p_norm_decrement': self.p_norm_decrement,
            'p_norm_min': self.p_norm_min,
            'warmStartInputTrajectories': self.warmStartInputTrajectories.tolist()
        }

        return opt_specs
    
    def __getOptSpecsWithoutSerialization__(self):
        """
        Generates and returns a dictionary containing the optimization specifications 
        along with the system dynamics for the current instance. It does not serialize any functions, 
        therefore the dictionary is not suitable for paralellization, but can be used for directly creating 
        to initialize the CBF computation (see also remark).
        The dictionary includes the following keys:
        - 'h': The current value of the system's state variable.
        - 'cf': The terminal condition for the optimization.
        - 'dynamics': The system dynamics.
        - 'T': The total time horizon for the optimization.
        - 'N': The number of discretization steps.
        - 'dt': The time step size.
        - 'gamma': A parameter related to the optimization problem.
        - 'h_offset': The offset value computed for the state variable.
        - 'p_opts': Placeholder for optimization parameters (currently None).
        - 's_opts': Placeholder for solver options (currently None).
        - 'eps': A small value used in the optimization (default is 10).
        - 'p_norm': The norm used in the optimization.
        - 'p_norm_decrement': The decrement value for the norm.
        - 'p_norm_min': The minimum value for the norm.
        - 'warmStartInputTrajectories': Input trajectories for warm starting the optimization.
        If the attribute 'warmStartInputTrajectories' does not exist or is None, it is 
        initialized by calling the method `setWarmStartInputTrajectories`.

        Remark: creates the dictonary required as argument to CBFcomputation.initializeCbfComputation(). 

        Returns:
            dict: A dictionary containing the optimization specifications and dynamics.
        """

        if not hasattr(self, 'warmStartInputTrajectories'):
            self.warmStartInputTrajectories = self.setWarmStartInputTrajectories()
        elif self.warmStartInputTrajectories is None:
            self.warmStartInputTrajectories = self.setWarmStartInputTrajectories()

        h_offset = self.computeHoffset()

        opt_specs_with_dynamics = {
            'h': self.h,
            'cf': self.terminal_condition,
            'dynamics': self.dynamics,
            'T': self.T,
            'N': self.N,
            'dt': self.dt,
            'gamma': self.gamma,
            'h_offset':h_offset,
            'p_opts': None,
            's_opts': None,
            'eps': 10,
            'p_norm': self.p_norm, 
            'p_norm_decrement': self.p_norm_decrement,
            'p_norm_min': self.p_norm_min,
            'warmStartInputTrajectories': self.warmStartInputTrajectories
        }

        return opt_specs_with_dynamics


    def computeHoffset(self):
        """Computes the smallest value of state constraint function h in any point of the domain.
        
        Returns:
            float: smallest value of h in any point of the domain
        """

        self.h_offset = 0
        
        for idx in np.ndindex(self.cbf.domain[0].shape):
            point = np.array([m[idx] for m in self.cbf.domain])
            self.h_offset = max(self.h_offset,-self.h(point))
        
        return self.h_offset


    def setWarmStartInputTrajectories(self,warmStartInputTrajectories=None):
        """
        Setter method for the warm start input trajectories. The method conducts a check of the validity of the provided warmStartInputTrajectories before setting them. The warm start input trajectories are used as initial guesses for the control input trajectory optimization problem.
        If no warms start input trajectories are provided, the method initializes them with a default value (all zero input trajectory with respective dimenionality) and prints a warning message.

        Args:
            warmStartInputTrajectories (NumPy array of shape (number of trajectories, u_dim, N) or None): warm start input trajectories

        Returns:
            NumPy array of shape (number of trajectories, u_dim, N): warm start input trajectories
        """

        # Check if the warm start input trajectories have the correct shape
        # Required shape: (number of trajectories, u_dim, N)
        if warmStartInputTrajectories is not None and any(warmStartInputTrajectories[k].shape != (self.dynamics.u_dim, self.N) for k in range(warmStartInputTrajectories.shape[0])):
            raise ValueError("The shape of the warm start input trajectories does not match the expected shape.")

        # Generate warm start input trajectories if not provided
        if warmStartInputTrajectories is None:
            warmStartInputTrajectories = np.array([np.zeros((self.dynamics.u_dim,self.N))])
            colorWarning("Warm start input trajectories are not provided. Default initialization is used.")

        self.warmStartInputTrajectories = warmStartInputTrajectories

        return self.warmStartInputTrajectories
    

    def save(self, filename, folder_name="Data"):
        # Create the folder if it does not exist
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            print(f"Folder '{folder_name}' created.")

        # Save the CBFmodule to a file
        print("Saving CBFmodule to file started")

        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        filename = timestamp_str + "_" + filename

        attributes = {}
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):  # Convert arrays to lists
                attributes[key] = value.tolist()
            elif callable(value):  # Save functions or lambdas
                if isinstance(value, types.LambdaType) and value.__name__ == "<lambda>":
                    lambda_source = inspect.getsource(value)
                    attributes[key] = {
                        "type": "lambda",
                        "source": lambda_source[lambda_source.find("=") + 1:].strip()
                    }
                else:
                    attributes[key] = {
                        "type": "function",
                        "name": value.__name__,
                        "source": textwrap.dedent(inspect.getsource(value))
                    }
            elif isinstance(value, DynamicSystem):
                value.save(filename + "_dynamics", folder_name)
            elif isinstance(value, CBF):
                value.save(filename + "_CBF", folder_name)
            elif aux.is_casadi_related(value):  # Convert CasADi matrices to NumPy arrays
                # skip CasADi related objects
                pass
            else:  # Save other types directly
                attributes[key] = value

        file_path = os.path.join(folder_name, filename)

        try:
            with open(file_path, "w") as file:
                json.dump(attributes, file, indent=4)
        except IOError as e:
            print(f"An error occurred while saving the file: {e}")

        print("Saving CBFmodule to file finished")

        return filename

    def load(self, filename, folder_name="Data"):  
        # create the file path
        file_path = os.path.join(folder_name, filename)

        try:
            with open(file_path, "r") as file:
                attributes = json.load(file)
        except IOError as e:
            print(f"An error occurred while loading the file. File not found. Please check the filename and folder name.")

        for key, value in attributes.items():
            if isinstance(value, list):  # Convert lists back to NumPy arrays
                self.__dict__[key] = np.array(value)
            elif isinstance(value, dict):
                if value.get("type") == "function":  # Reconstruct functions
                    exec(value["source"], globals())
                    self.__dict__[key] = eval(value["name"])
                elif value.get("type") == "lambda":  # Reconstruct lambdas
                    self.__dict__[key] = eval(value["source"])
            else:
                self.__dict__[key] = value

        self.dynamics = GenericDynamicSystem()
        self.dynamics.load(filename + "_dynamics", folder_name)
        
        self.cbf = CBF()
        self.cbf.load(filename + "_CBF", folder_name)

        print("Loading CBFmodule from file finished")