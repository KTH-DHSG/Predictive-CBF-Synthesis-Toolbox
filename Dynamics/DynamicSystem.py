"""

    Implementation of an abstract class for handling dynamic systems. 

    It is designed to be used as a base class for any type of dynamic system and provdides the dynamics to the software framework in a suitable format.
    The class provides methods for simulating the system over a time interval with a given control input, resetting the system to its initial state, and saving/loading the system to/from a file. The class also provides methods for defining the system dynamics, which are implemented in subclasses. 

    The implementation uses the casadi library for symbolic computation, which allows for efficient numerical optimization and automatic differentiation.

    (c) Adrian Wiltz, 2025

"""

from abc import ABC, abstractmethod
import numpy as np
import Auxiliaries.auxiliary as aux
import casadi as ca
import json
import inspect
import textwrap
import types
import os

class DynamicSystem(ABC):
    """Abstract class for defining a dynamic system. 
    The class provides a method for simulating the system over a time interval deltaT 
    for a given control input u. The system is simulated with a zero-order hold on the control input. 
    The class also provides a method for resetting the system to its initial state. 
    The class can be saved to a file and loaded from a file. The class is designed as a template for 
    further dynamics classes."""
    
    def __init__(self,x0=None,x_dim=None,u_dim=None,u_min=None,u_max=None):
        """

        Args:
            x0 (NumPy array with length x_dim): initial state
            x_dim (int): number of state dimensions
            u_dim (int): number of control input dimensions
            u_min (NumPy array of with length u_dim): lower bound input constraint
            u_max (NumPy array of with length u_dim): upper bound input constraint

        Raises:
            ValueError: raised if x0 has incorrect length
        """
        
        if not all(value is None for value in [x0, x_dim, u_dim, u_min, u_max]):
            # initialization with variables

            # initialize state dimensions
            if x_dim is None:
                raise ValueError("x_dim must be specified") 
            else:
                self.x_dim = x_dim           # x_dim: length of state vector
            
            # initialize input dimensions
            if u_dim is None:
                raise ValueError("u_dim must be specified")
            else:
                self.u_dim = u_dim           # u_dim: length of input vector
            
            # initialize internal time
            self.t0 = 0                   # t0: initial time of system
            self.t = 0                    # t: current time of system
            self.dt = None                 # dt: integration time interval, updated upon first simulation

            # initialize the initial state  
            if x0 is None:
                raise ValueError("x0 must be specified")
            else:
                if len(x0) == self.x_dim:
                    self.x0 = x0
                    self.x = x0
                else:
                    raise ValueError("x0 has the wrong length. x0 must be an array of length {self.x_dim}") 
            
            # initialize input constraints
            if u_min is None:
                self.u_min = -np.inf*np.ones(self.u_dim) # u_min: lower bound input constraint is unbounded
            else:
                self.u_min = u_min           # u_min: lower bound input constraint
            if u_max is None:
                self.u_max = np.inf*np.ones(self.u_dim)  # u_max: upper bound input constraint is unbounded
            else:   
                self.u_max = u_max           # u_max: upper bound input constraint
            
            # Attributes for storing the solution of the simulated system
            self.x_sol = np.zeros((0, x_dim))  # solution: state trajectory
            self.u_sol = np.zeros((0, u_dim))  # applied input trajectory
            self.t_sol = np.array([self.t0])   # corresponding time series
            self.x_sol = np.vstack([self.x_sol, self.x0])

            self.f_attr = self.f
            self.str_attr = self.__str__
            
        else:
            # empty initialization of instance, can be used e.g. for loading data from a file
            pass
    
    @abstractmethod
    def f(self,x,u):
        """Implementation of the system dynamics.

        Args:
            x (casadi.DM with shape (x_dim,)): current state
            u (casadi.DM with shape (u_dim,)): control input

        Returns:
            casadi.DM: time derivative of state x_dot
        """
        pass
    
    def simulate(self,x=None,u=None,dt=0.1,saveSolution=False):
        """Simulates the system over a time interval dt for a given control input u. 
        The control input is applied with a zero-order hold. The attribute x of the class is updated with the new system state, 
        and the new system state is returned.

        Args:
            x (casadi.DM with shape (x_dim,), optional): current state of the system. If None, the current state of the system is used.
            u (casadi.DM with shape (u_dim,)): control input, applied as zero-order hold to the system on time interval dt.
            dt (float): integration time interval.
            saveSolution (bool): flag to save the solution of the system.

        Returns:
            casadi.DM: state of the system after one time step.
        """

        # Initialize discrete time dynamics function if dt has changed or if the function has not been initialized yet. The latter corresponds to self.dt = None.
        if dt != self.dt:
            if dt == -np.inf or dt == 0 or dt == np.inf:
                raise ValueError(f"Integration time interval dt may not be self.dt")
            # update integration time interval if it has changed and has a valid value
            self.dt = dt
            # initialize one step simulation function
            self.f_discete = self.__createOneStepSimultationFcn__(dt=dt)

        # check if x is given, otherwise use the class attributes
        if x is None:
            x = self.x
        if u is None:
            raise ValueError("Control input u must be specified")

        # compute next state
        x_next = self.f_discete(x,u)

        # save solution
        if saveSolution:
            self.x_sol = np.vstack([self.x_sol, np.array(x_next).reshape(1, -1)])
            self.u_sol = np.vstack([self.u_sol, np.array(u).reshape(1, -1)])
            self.t_sol = np.append(self.t_sol, self.t + dt)

            self.t = self.t + dt
            self.x = x_next
        
        return x_next
    
    def simulateOverHorizon(self, x0=None, u=None, dt=0.1):
        """Simulates the system over a horizon of N time steps with control input u applied with a zero-order hold. 
        The attribute x of the class is updated with new system states, and the new system states are returned.

        Args:
            x0 (casadi.DM with shape (x_dim,)): initial state
            u (casadi.DM with shape (u_dim,N)): control input, applied as zero-order hold to the system on time interval dt
            dt (float): integration time interval
            
        Returns:
            np.ndarray: time trajectory
            np.ndarray: state trajectory
        """
        # initializations
        if x0 is None:
            x0 = self.x0
        if u is None:
            raise ValueError("Control input u must be specified")
        
        N = u.shape[1]

        # simulate system
        stateTrajectory = ca.DM.zeros((self.x_dim, N+1))
        timeTrajectory = ca.DM.zeros(N+1)
        stateTrajectory[:, 0] = x0
        timeTrajectory[0] = self.t0
        for k in range(N):
            stateTrajectory[:, k+1] = self.simulate(stateTrajectory[:, k], u[:, k], dt, saveSolution=False)
            timeTrajectory[k+1] = timeTrajectory[k] + dt
        
        return np.array(timeTrajectory), np.array(stateTrajectory)
    
    def __createOneStepSimultationFcn__(self, dt=0.1, number_of_finite_elements=4):
        """
        Creates a CasADi function for simulating the system over one time step with a given control input.
        The function uses the CasADi integrator to discretize the system dynamics. 

        Args:
            dt (float): integration time interval
            number_of_finite_elements (int): number of finite elements for the integrator

        Returns:    
            casadi.Function: CasADi function for simulating the system over one time step
        """

        state = ca.MX.sym('state', self.x_dim)   
        control = ca.MX.sym('control', self.u_dim)
        dynamics = ca.Function('dynamics', [state, control], [self.f(state, control)])

        # Use CasADi's built-in integrator for discretization
        ode = {'x': state,
                'p': control,
                'ode': dynamics(state, control)}       # Dynamics: dx/dt = f(x, u)
        opts = {'simplify': True,                      # Simplify the integrator model
                'number_of_finite_elements': number_of_finite_elements} # Number of finite elements for the integrator
        integrator = ca.integrator('integrator', 'rk', ode, 0, dt, opts)

        x_next = integrator(x0=state, p=control)['xf']
        dynamics_discretized = ca.Function('dynamics_discretized', [state, control], [x_next], ['x', 'u'], ['x_next'])

        return dynamics_discretized

    
    def reset(self):
        """Resets system settings to initialization and resets the solution arrays.
        """
        # reset state
        self.x = self.x0
        self.t = 0
        
        # reset solutions
        self.x_sol = ca.DM.zeros((0, self.x_dim))
        self.x_sol = ca.vertcat(self.x_sol, ca.DM(self.x0).T)
        self.u_sol = ca.DM.zeros((0, self.u_dim))
        self.t_sol = ca.DM([self.t0])

    def __getAttributes__(self):
        """Returns the attributes of the class as a dictionary.
        
        Returns:
            dict: dictionary of class attributes
        """

        attributes = {}
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):  # Convert arrays to lists
                attributes[key] = value.tolist()
            elif aux.is_casadi_related(value):  # Convert CasADi matrices to NumPy arrays
                # skip CasADi related objects
                pass
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
            else:  # Save other types directly
                attributes[key] = value
        
        return attributes
    
    def getAttributesAsJSON(self):
        """Returns the attributes of the class as a JSON string.
        
        Returns:
            str: JSON string of class attributes
        """
        return json.dumps(self.__getAttributes__(), indent=4)

    def save(self, filename, folder_name="Data"):
        # Create the folder if it does not exist
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            print(f"Folder '{folder_name}' created.")

        # Save the DynamicSystem to a file
        print("Saving DynamicSystem to file started")
        attributes = self.__getAttributes__()

        file_path = os.path.join(folder_name, filename)

        with open(file_path, "w") as file:
            json.dump(attributes, file, indent=4)

        print("Saving DynamicSystem to file finished")


    def __loadAttributes__(self, attributes):
        """Loads the attributes of the class from a dictionary.
        
        Args:
            attributes (dict): dictionary of class attributes
        """

        for key, value in attributes.items():
            if isinstance(value, list):  # Convert lists back to NumPy arrays
                self.__dict__[key] = np.array(value)
            elif isinstance(value, dict) and "source" in value:  # Reconstruct functions
                # Execute the source code to recreate the function
                exec(value["source"], globals())
                # Bind the function to the instance
                self.__dict__[key] = types.MethodType(eval(value["name"]), self)
            else:
                self.__dict__[key] = value

    def loadAttributesFromJSON(self, json_string):
        """Loads the attributes of the class from a JSON string. 

        Args:
            json_string (str): JSON string of class attributes

        Returns:
            dict: dictionary of class attributes
        """
        attributes = json.loads(json_string)
        self.__loadAttributes__(attributes)

        return attributes

    def load(self, filename, folder_name="Data"):
        # Create the file path
        file_path = os.path.join(folder_name, filename )

        with open(file_path, "r") as file:
            attributes = json.load(file)

        self.__loadAttributes__(attributes)

        self.reset()

        