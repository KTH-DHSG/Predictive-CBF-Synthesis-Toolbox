"""

    Implementation of a generic dynamics system. 

    The class is suitable for loading any arbitrary type of DynamicSystem from a file.
    
    The class is a subclass of the DynamicSystem class.

    (c) Adrian Wiltz, 2025

"""

import numpy as np
from Dynamics.DynamicSystem import DynamicSystem
import casadi as ca

class GenericDynamicSystem(DynamicSystem):
    """
    The class can be used for loading any arbitrary type of DynamicSystem from a file.
    """
    
    def __init__(self,x0=None,u_min=None,u_max=None):
        """Initialization,

        Args:
            x0 (NumPy array with length 3): initial state
            L (float): bicyle length, distance between wheels
            u_min (NumPy array of with length u_dim): lower bound input constraint
            u_max (NumPy array of with length u_dim): upper bound input constraint
        """

        if not all(value is None for value in [x0,u_min,u_max]):
            super().__init__(x0=x0,u_min=u_min,u_max=u_max)   
        
        else:
            # empty initialization of instance, can be used e.g. for loading data from a file
            pass       
    
    def f(self,x,u):
        """
        Args:
            x (NumPy array with length 3): current state
            u (NumPy array with length 2): control input

        Returns:
            np.ndarray: time derivative of system state
        """
        
        return self.f_attr(x,u)
    
    def __str__(self):
        return self.str_attr()