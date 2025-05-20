"""

    Implementation of the dynamics of general linear dynamics.

    The class includes methods for initialization, the dynamic model, a line following controller and various functions for plotting simulation results. Further functions for loading and saving data are inherited from the DynamicSystem class.

    The implementation uses the casadi library for symbolic computation, which allows for efficient numerical optimization and automatic differentiation.

    (c) Adrian Wiltz, 2025

"""

import numpy as np
from Dynamics.DynamicSystem import DynamicSystem
import casadi as ca

class LinearSystem(DynamicSystem):
    
    def __init__(self,A,B,x0=None,u_min=None,u_max=None):
        """Initialization.

        Args:
            A (NumPy array): state matrix
            B (NumPy array): input matrix
            x0 (NumPy array with length x_dim): initial state vector; Default is zero vector with corresponding dimension
            u_min (NumPy array of with length u_dim): lower bound input constraint; Default is -inf with corresponding dimension
            u_max (NumPy array of with length u_dim): upper bound input constraint; Default is inf with corresponding dimension
        """

        x_dim = A.shape[0]
        u_dim = B.shape[1]

        if x0 is None:
            x0 = np.zeros(x_dim)

        if u_min is None:
            u_min = np.full((u_dim,), -np.inf)

        if u_max is None:
            u_max = np.full((u_dim,), np.inf)

        if not all(value is None for value in [A,B,x0,u_min,u_max]):
            # initialization with variables
            super().__init__(x0,x_dim,u_dim,u_min,u_max)
        else:
            # empty initialization of instance, can be used e.g. for loading data from a file
            pass     

        self.A = A
        self.B = B      
    
    def f(self, x, u):
        """Implementation of the single integrator dynamics using casadi data types.

        Args:
            x (casadi.MX or casadi.SX): current state
            u (casadi.MX or casadi.SX): control input

        Returns:
            casadi.MX or casadi.SX: time derivative of system state
        """

        return ca.mtimes(self.A, x) + ca.mtimes(self.B, u)
    
    def __str__(self):
        str = "LinearSystem with state matrix A:\n"
        str += str(self.A)  
        str += "\n and input matrix B:\n"
        str += str(self.B)
        return str