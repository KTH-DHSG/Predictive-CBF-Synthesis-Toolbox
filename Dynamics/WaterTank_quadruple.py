"""

    Implementation of a quadruple water tank dynamic system.
    
    (c) Adrian Wiltz, 2025

"""

import numpy as np
from Dynamics.DynamicSystem import DynamicSystem
import casadi as ca

class WaterTank(DynamicSystem):
    
    def __init__(self,x0=None,A_1=30,a_1=0.6,A_2=30,a_2=0.3,A_3=30,a_3=0.4,A_4=30,a_4=0.4,u_min=None,u_max=None):
        """Initialization,

        Args:
            x0 (NumPy array with length 3): initial state
            L (float): bicyle length, distance between wheels
            u_min (NumPy array of with length u_dim): lower bound input constraint
            u_max (NumPy array of with length u_dim): upper bound input constraint
        """

        if not all(value is None for value in [x0,A_1,a_1,A_2,a_2,A_3,a_3,A_4,a_4,u_min,u_max]):
            # initialization with variables
            x_dim = 4
            u_dim = 2
            super().__init__(x0,x_dim,u_dim,u_min,u_max)
            self.A1 = A_1
            self.a1 = a_1
            self.A2 = A_2
            self.a2 = a_2
            self.A3 = A_3
            self.a3 = a_3
            self.A4 = A_4
            self.a4 = a_4
            self.g = 9.81  # gravitational acceleration in m/s^2

            self.u_min = u_min
            self.u_max = u_max
        else: 
            # empty initialization of instance, can be used e.g. for loading data from a file
            pass
    
    def f(self, x, u):
        """Implementation of the kinematic bicycle model using casadi data types.

        Args:
            x (casadi.MX with length 3): current state
            u (casadi.MX with length 2): control input

        Returns:
            casadi.MX: time derivative of system state
        """

        h1 = x[0]
        h2 = x[1]
        h3 = x[2]
        h4 = x[3]
    
        v1 = u[0]
        v2 = u[1]

        h1_dot = 1/self.A1 * v1 - self.a1/self.A1 * ca.sqrt(2.0*self.g*h1) - 0.5*self.a2/self.A2 * ca.sqrt(2.0*self.g*h2)
        h2_dot = 1/self.A2 * v2 - self.a2/self.A2 * ca.sqrt(2.0*self.g*h2)
        h3_dot = self.a1/self.A1 * ca.sqrt(2.0*self.g*h1) - self.a3/self.A3 * ca.sqrt(2.0*self.g*h3)
        h4_dot = 1.5*self.a2/self.A2 * ca.sqrt(2.0*self.g*h2) - self.a4/self.A4 * ca.sqrt(2.0*self.g*h4)

        x_dot = ca.vertcat(h1_dot, h2_dot, h3_dot, h4_dot)

        return x_dot    

    def __str__(self):
        str = "Watertank"
        return str