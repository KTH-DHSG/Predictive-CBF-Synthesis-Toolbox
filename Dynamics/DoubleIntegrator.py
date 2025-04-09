"""

    Implementation of the dynamics of the double integrator dynamics.

    The class includes methods for initialization, the dynamic model, a line following controller and various functions for plotting simulation results. Further functions for loading and saving data are inherited from the DynamicSystem class.

    The implementation uses the casadi library for symbolic computation, which allows for efficient numerical optimization and automatic differentiation.

    (c) Adrian Wiltz, 2025

"""

import numpy as np
from Dynamics.DynamicSystem import DynamicSystem
import casadi as ca

class DoubleIntegrator(DynamicSystem):
    
    def __init__(self,x_dim=4,x0=None,u_min=None,u_max=None):
        """Initialization,

        Args:
            x0 (NumPy array with length 3): initial state
            u_min (NumPy array of with length u_dim): lower bound input constraint
            u_max (NumPy array of with length u_dim): upper bound input constraint
        """

        if not all(value is None for value in [x0,u_min,u_max]):
            # initialization with variables
            if x_dim % 2 == 0:
                u_dim = x_dim // 2
            else:
                raise ValueError("The state dimension must be even.")
            if len(u_min) != u_dim or len(u_max) != u_dim:
                raise ValueError("The length of u_min and u_max must be equal to the control input dimension.")
            
            super().__init__(x0,x_dim,u_dim,u_min,u_max)
            
        else:   
            # empty initialization of instance, can be used e.g. for loading data from a file
            pass
    
    def f(self, x, u):
        """Implementation of the double integrator dynamics.

        Args:
            x (casadi.MX or casadi.SX with length 3): current state
            u (casadi.MX or casadi.SX with length 2): control input

        Returns:
            casadi.MX or casadi.SX: time derivative of system state
        """

        pos = x[0:self.x_dim//2]
        vel = x[self.x_dim//2:]

        x_dot = ca.vertcat(vel, u)

        return x_dot
    
    @staticmethod
    def u_follow_straight_line(x,v_d,y_d=0):
        """
        Computes the control input to follow a straight line defined by y = y_d.

        Parameters:
        x (array-like): The state vector of the vehicle, where x[0] is the x-position, x[1] is the y-position (y_state) 
                        and x[2] is the vehicle orientation (psi_state).
        v_d (float): The desired velocity of the vehicle.
        y_d (float, optional): The desired y-position of the vehicle. Default is 0.
        
        Returns:
        numpy.ndarray: The control input vector [v_d, rho], where rho is the control input for the 
                       orientation adjustment based on the desired vehicle orientation.
        """

        K = np.array([[0,0,1/np.sqrt(2),0],
                       [0,0.5,0,1.2]])

        error = np.array([x[0], x[1]-y_d,x[2]-v_d,x[3]]).flatten()

        u = -K @ error

        return u
    
    @staticmethod
    def plot_trajectory_with_markers(plt, dintegrator_object, steps=1, marker_spacing=10, line_color = 'b', linewidth=2, marker_size=2):
        """
        Plots the trajectory of a bicycle object with markers at specified intervals.

        Parameters:
            plt (matplotlib.pyplot): The matplotlib pyplot module for plotting.
            dintegrator_object (object): An object representing the bicycle, which contains the trajectory data.
            steps (int, optional): The number of steps to skip when plotting the trajectory. Default is 1.
            marker_spacing (int, optional): The spacing between markers along the trajectory. Default is 10.
            line_color (str, optional): The color of the trajectory line. Default is 'b' (blue).
            linewidth (int, optional): The width of the trajectory line. Default is 2.
            marker_size (int, optional): The size of the markers. Default is 2.

        Returns:
            None
        """

        plt.plot(dintegrator_object.x_sol[::steps,0], dintegrator_object.x_sol[::steps,1], color=line_color, linewidth=linewidth)  # Draw trajectory

        DoubleIntegrator.plot_trajectory_at_indices_withmarkers(plt, dintegrator_object, range(0,dintegrator_object.x_sol.shape[0],marker_spacing), marker_size=marker_size, color=line_color)

    @staticmethod
    def plot_trajectory_at_indices_withmarkers(plt, dintegrator_object, indices, marker_size=2, color='k'):
        """
        Plots the trajectory of a bicycle model at specified indices with markers representing the bicycle's position and orientation.
        
        Parameters:
            plt (matplotlib.pyplot): The matplotlib plotting object.
            dintegrator_object (object): An object containing the bicycle's state and control solutions.
                - x_sol (numpy.ndarray): Array of shape (N, 3) containing the state solutions [x, y, theta].
                - u_sol (numpy.ndarray): Array of shape (N, 2) containing the control solutions [v, delta].
            indices (list of int): List of indices at which to plot the bicycle markers.
            line_color (str, optional): Color of the trajectory line. Default is 'b' (blue).
            linewidth (int, optional): Width of the trajectory line. Default is 2.
            marker_size (int, optional): Size of the bicycle markers. Default is 2.
        
        Returns:
            None
        """

        for i in indices:
            x = dintegrator_object.x_sol[i,0]
            y = dintegrator_object.x_sol[i,1]

            # Draw center dot
            plt.plot(x, y, marker='o', color=color, markersize=marker_size)

    def __str__(self):
        str = "Double Integrator"
        return str