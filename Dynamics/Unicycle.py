"""

    Implementation of the dynamics of the kinematic unicycle model.

    The class includes methods for initialization, the dynamic model, a line following controller and various functions for plotting simulation results. Further functions for loading and saving data are inherited from the DynamicSystem class.

    The implementation uses the casadi library for symbolic computation, which allows for efficient numerical optimization and automatic differentiation.

    (c) Adrian Wiltz, 2025

"""

import numpy as np
from Dynamics.DynamicSystem import DynamicSystem
from Auxiliaries import auxiliary_math as aux_math
import casadi as ca

class Unicycle(DynamicSystem):
    
    def __init__(self,x0=None,u_min=None,u_max=None):
        """Initialization,

        Args:
            x0 (NumPy array with length 3): initial state
            u_min (NumPy array of with length u_dim=2): lower bound input constraints
            u_max (NumPy array of with length u_dim=2): upper bound input constraints
        """
        
        if not all(value is None for value in [x0,u_min,u_max]):
            # initialization with variables
            x_dim = 3
            u_dim = 2

            v_min = u_min[0]
            omega_max = u_max[1]

            self.turning_radius = v_min / omega_max

            super().__init__(x0,x_dim,u_dim,u_min,u_max)
        else:
            # empty initialization of instance, can be used e.g. for loading data from a file
            pass
    
    def f(self, x, u) -> ca.MX:
        """Implementation of the unicycle model using casadi data types.

        Args:
            x (casadi.MX with length 3): current state
            u (casadi.MX with length 2): control input

        Returns:
            casadi.MX: time derivative of system state
        """
        x_dot1 = u[0] * ca.cos(x[2])
        x_dot2 = u[0] * ca.sin(x[2])
        x_dot3 = u[1]
        x_dot = ca.vertcat(x_dot1, x_dot2, x_dot3)

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
        numpy.ndarray: The control input vector [v_d, omega], where omega is the control input for the
                       orientation adjustment based on the desired vehicle orientation.
        """
        
        # extract required states
        y_state = float(x[1]) # -0.15
        psi_state = float(x[2])

        k_psi = np.pi/4     # Maximum value of desired orientation angle in radians in order to get back to the straight line 
        k_damp = 0.3
        k = 1

        psi_d = -k_psi * np.clip(k_damp * (y_state - y_d), -1, 1)

        omega = k * (psi_d - psi_state) 

        u = np.array([v_d, omega])

        return u
    
    @staticmethod
    def plot_trajectory_as_triangular_marker(plt, unicycle_object, marker_shape=(0.5,0.25), marker_spacing=10, line_color = 'b', linewidth=2, marker_color='k', marker_linewidth=1):
        """
        Plots the trajectory of a unicycle object as a line with triangular markers indicating the orientation.

        Parameters:
            plt (module): Matplotlib pyplot module for plotting.
            unicycle_object (object): Unicycle object or DynamicSystem object representing the bicycle.
            marker_shape (tuple): A tuple (length, width) defining the size of the triangular markers. Default is (0.5, 0.25).
            marker_spacing (int): The spacing between consecutive markers along the trajectory. Default is 10.
            line_color (str): Color of the trajectory line. Default is 'b' (blue).
            linewidth (int): Width of the trajectory line. Default is 2.
            marker_color (str): Color of the triangular markers. Default is 'k-' (black).
            marker_linewidth (int): Width of the triangular marker lines. Default is 1.
        
        Returns:
            None
        """

        marker_length = marker_shape[0]
        marker_width = marker_shape[1]

        plt.plot(unicycle_object.x_sol[:,0], unicycle_object.x_sol[:,1], color=line_color, linewidth=linewidth)  # Draw trajectory

        for i in range(0,unicycle_object.x_sol.shape[0],marker_spacing):
            x = unicycle_object.x_sol[i,0]
            y = unicycle_object.x_sol[i,1]
            theta = unicycle_object.x_sol[i,2]

            # Compute triangle vertices relative to (x, y)
            front_x = x + marker_length * np.cos(theta)
            front_y = y + marker_length * np.sin(theta)
            
            left_x = x + marker_width * np.cos(theta + 2*np.pi/3)
            left_y = y + marker_width * np.sin(theta + 2*np.pi/3)

            right_x = x + marker_width * np.cos(theta - 2*np.pi/3)
            right_y = y + marker_width * np.sin(theta - 2*np.pi/3)

            # Draw triangle marker
            triangle = np.array([[front_x, front_y], [left_x, left_y], [right_x, right_y], [front_x, front_y]])
            plt.plot(triangle[:, 0], triangle[:, 1], color=marker_color, linewidth=marker_linewidth)  # Draw triangle@staticmethod

    @staticmethod
    def plot_trajectory_at_indices_as_triangular_marker(plt, unicycle_object, indices, marker_shape=(0.5,0.25), marker_color='k', marker_linewidth=1):
        """
        Plots the trajectory of a unicycle object as a line with triangular markers indicating the orientation.

        Parameters:
            plt (module): Matplotlib pyplot module for plotting.
            unicycle_object (object): Unicycle object or DynamicSystem object representing the bicycle.
            marker_shape (tuple): A tuple (length, width) defining the size of the triangular markers. Default is (0.5, 0.25).
            marker_spacing (int): The spacing between consecutive markers along the trajectory. Default is 10.
            line_color (str): Color of the trajectory line. Default is 'b' (blue).
            linewidth (int): Width of the trajectory line. Default is 2.
            marker_color (str): Color of the triangular markers. Default is 'k-' (black).
            marker_linewidth (int): Width of the triangular marker lines. Default is 1.
        
        Returns:
            None
        """

        marker_length = marker_shape[0]
        marker_width = marker_shape[1]

        for i in indices:
            x = unicycle_object.x_sol[i,0]
            y = unicycle_object.x_sol[i,1]
            theta = unicycle_object.x_sol[i,2]

            # Compute triangle vertices relative to (x, y)
            front_x = x + marker_length * np.cos(theta)
            front_y = y + marker_length * np.sin(theta)
            
            left_x = x + marker_width * np.cos(theta + 2*np.pi/3)
            left_y = y + marker_width * np.sin(theta + 2*np.pi/3)

            right_x = x + marker_width * np.cos(theta - 2*np.pi/3)
            right_y = y + marker_width * np.sin(theta - 2*np.pi/3)

            # Draw triangle marker
            triangle = np.array([[front_x, front_y], [left_x, left_y], [right_x, right_y], [front_x, front_y]])
            plt.plot(triangle[:, 0], triangle[:, 1], color=marker_color, linewidth=marker_linewidth)  # Draw triangle

    
    def __str__(self):
        str = "Kinematic Unicycle Model"
        return str