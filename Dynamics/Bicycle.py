"""

    Implementation of the dynamics of the kinematic bicycle model.

    The class includes methods for initialization, the dynamic model, a line following controller and various functions for plotting simulation results. Further functions for loading and saving data are inherited from the DynamicSystem class.

    The implementation uses the casadi library for symbolic computation, which allows for efficient numerical optimization and automatic differentiation.

    (c) Adrian Wiltz, 2025

"""

import numpy as np
from Dynamics.DynamicSystem import DynamicSystem
import casadi as ca

class Bicycle(DynamicSystem):
    
    def __init__(self,x0=None,L=None,u_min=None,u_max=None):
        """Initialization,

        Args:
            x0 (NumPy array with length 3): initial state
            L (float): bicyle length, distance between wheels
            u_min (NumPy array of with length u_dim): lower bound input constraint
            u_max (NumPy array of with length u_dim): upper bound input constraint
        """

        if not all(value is None for value in [x0,L,u_min,u_max]):
            # initialization with variables
            x_dim = 3
            u_dim = 2
            super().__init__(x0,x_dim,u_dim,u_min,u_max)
            self.L = L
            beta = np.arctan(0.5 * np.tan(u_max[1]))
            self.turning_radius = L/(np.cos(beta)*np.tan(u_max[1]))
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
        beta = ca.atan(0.5 * ca.tan(u[1]))
        x_dot1 = u[0] * ca.cos(x[2] + beta)
        x_dot2 = u[0] * ca.sin(x[2] + beta)
        x_dot3 = u[0] * ca.cos(beta) * ca.tan(u[1]) / self.L
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
        numpy.ndarray: The control input vector [v_d, rho], where rho is the control input for the 
                       orientation adjustment based on the desired vehicle orientation.
        """

        # extract required states
        y_state = float(x[1]) # -0.15
        psi_state = float(x[2])

        # desired vehicle orientation
        psi_d = -np.pi/2 * np.exp(y_state-y_d)/(np.exp(y_state-y_d) + 1) + np.pi/4
        rho = 2 * (psi_d - psi_state)

        u = np.array([v_d, rho])

        return u
    
    @staticmethod
    def plot_trajectory_as_triangular_marker(plt, bicycle_object, marker_shape=(0.5,0.25), marker_spacing=10, line_color = 'b', linewidth=2, marker_color='k-', marker_linewidth=1):
        """
        Plots the trajectory of a bicycle object as a line with triangular markers indicating the orientation.

        Parameters:
            plt (module): Matplotlib pyplot module for plotting.
            bicycle_object (object): Bicycle object or DynamicSystem object representing the bicycle.
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

        plt.plot(bicycle_object.x_sol[:,0], bicycle_object.x_sol[:,1], color=line_color, linewidth=linewidth)  # Draw trajectory

        for i in range(0,bicycle_object.x_sol.shape[0],marker_spacing):
            x = bicycle_object.x_sol[i,0]
            y = bicycle_object.x_sol[i,1]
            theta = bicycle_object.x_sol[i,2]

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

    @staticmethod
    def plot_trajectory_with_bicycle_markers(plt, bicycle_object, steps=1, marker_spacing=10, line_color = 'b', linewidth=2, marker_size=1, dot_size=2, label=None):
        """
        Plots the trajectory of a bicycle object with markers at specified intervals.

        Parameters:
            plt (matplotlib.pyplot): The matplotlib pyplot module for plotting.
            bicycle_object (object): An object representing the bicycle, which contains the trajectory data.
            steps (int, optional): The number of steps to skip when plotting the trajectory. Default is 1.
            marker_spacing (int, optional): The spacing between markers along the trajectory. Default is 10.
            line_color (str, optional): The color of the trajectory line. Default is 'b' (blue).
            linewidth (int, optional): The width of the trajectory line. Default is 2.
            marker_size (int, optional): The size of the markers. Default is 1.
            dot_size (int, optional): The size of the center dot. Default is 2.

        Returns:
            None
        """

        plt.plot(bicycle_object.x_sol[::steps,0], bicycle_object.x_sol[::steps,1], color=line_color, linewidth=linewidth, label=label)  # Draw trajectory

        Bicycle.plot_trajectory_at_indices_with_bicycle_markers(plt, bicycle_object, range(0,bicycle_object.x_sol.shape[0],marker_spacing), marker_size=marker_size, dot_size=dot_size, dot_color=line_color)

    @staticmethod
    def plot_trajectory_at_indices_with_bicycle_markers(plt, bicycle_object, indices, marker_size=1, dot_size=2, frame_color='k', dot_color=None, wheel_color='r', label=None):
        """
        Plots the trajectory of a bicycle model at specified indices with markers representing the bicycle's position and orientation.
        
        Parameters:
            plt (matplotlib.pyplot): The matplotlib plotting object.
            bicycle_object (object): An object containing the bicycle's state and control solutions.
                - x_sol (numpy.ndarray): Array of shape (N, 3) containing the state solutions [x, y, theta].
                - u_sol (numpy.ndarray): Array of shape (N, 2) containing the control solutions [v, delta].
            indices (list of int): List of indices at which to plot the bicycle markers.
            line_color (str, optional): Color of the trajectory line. Default is 'b' (blue).
            linewidth (int, optional): Width of the trajectory line. Default is 2.
            marker_size (int, optional): Size of the bicycle markers. Default is 1.
            dot_size (int, optional): Size of the center dot. Default is 2.
        
        Returns:
            None
        """

        wheelbase = 0.6 * marker_size  # Distance between front and rear wheels
        wheel_length = 0.3 * marker_size  # Length of the wheels

        for i in indices:
            x = bicycle_object.x_sol[i,0]
            y = bicycle_object.x_sol[i,1]
            theta = bicycle_object.x_sol[i,2]

            if i < bicycle_object.u_sol.shape[0]-1:
                delta = bicycle_object.u_sol[i,1]
            else:
                # set steering angle to zero for last time step
                delta = 0

            rear_x = x - (wheelbase / 2) * np.cos(theta)
            rear_y = y - (wheelbase / 2) * np.sin(theta)

            front_x = x + (wheelbase / 2) * np.cos(theta)
            front_y = y + (wheelbase / 2) * np.sin(theta)

            # Compute wheel endpoints for visualization
            # Rear wheel
            rear_wheel_x1 = rear_x + wheel_length / 2 * np.cos(theta)
            rear_wheel_y1 = rear_y + wheel_length / 2 * np.sin(theta)
            rear_wheel_x2 = rear_x - wheel_length / 2 * np.cos(theta)
            rear_wheel_y2 = rear_y - wheel_length / 2 * np.sin(theta)

            # Front wheel (rotated by steering angle)
            front_wheel_x1 = front_x + wheel_length / 2 * np.cos(theta + delta)
            front_wheel_y1 = front_y + wheel_length / 2 * np.sin(theta + delta)
            front_wheel_x2 = front_x - wheel_length / 2 * np.cos(theta + delta)
            front_wheel_y2 = front_y - wheel_length / 2 * np.sin(theta + delta)

            # Draw frame (line connecting rear and front wheel)
            if i == indices[0]:
                plt.plot([rear_x, front_x], [rear_y, front_y], color=frame_color, linewidth=1.5, label=label)  # Bicycle frame
            else:
                plt.plot([rear_x, front_x], [rear_y, front_y], color=frame_color, linewidth=1.5)  # Bicycle frame

            # Draw center dot
            if dot_color is None:
                dot_color = frame_color

            plt.plot(x, y, marker='o', color=dot_color, markersize=dot_size)

            # Draw rear wheel
            plt.plot([rear_wheel_x1, rear_wheel_x2], [rear_wheel_y1, rear_wheel_y2], color=wheel_color, linewidth=2)

            # Draw front wheel
            plt.plot([front_wheel_x1, front_wheel_x2], [front_wheel_y1, front_wheel_y2], color=wheel_color, linewidth=2)
            
    def __str__(self):
        str = "Kinematic Bicycle Model"
        return str