""""

    Auxiliary functions for numerical analysis and interpolation.

    These functions include gradient approximation, directional gradient approximation, checking if a point is within a certian domain, and a sigmoid function.

    (c) Adrian Wiltz, 2025

"""

import numpy as np

def approximate_gradient(function, x, h=1e-5):
    """
    Approximates the gradient of an interpolated function at a given point using central differences.

    Parameters:
    - function: RegularGridInterpolator object
    - x: ndarray, shape (n,) -> the point where the gradient is computed
    - h: float -> small step size for numerical differentiation

    Returns:
    - gradient: ndarray, shape (n,) -> numerical gradient at the given point
    """
    point = np.asarray(x)  # Ensure the point is a NumPy array
    forward_gradient = np.zeros_like(point, dtype=float)  # Initialize gradient array
    backward_gradient = np.zeros_like(point, dtype=float)  # Initialize gradient array

    for i in range(len(point)):  # Loop over each dimension
        step = np.zeros_like(point)
        step[i] = h  # Perturb one dimension at a time

        forward = function(point + step) 
        backward = function(point) 

        # Handle output from interpolator which is a list of one element
        if isinstance(forward, np.ndarray) and forward.size == 1:
            forward = forward[0]
        if isinstance(backward, np.ndarray) and backward.size == 1:
            backward = backward[0]

        forward_gradient[i] = (forward - backward) / (h)  # Central difference formula

    for i in range(len(point)):  # Loop over each dimension
        step = np.zeros_like(point)
        step[i] = h  # Perturb one dimension at a time

        forward = function(point)
        backward = function(point - step) 

        # Handle output from interpolator which is a list of one element
        if isinstance(forward, np.ndarray) and forward.size == 1:
            forward = forward[0]
        if isinstance(backward, np.ndarray) and backward.size == 1:
            backward = backward[0]

        backward_gradient[i] = (forward - backward) / (h)  # Central difference formula

    grads = [forward_gradient, backward_gradient]
    grad_norms = [np.linalg.norm(forward_gradient), np.linalg.norm(backward_gradient)]

    # Choose gradient with sufficiently large norm
    if grad_norms[0] > 0.1:
        grad_idx = 0
    else:
        grad_idx = np.argmax(grad_norms)

    return grads[grad_idx]

def approximate_directional_gradient(interpolator, x, u, dynamics, step_size=0.1):
    """
    Approximate the directional gradient of a function at a given point.

    Parameters:
        interpolator (callable): A function that interpolates the value at a given point.
        x (array-like): The point at which to evaluate the gradient.
        u (array-like): The control input or direction vector.
        dynamics (object): An object with a method `f(x, u)` that computes the dynamics at point `x` with input `u`.
        step_size (float, optional): The step size for the finite difference approximation. Default is 0.1.

    Returns:
        gradient (float): The approximated directional gradient at the given point.
    """


    point = np.asarray(x)  # Ensure the point is a NumPy array
    gradient = np.zeros_like(point, dtype=float)  # Initialize gradient array

    f_x_u = np.asarray(dynamics.f(x,u))
    f_x_u_norm = np.linalg.norm(f_x_u)

    eps = 0.01

    if not (f_x_u_norm < eps):
        new_point = point + f_x_u/f_x_u_norm * step_size

        current = interpolator(point)[0]  
        forward = interpolator(new_point)[0]

        gradient = (forward - current) / step_size
    else:
        # return zero gradient if the norm of the dynamics is zero
        pass

    return gradient

def isPointWithinDomain(grid_points,point):
    """
    Check if a given point is within the domain defined by grid points.

    Parameters:
    grid_points (list of list of float): A list where each element is a list of grid points defining the domain in each dimension.
    point (list of float): A list of coordinates representing the point to be checked.
    
    Returns:
    bool: True if the point is within the domain, False otherwise.
    """

    is_within_domain = True  # Assume the point is within the domain

    for i in range(len(grid_points)):  # Loop over each dimension
        if point[i] < grid_points[i][0] or point[i] > grid_points[i][-1]:  # Check if the point is outside the domain
            is_within_domain = False  # Update the flag
            break  # Exit the loop

    return is_within_domain

def sigmoid(t):
    """
    Compute the sigmoid function for a given input. The sigmoid function is defined as f(t) = exp(t)/(1+exp(t)).

    Remarks: 
    - The sigmoid function maps the input to the range (0,1) with sigmoid(0) = 0.5.
    - For the derivative to equal 1 at the origin, scale scale t with 1/4, i.e., call sigmoid(t/4).
    
    Parameters:
        t (float or ndarray): Input to the sigmoid function.
    
    Returns:
        float or ndarray: Output of the sigmoid function.
    """

    sigmoid = np.exp(t)/(1+np.exp(t))

    return sigmoid


########################################################################################
# Test the functions

# if __name__ == "__main__":

#     import sys
#     import os

#     sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#     from Dynamics.Bicycle import Bicycle
#     from CBF.CBFmodule import CBFmodule

#     test_point = np.array([-3,0,0])
#     test_input = np.array([2,0.5])


#     cbf_module_filename = "2025-02-11_21-51-13_bicycle_example_1_cbf_module_finer_grid_1.json"
#     cbf_module_folder_path = r'Examples\Bicycle_1\Data'

#     bicycle_cbf_module = CBFmodule()
#     bicycle_cbf_module.load(cbf_module_filename, cbf_module_folder_path)

#     cbf_interpolator = bicycle_cbf_module.cbf.getCbfInterpolator()

#     my_bicycle = bicycle_cbf_module.dynamics

#     print(approximate_directional_gradient(cbf_interpolator,test_point,test_input,my_bicycle,step_size=0.1))
