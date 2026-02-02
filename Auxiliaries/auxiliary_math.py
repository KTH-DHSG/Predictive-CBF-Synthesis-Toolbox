""""

    Auxiliary functions for numerical analysis and interpolation.

    These functions include gradient approximation, directional gradient approximation, checking if a point is within a certian domain, and a sigmoid function.

    (c) Adrian Wiltz, 2025

"""

import numpy as np

def wrap_to_pi(angle):
    """
    Wraps an angle to the range [-pi, pi].

    Args:
        angle (float or array-like): The angle(s) to be wrapped.

    Returns:
        float or array-like: The wrapped angle(s) in the range [-pi, pi].
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

def rotation_matrix_2d(theta):
    """
    Generate a 2D rotation matrix for a given angle.

    Parameters:
    theta (float): The rotation angle in radians.

    Returns:
    numpy.ndarray: A 2x2 rotation matrix.
    """
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s],
                  [s,  c]])
    return R


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
    point = np.asarray(x, dtype=float)  # Ensure the point is a NumPy array
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
        grad_idx = 1
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

def triangular_wave(t, max_rate=1.0, amplitude=1.0):
    """
    Generate a triangular wave signal for a scalar time t.

    Parameters:
        t (float): Scalar time variable.
        max_rate (float): Maximum absolute slope (rate of change). Must be > 0.
        amplitude (float): Peak amplitude (wave ranges [-amplitude, +amplitude]).

    Returns:
        float: Value of the triangular wave at time t.

    Raises:
        TypeError: If t is not a scalar.
        ValueError: If max_rate is not positive.
    """
    # Ensure scalar input
    if np.ndim(t) != 0:
        raise TypeError("t must be a scalar (float or int)")

    A = float(abs(amplitude))
    if A == 0.0:
        return 0.0

    if max_rate <= 0.0:
        raise ValueError("max_rate must be positive")

    period = 4.0 * A / float(max_rate)

    tt = float(t)
    z = 2.0 * (tt / period - np.floor(tt / period + 0.5))
    result = A * (2.0 * abs(z) - 1.0)

    return float(result)


def smooth_triangular_wave(t, max_rate=1.0, amplitude=1.0, smoothness=0.1, phase=0.0):
    """
    Generate a smoothed triangular wave signal.

    This version keeps the same amplitude and rate-based period definition,
    but replaces sharp |x| corners with a differentiable smooth approximation.

    Parameters:
        t (float or ndarray): Time variable.
        max_rate (float): Maximum absolute slope (rate of change). Must be > 0.
        amplitude (float): Peak amplitude (wave ranges [-amplitude, +amplitude]).
        smoothness (float): Corner rounding factor (>0, smaller = sharper, larger = smoother).
        phase (float): Phase shift as a fraction of the period (-1, 1).

    Returns:
        float or ndarray: Value of the smooth triangular wave at time t.
    """
    A = float(abs(amplitude))
    if A == 0.0:
        return 0.0 if np.ndim(t) == 0 else np.zeros_like(t, dtype=float)
    if max_rate <= 0.0:
        raise ValueError("max_rate must be positive")

    period = 4.0 * A / np.double(max_rate)
    t = t - phase*period 

    # Helper: smooth absolute value
    def smooth_abs(x, A, eps):
        offset = A/2-np.sqrt((A/2)**2 + eps**2)
        return np.sqrt(x**2 + eps**2) + offset
    
    half_period_counter = np.floor(t/(0.5*period))

    # Compute the smoothed triangle core
    y = 0.5*triangular_wave(t, max_rate=max_rate, amplitude=amplitude)  
    if half_period_counter % 2 == 0: # even half-periods
        tri = A - 2 * smooth_abs(y, A, smoothness)  # replace abs(y) with smooth version
    else:  # odd half-periods
        tri = -A + 2 * smooth_abs(y, A, smoothness)  # replace abs(y) with smooth version
    
    result = tri

    if np.ndim(t) == 0:
        return float(result)
    return result
