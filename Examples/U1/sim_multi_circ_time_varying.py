"""
    Simulation of the unicycle system in a multi-circular environment with time-varying obstacles.

    The simulation is performed using the SafeController module and the CBF module. 

    (c) Adrian Wiltz, 2025
    
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
import Controller.SafeController as sc
import Auxiliaries.auxiliary_math as aux_math
from Dynamics.Unicycle import Unicycle
from CBF.CBFmodule import CBFmodule
import CBF.CBF as CBF
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

########################################################################################
# some parameters

cbf_module_filename = "2025-03-22_17-44-16_u1_2_cbfm_1p12.json"

cbf_module_folder_path = r'Examples_paper\U1\Data'

movie_name = "u1_2_1p12_multi-circ-tv_movie"


# simulation paramters
T_sim = 40
dt = 0.02
plotting_steps = int(0.1 // dt)
marker_spacing = int(2 // dt)

t0 = 0
x0 = np.array([-20, 0, 0])
y_d = 0       # desired lateral position of vehicle

v_d = 2     # desired forward speed of vehicle

data_steps = 5  # number of data points to skip for the movie
fps = 1/(dt*data_steps)  # frames per second for the movie

# obstacle positions (distance of obstacles at least 2 x turning radius)
circ_centers = [np.array([-12, 1, 0]),
                    np.array([0, -1, 0]),
                    np.array([12, 1, 0])]
period = 24.0
circ_radii_max = 4.0*np.ones(3)
circ_radii = lambda t: [circ_radii_max[0]*np.abs(np.sin(np.pi*t/period)), circ_radii_max[1]*np.abs(np.sin(np.pi*t/period - 1/3 * np.pi)), circ_radii_max[2]*np.abs(np.sin(np.pi*t/period - 2/3 * np.pi))]

########################################################################################
# load system dynamics and cbf

unicycle_cbf_module = CBFmodule()
unicycle_cbf_module.load(cbf_module_filename, cbf_module_folder_path)

my_unicycle = unicycle_cbf_module.dynamics

########################################################################################
# Setup system and controller for simulation

def alpha(b, c=1, gamma=1):
    """
    Computes the alpha value based on the input parameters b, c, and gamma. 
    
    Parameters:
        b (float): CBF value. 
        c (float, optional): A scaling factor for the positive b cbf values. Default is 1.
        gamma (float, optional): Gamma parameter characterizing the shiftability property of the CBF. Default is 1.
    
    Returns:
        float: The computed alpha value.
    """

    if b >= 0:
        # smoothing of transition between safety and baseline controller
        alpha_val = c * b
    else:
        # According to proof of main theorem
        alpha_val = 2*gamma * (aux_math.sigmoid(c/4 * b) - 1/2)

    return alpha_val

cbf_interpolator = unicycle_cbf_module.cbf.getCbfInterpolator(method='linear')

multi_circ_cbf_grid_points = CBF.computeGridPoints([-20,-10,-np.pi], [20,10,np.pi], [81,41,41])

def multi_circ_cbf(t,x):
    # obstacle specifications
    circ_centers_local = copy.deepcopy(circ_centers)
    
    # trafo of x for each of the obstacle circles
    x_transformed_values = [x - circ_center for circ_center in circ_centers_local]

    # compute the CBF values for each of the obstacle circles
    cbf_values_at_x = [cbf_interpolator(x_transformed_values[i]) - circ_radii(t)[i] for i in range(len(x_transformed_values))]

    return np.nanmin(cbf_values_at_x)

def multi_circ_h(t,x):
    # obstacle specifications
    circ_centers_local = copy.deepcopy(circ_centers)
    
    # trafo of x for each of the obstacle circles
    x_transformed_values = [x - circ_center for circ_center in circ_centers_local]

    # compute the CBF values for each of the obstacle circles
    h_values_at_x = [unicycle_cbf_module.h(x_transformed_values[i]) - circ_radii(t)[i] for i in range(len(x_transformed_values))]

    return np.nanmin(h_values_at_x)

def multi_circ_h_max(x):
    # obstacle specifications
    circ_centers_local = copy.deepcopy(circ_centers)
    
    # trafo of x for each of the obstacle circles
    x_transformed_values = [x - circ_center for circ_center in circ_centers_local]

    # compute the CBF values for each of the obstacle circles
    h_values_at_x = [unicycle_cbf_module.h(x_transformed_values[i]) - circ_radii_max[i] for i in range(len(x_transformed_values))]

    return np.nanmin(h_values_at_x)

controller_settings = {}

controller_settings['cbf_grid_points'] = multi_circ_cbf_grid_points
controller_settings['cbf_function'] = multi_circ_cbf
controller_settings['alpha'] = lambda b: alpha(b, c=0.5, gamma=unicycle_cbf_module.gamma)
controller_settings['alpha_offset'] = 0.1                                               
controller_settings['dynamics'] = my_unicycle
controller_settings['dt'] = 0.5
controller_settings['step_size'] = 1.0



########################################################################################
# Simulation

print("Simulation is running...")

# Create object for saving the baseline controller trajectory
u_baseline_sol = np.zeros((0,my_unicycle.u_dim))

# Set initial time and state in dynamic systen
my_unicycle.t0 = t0
my_unicycle.x0 = x0
my_unicycle.reset()

current_state = x0
for current_time in np.arange(t0, T_sim, dt):
    # Compute the control input
    u_baseline = Unicycle.u_follow_straight_line(current_state,v_d,y_d)
    u_safe = sc.compute_safe_input(controller_settings,current_time,current_state,u_baseline)

    # simulate the system and store the solution automatically in the DynamicSystem object; the current state is updated
    current_state = my_unicycle.simulate(x=current_state, u=u_safe, dt=dt,saveSolution=True)

    # save the baseline controller
    u_baseline_sol = np.vstack([u_baseline_sol,np.array(u_baseline).reshape(1,-1)])

print("Simulation finished.")

########################################################################################
# Visualization
    
# Plot obstacle
x_obstacle_grid = np.linspace(controller_settings['cbf_grid_points'][0][0], controller_settings['cbf_grid_points'][0][-1], 100)
y_obstacle_grid = np.linspace(controller_settings['cbf_grid_points'][1][0], controller_settings['cbf_grid_points'][1][-1], 100)

H_values_start = np.array([[multi_circ_h(0.0,[xi, yi,0]) for xi in x_obstacle_grid] for yi in y_obstacle_grid])
H_values_max = np.array([[multi_circ_h_max([xi, yi, 0]) for xi in x_obstacle_grid] for yi in y_obstacle_grid])

plt.figure(figsize=(6,4))  
plt.contour(x_obstacle_grid, y_obstacle_grid, H_values_start, levels=[0], colors='lightblue', linestyles='--')      # plot the boundary of the obstacle
plt.contour(x_obstacle_grid, y_obstacle_grid, H_values_max, levels=[0], colors='g', linestyles='--')

plt.axhline(0, color='black', linewidth=1, linestyle='--')
plt.axvline(0, color='black', linewidth=1, linestyle='--')
plt.xlim([controller_settings['cbf_grid_points'][0][0], controller_settings['cbf_grid_points'][0][-1]])
plt.ylim([-5, 5])
plt.gca().set_aspect('equal')  # Ensure aspect ratio is equal
plt.xlabel('x')
plt.ylabel('y')

# Plot reference trajectory
plt.axhline(y=y_d, color='r', linestyle='-', label='Desired trajectory')

# Plot trajectory
Unicycle.plot_trajectory_as_triangular_marker(plt, my_unicycle, marker_spacing=marker_spacing, line_color='k', linewidth=2, marker_shape=(0.75,0.4))

plt.show(block=False)

# Plot difference in the steering angles between the baseline and the safe controller and CBF values along the trajectory
cbf_values_along_trajectory = np.array([multi_circ_cbf(my_unicycle.t_sol[i],my_unicycle.x_sol[i]) for i in range(len(my_unicycle.x_sol))])

distance_to_obstacle_along_trajectory = np.array([multi_circ_h(my_unicycle.t_sol[i], my_unicycle.x_sol[i,:]) for i in range(len(my_unicycle.x_sol))])

valid_indices = [i for i, dist in enumerate(distance_to_obstacle_along_trajectory) if dist is not None and not np.isnan(dist)]
t_filtered = my_unicycle.t_sol[valid_indices]
distance_to_obstacle_along_trajectory_filtered = distance_to_obstacle_along_trajectory[valid_indices]

fig, ax = plt.subplots(figsize=(6,2))

ax.plot(my_unicycle.t_sol, cbf_values_along_trajectory, label='CBF value', color='k')
ax.plot(t_filtered, distance_to_obstacle_along_trajectory_filtered, label='Distance to obstacle', color='k', linestyle='--')
ax.legend()
ax.grid(True)
ax.set_xlim([0, 30])
ax.set_ylim([-1,10])
ax.set_xlabel('Time [s]')
ax.set_ylabel('CBF value')
plt.show(block=False)

########################################################################################

input("Press Enter to continue...")

########################################################################################
# Make a movie

print("Creating movie...")

import imageio.v2 as imageio

frames_dir = "frames_u124"
os.makedirs(frames_dir, exist_ok=True)

frames = []
for i in tqdm(range(0,len(my_unicycle.t_sol),data_steps), desc="Creating frames", unit="frame"):
    fig, ax = plt.subplots(figsize=(12,6))
    
    ax.set_xlim([x0[0], controller_settings['cbf_grid_points'][0][-1]+5])
    ax.set_ylim([-6, 6])
    ax.set_aspect('equal')

    # Plot obstacle
    H_values_tv = np.array([[multi_circ_h(my_unicycle.t_sol[i],[xi, yi,0]) for xi in x_obstacle_grid] for yi in y_obstacle_grid])


    ax.contourf(x_obstacle_grid, y_obstacle_grid, H_values_tv, levels=[-np.inf,0], colors='gray', alpha=0.5)  # plot the obstacle
    ax.contour(x_obstacle_grid, y_obstacle_grid, H_values_tv, levels=[0], colors='k') 

    # Plot reference trajectory
    ax.axhline(y=y_d, color='r', linestyle='-')

    # Plot vehicle at current time step
    ax.plot(my_unicycle.x_sol[:i,0], my_unicycle.x_sol[:i,1], 'b-', linewidth=1)  # Draw trajectory until current time step
    Unicycle.plot_trajectory_at_indices_as_triangular_marker(plt, my_unicycle, [i])

    # Add current time to the upper right corner
    ax.text(0.95, 0.95, f"t = {my_unicycle.t_sol[i]:.2f}", transform=ax.transAxes, ha='right', va='top')

    # Save the current frame
    frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
    plt.savefig(frame_path)
    plt.close(fig)
    frames.append(frame_path)

# Create a movie
output_gif = movie_name + ".gif"
with imageio.get_writer(output_gif, mode='I', fps=fps) as writer:
    for frame in frames:
        image = imageio.imread(frame)
        writer.append_data(image)

# Create an MP4 video
output_mp4 = movie_name + ".mp4"
imageio.mimsave(output_mp4, [imageio.imread(f) for f in frames], format='FFMPEG', fps=fps)

print(f"Movie saved as {output_gif} and {output_mp4}.")

input("Press Enter to close...")

    



