"""

    Simulation example with time-varying circular obstacles.

    (c) Adrian Wiltz, 2025
    
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
import Controller.SafeController as sc
import Auxiliaries.auxiliary_math as aux_math
import Auxiliaries.auxiliary as aux
from Dynamics.Bicycle import Bicycle
from CBF.CBFmodule import CBFmodule
import CBF.CBF as CBF
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from tqdm import tqdm
import copy

########################################################################################
# some parameters

cbf_module_filename = "2025-11-11_01-30-44_b_cbfm_1p12.json"

cbf_module_folder_path = r'Examples_paper\B1\Data'

movie_name = "b1_multi-circ-tv-cpg_movie"


# simulation paramters
T_sim = 50
dt = 0.02
plotting_steps = int(0.1 // dt)
marker_spacing = int(2 // dt)

t0 = 0
x0_array = [np.array([-30.0, 4.0, 0.0]), np.array([-30.0, -10.0, 0.0]), np.array([-30.0, -30.0, 0.0]), np.array([-15.0, -30.0, 0.0]), np.array([0.0,-30.0,0.0])]
y_d = [4.0, 4.0, 4.0, 4.0, 4.0]       # desired lateral position of vehicle

# Compute orientation towards the desired trajectory
for i in range(len(x0_array)):
    if x0_array[i][0] == 0.0:
        x0_array[i][2] = np.pi/2 if y_d[i] > x0_array[i][1] else -np.pi/2
    else:
        x0_array[i][2] = np.arctan2(y_d[i]-x0_array[i][1], 0.0 - x0_array[i][0])

v_d = 1.9     # desired forward speed of vehicle

data_steps = 5  # number of data points to skip for the movie
fps = 1/(dt*data_steps)  # frames per second for the movie

# obstacle positions (distance of obstacles at least 2 x turning radius)
d = 4.0
circ_centers = [np.array([-2*d, -4*d, 0]), np.array([2*d, -4*d, 0]), 
                np.array([-4*d, -3*d, 0]), np.array([0, -3*d, 0]), np.array([4*d, -3*d, 0]), 
                np.array([-2*d, -2*d, 0]), np.array([2*d, -2*d, 0]), 
                np.array([-4*d, -d, 0]), np.array([0, -d, 0]), np.array([4*d, -d, 0]),
                np.array([-2*d, 0, 0]), np.array([2*d, 0, 0]),  
                np.array([-4*d, d, 0]), np.array([0, d, 0]), np.array([4*d, d, 0]),
                np.array([-2*d, 2*d, 0]), np.array([2*d, 2*d, 0]), 
                np.array([-4*d, 3*d, 0]), np.array([0, 3*d, 0]), np.array([4*d, 3*d, 0]), 
                np.array([-2*d, 4*d, 0]), np.array([2*d, 4*d, 0])]
r1 = lambda t: 3.0 + aux_math.smooth_triangular_wave(t, max_rate=1.0, amplitude=3.0, smoothness=0.2,phase=0.75)
r2 = lambda t: 3.0 + aux_math.smooth_triangular_wave(t, max_rate=1.0, amplitude=3.0, smoothness=0.2,phase=0.25)
r3 = lambda t: 1.5 + aux_math.smooth_triangular_wave(t, max_rate=1.0, amplitude=1.5, smoothness=0.2,phase=0.25)

r1_max = 3.0 + aux_math.smooth_triangular_wave(0, max_rate=1.0, amplitude=3.0, smoothness=0.2,phase=-0.25)
r1_min = 3.0 + aux_math.smooth_triangular_wave(0, max_rate=1.0, amplitude=3.0, smoothness=0.2,phase=0.25)
r2_max = r1_max
r2_min = r1_min
r3_max = 1.5 + aux_math.smooth_triangular_wave(0, max_rate=1.0, amplitude=1.5, smoothness=0.2,phase=-0.25)
r3_min = 1.5 + aux_math.smooth_triangular_wave(0, max_rate=1.0, amplitude=1.5, smoothness=0.2,phase=0.25)

circ_radii = lambda t: [r2(t), r2(t),
                        r1(t), r1(t), r1(t),
                        r3(t), r3(t),
                        r2(t), r3(t), r2(t),
                        r1(t), r1(t),
                        r3(t), r2(t), r3(t),
                        r3(t), r3(t), 
                        r1(t), r3(t), r1(t),
                        r2(t), r2(t)]

min_circ_radii = [r2_min, r2_min,
                   r1_min, r1_min, r1_min,
                   r3_min, r3_min,
                   r2_min, r3_min, r2_min,
                   r1_min, r1_min,
                   r3_min, r2_min, r3_min,
                   r3_min, r3_min,
                   r1_min, r3_min, r1_min,
                   r2_min, r2_min]
max_circ_radii = [r2_max, r2_max,
                   r1_max, r1_max, r1_max,
                   r3_max, r3_max,
                   r2_max, r3_max, r2_max,
                   r1_max, r1_max,
                   r3_max, r2_max, r3_max,
                   r3_max, r3_max,
                   r1_max, r3_max, r1_max,
                   r2_max, r2_max]

linestyles = [':', '--', '-']

lines_radii = [linestyles[1], linestyles[1],
                linestyles[0], linestyles[0], linestyles[0],
                linestyles[2], linestyles[2],
                linestyles[1], linestyles[2], linestyles[1],
                linestyles[0], linestyles[0],
                linestyles[2], linestyles[1], linestyles[2],
                linestyles[2], linestyles[2],
                linestyles[0], linestyles[2], linestyles[0],
                linestyles[1], linestyles[1]]

########################################################################################
# load system dynamics and cbf

bicycle_cbf_module = CBFmodule()
bicycle_cbf_module.load(cbf_module_filename, cbf_module_folder_path)

my_bicycle = bicycle_cbf_module.dynamics


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

cbf_interpolator = bicycle_cbf_module.cbf.getCbfInterpolator(method='linear')

multi_circ_cbf_grid_points = CBF.computeGridPoints([-30,-30,-np.pi], [30,30,np.pi], [81,81,41])

def multi_circ_cbf(t,x):
    # obstacle specifications
    circ_centers_local = copy.deepcopy(circ_centers)

    # Consider only circ centers with distance less than distance dist_max to x[0:2]
    dist_max = 10.0
    max_val = 2.0
    circ_centers_local = [circ_center for circ_center in circ_centers_local if np.linalg.norm(x[0:2]-circ_center[0:2]) <= dist_max]
    circ_radii_local = [circ_radii(t)[i] for i in range(len(circ_centers)) if np.linalg.norm(x[0:2]-circ_centers[i][0:2]) <= dist_max]

    # If no obstacle is close, return a large positive value
    if len(circ_centers_local) == 0:
        return max_val
    
    # trafo of x for each of the obstacle circles
    x_transformed_values = [x - circ_center for circ_center in circ_centers_local]

    # compute the CBF values for each of the obstacle circles
    cbf_values_at_x = [cbf_interpolator(x_transformed_values[i]) - circ_radii_local[i] for i in range(len(x_transformed_values))]

    return min(np.nanmin(cbf_values_at_x), max_val)

def multi_circ_h(t,x):
    # obstacle specifications
    circ_centers_local = copy.deepcopy(circ_centers)

    # Consider only circ centers with distance less than distance dist_max to x[0:2]
    dist_max = 10.0
    max_val = 6.0
    circ_centers_local = [circ_center for circ_center in circ_centers_local if np.linalg.norm(x[0:2]-circ_center[0:2]) <= dist_max]
    circ_radii_local = [circ_radii(t)[i] for i in range(len(circ_centers)) if np.linalg.norm(x[0:2]-circ_centers[i][0:2]) <= dist_max]
    
    # If no obstacle is close, return a large positive value
    if len(circ_centers_local) == 0:
        return max_val

    # trafo of x for each of the obstacle circles
    x_transformed_values = [x - circ_center for circ_center in circ_centers_local]

    # compute the CBF values for each of the obstacle circles
    h_values_at_x = [bicycle_cbf_module.h(x_transformed_values[i]) - circ_radii_local[i] for i in range(len(x_transformed_values))]

    return min(np.nanmin(h_values_at_x), max_val)

controller_settings = {}

controller_settings['cbf_grid_points'] = multi_circ_cbf_grid_points
controller_settings['cbf_function'] = multi_circ_cbf
controller_settings['alpha'] = lambda b: alpha(b, c=1.0, gamma=bicycle_cbf_module.gamma)
controller_settings['alpha_offset'] = 0.1                                               
controller_settings['dynamics'] = my_bicycle
controller_settings['dt'] = 0.01
controller_settings['step_size'] = 0.01


########################################################################################
# Simulation

print("Simulation is running...")

print("Start simulation of more agile system...")

# Create object for saving the baseline controller trajectory
u_baseline_sol_array = [np.zeros((0,my_bicycle.u_dim)) for _ in range(len(x0_array))]

my_bicycle_array = [copy.deepcopy(my_bicycle) for _ in range(len(x0_array))]

# Set initial time and state in dynamic systen
for i in range(len(x0_array)):
    my_bicycle_array[i].t0 = t0
    my_bicycle_array[i].x0 = x0_array[i]
    my_bicycle_array[i].reset()

current_state_array = copy.deepcopy(x0_array)
n_steps = int(np.ceil((T_sim - t0) / dt))
for step in tqdm(range(n_steps), desc="Simulating", unit="step"):
    current_time = t0 + step * dt

    # Compute the control input
    u_baseline_array = [Bicycle.u_follow_straight_line(x=current_state_array[i], v_d=v_d, y_d=y_d[i], rot=x0_array[i][2]) for i in range(len(x0_array))]
    u_safe_array = [sc.compute_safe_input(controller_settings, current_time, current_state_array[i], u_baseline_array[i]) for i in range(len(x0_array))]

    # simulate the system and store the solution automatically in the DynamicSystem object; the current state is updated
    for i in range(len(x0_array)):
        current_state_array[i] = my_bicycle_array[i].simulate(x=current_state_array[i], u=u_safe_array[i], dt=dt, saveSolution=True)

    # save the baseline controller
    u_baseline_sol_array = [np.vstack([u_baseline_sol_array[i], np.array(u_baseline_array[i]).reshape(1, -1)]) for i in range(len(x0_array))]

print("Simulation of more agile system finished.")

print("Simulation finished.")

########################################################################################
# Visualization
    
# Plot obstacle
x_obstacle_grid = np.linspace(controller_settings['cbf_grid_points'][0][0], controller_settings['cbf_grid_points'][0][-1], 100)
y_obstacle_grid = np.linspace(controller_settings['cbf_grid_points'][1][0], controller_settings['cbf_grid_points'][1][-1], 100)

# Create a list with the first len(x0_array) colors from the default matplotlib color cycle
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(x0_array)]
color_cycle_light = [aux.lighten(color, amount=0.5) for color in color_cycle]
color_cycle_dark = [aux.darken(color, amount=0.5) for color in color_cycle]

# plot circ radii over time
t_values = np.linspace(0, 50, 300)
radii_values_1 = np.array([r1(t) for t in t_values])
radii_values_2 = np.array([r2(t) for t in t_values])
radii_values_3 = np.array([r3(t) for t in t_values])
plt.figure(figsize=(2,1.5))
plt.plot(t_values, radii_values_1, color='black', linestyle=linestyles[0])
plt.plot(t_values, radii_values_2, color='black', linestyle=linestyles[1])
plt.plot(t_values, radii_values_3, color='black', linestyle=linestyles[2])
plt.xlabel('Time [s]')
plt.ylabel('Obstacle Radius')
plt.title('Time-Varying Obstacle Radii')
plt.grid(True)
plt.show(block=False)

plt.figure(figsize=(6,4))  
ax = plt.gca()
# plot circles at circ centers with respective r_max
for i, center in enumerate(circ_centers):
    radius = max_circ_radii[i]
    circ = Circle((center[0], center[1]), radius, fill=False, edgecolor='gray', linewidth=1, linestyle=lines_radii[i], alpha=0.9)
    ax.plot(center[0], center[1], 'ko', markersize=3)
    ax.add_patch(circ)

plt.xlim([controller_settings['cbf_grid_points'][0][0], controller_settings['cbf_grid_points'][0][-1]])
plt.ylim([-30, 30])
plt.gca().set_aspect('equal')  # Ensure aspect ratio is equal
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)

# Reference trajectories
for i in range(len(x0_array)):
    ax.axline(x0_array[i][0:2], [0.0, y_d[i]], color=color_cycle[i], linestyle='--', linewidth=1)

# Plot trajectory
for i in range(len(x0_array)):
    my_bicycle = my_bicycle_array[i]
    Bicycle.plot_trajectory_as_triangular_marker(plt, my_bicycle, marker_spacing=marker_spacing, line_color=color_cycle[i], linewidth=2, marker_shape=(1.5,0.8))

plt.show(block=False)

# Plot difference in the steering angles between the baseline and the safe controller and CBF values along the trajectory
cbf_values_along_trajectory_array = [np.array([multi_circ_cbf(my_bicycle.t_sol[i],my_bicycle.x_sol[i]) for i in range(len(my_bicycle.x_sol))]) for my_bicycle in my_bicycle_array]

distance_to_obstacle_along_trajectory_array = [np.array([multi_circ_h(my_bicycle.t_sol[i], my_bicycle.x_sol[i,:]) for i in range(len(my_bicycle.x_sol))]) for my_bicycle in my_bicycle_array]

valid_indices_array = [[i for i, dist in enumerate(distance_to_obstacle_along_trajectory) if dist is not None and not np.isnan(dist)] for distance_to_obstacle_along_trajectory in distance_to_obstacle_along_trajectory_array]
t_filtered_array = [my_bicycle.t_sol[valid_indices] for my_bicycle, valid_indices in zip(my_bicycle_array, valid_indices_array)]
distance_to_obstacle_along_trajectory_filtered_array = [distance_to_obstacle_along_trajectory[valid_indices] for distance_to_obstacle_along_trajectory, valid_indices in zip(distance_to_obstacle_along_trajectory_array, valid_indices_array)]

fig, ax = plt.subplots(figsize=(6,2))

for i in range(len(x0_array)):
    my_bicycle = my_bicycle_array[i]
    cbf_values_along_trajectory = cbf_values_along_trajectory_array[i]
    distance_to_obstacle_along_trajectory = distance_to_obstacle_along_trajectory_array[i]
    t_filtered = t_filtered_array[i]
    distance_to_obstacle_along_trajectory_filtered = distance_to_obstacle_along_trajectory_filtered_array[i]
    ax.plot(my_bicycle.t_sol, cbf_values_along_trajectory, label='CBF value', color=color_cycle[i])
    ax.plot(t_filtered, distance_to_obstacle_along_trajectory_filtered, label='Distance to obstacle', linestyle='--', color=color_cycle[i])
ax.grid(True)
ax.set_xlim([0, 40])
ax.set_ylim([-1,4.0])
ax.set_xlabel('Time [s]')
ax.set_ylabel('CBF value')
plt.show(block=False)

########################################################################################

input("Press Enter to continue...")

########################################################################################
# Make a movie

print("Creating movie...")

import imageio.v2 as imageio

frames_dir = "frames_b1_tv_cpg"
os.makedirs(frames_dir, exist_ok=True)

frames = []
for i in tqdm(range(0,len(my_bicycle.t_sol),data_steps), desc="Creating frames", unit="frame"):
    fig, ax = plt.subplots(figsize=(12,6))
    
    ax.set_xlim([controller_settings['cbf_grid_points'][0][0], controller_settings['cbf_grid_points'][0][-1]])
    ax.set_ylim([-30, 30])
    ax.set_aspect('equal')

    H_values_tv = np.array([[multi_circ_h(my_bicycle.t_sol[i],[xi, yi,0]) for xi in x_obstacle_grid] for yi in y_obstacle_grid])

    ax.contourf(x_obstacle_grid, y_obstacle_grid, H_values_tv, levels=[-np.inf,0], colors='gray', alpha=0.5)  # plot the obstacle
    ax.contour(x_obstacle_grid, y_obstacle_grid, H_values_tv, levels=[0], colors='k') 

    # Plot reference trajectories
    for k in range(len(x0_array)):
        ax.axline(x0_array[k][0:2], [0.0, y_d[k]], color=color_cycle_light[k], linestyle='--', linewidth=1)

    # Plot vehicle at current time step
    for k in range(len(x0_array)):
        my_bicycle = my_bicycle_array[k]
        ax.plot(my_bicycle.x_sol[:i,0], my_bicycle.x_sol[:i,1], color=color_cycle_dark[k], linewidth=2)  # Draw trajectory until current time step
        Bicycle.plot_trajectory_at_indices_with_bicycle_markers(plt, my_bicycle, [i], marker_size=2, dot_size=2)

    # Add current time to the upper right corner
    ax.text(0.95, 0.95, f"t = {my_bicycle.t_sol[i]:.2f}", transform=ax.transAxes, ha='right', va='top')

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

    



