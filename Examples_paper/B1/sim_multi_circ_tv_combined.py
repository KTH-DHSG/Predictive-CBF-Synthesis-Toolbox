"""

    Simulation of the less agile and more agile system in a multi-circular environment with time-varying obstacles.

    The simulation is performed using the SafeController module and the CBF module. 

    (c) Adrian Wiltz, 2025

"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
import Controller.SafeController as sc
import Auxiliaries.auxiliary_math as aux_math
from Dynamics.Bicycle import Bicycle
from CBF.CBFmodule import CBFmodule
import CBF.CBF as CBF
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

########################################################################################
# Specify the system dynamics, the CBF computation parameters, and initialize the CBF module

# some parameters

cbf_module_filename_2p8 = "2025-03-22_00-17-20_b1_1_cbfm_2p8.json"
cbf_module_filename_1p12 = "2025-03-22_16-06-04_b1_2_cbfm_1p12.json"

cbf_module_folder_path = r'Examples_paper\B1\Data'

movie_name = "b1_combined_multi-circ-tv_movie"

# simulation paramters
T_sim = 25
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
period = 23
max_radii = np.array([4.0, 4.0, 4.0])
circ_radii = lambda t: [max_radii[0]*np.abs(np.sin(np.pi*t/period)), max_radii[1]*np.abs(np.sin(np.pi*t/period - 1/3 * np.pi)), max_radii[2]*np.abs(np.sin(np.pi*t/period - 2/3 * np.pi))]

########################################################################################
# load system dynamics and cbf

# load less agile system
bicycle_cbf_module_2p8 = CBFmodule()
bicycle_cbf_module_2p8.load(cbf_module_filename_2p8, cbf_module_folder_path)

my_bicycle_2p8 = bicycle_cbf_module_2p8.dynamics

# load more agile system
bicycle_cbf_module_1p12 = CBFmodule()
bicycle_cbf_module_1p12.load(cbf_module_filename_1p12, cbf_module_folder_path)

my_bicycle_1p12 = bicycle_cbf_module_1p12.dynamics

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

cbf_interpolator_2p8 = bicycle_cbf_module_2p8.cbf.getCbfInterpolator(method='linear')
cbf_interpolator_1p12 = bicycle_cbf_module_1p12.cbf.getCbfInterpolator(method='linear')

multi_circ_cbf_2p8_grid_points = CBF.computeGridPoints([-20,-10,-np.pi], [20,10,np.pi], [81,41,41])
multi_circ_cbf_1p12_grid_points = CBF.computeGridPoints([-20,-10,-np.pi], [20,10,np.pi], [81,41,41])

def multi_circ_cbf_2p8(t,x):
    # obstacle specifications
    circ_centers_local = copy.deepcopy(circ_centers)
    
    # trafo of x for each of the obstacle circles
    x_transformed_values = [x - circ_center for circ_center in circ_centers_local]

    # compute the CBF values for each of the obstacle circles
    cbf_values_at_x = [cbf_interpolator_2p8(x_transformed_values[i]) - circ_radii(t)[i] for i in range(len(x_transformed_values))]

    return np.nanmin(cbf_values_at_x)

def multi_circ_cbf_1p12(t,x):
    # obstacle specifications
    circ_centers_local = copy.deepcopy(circ_centers)
    
    # trafo of x for each of the obstacle circles
    x_transformed_values = [x - circ_center for circ_center in circ_centers_local]

    # compute the CBF values for each of the obstacle circles
    cbf_values_at_x = [cbf_interpolator_1p12(x_transformed_values[i]) - circ_radii(t)[i] for i in range(len(x_transformed_values))]

    return np.nanmin(cbf_values_at_x)

# identical for both systems
def multi_circ_h(t,x):
    # obstacle specifications
    circ_centers_local = copy.deepcopy(circ_centers)
    
    # trafo of x for each of the obstacle circles
    x_transformed_values = [x - circ_center for circ_center in circ_centers_local]

    # compute the CBF values for each of the obstacle circles
    h_values_at_x = [bicycle_cbf_module_2p8.h(x_transformed_values[i]) - circ_radii(t)[i] for i in range(len(x_transformed_values))]

    return np.nanmin(h_values_at_x)

def multi_circ_h_max(x):
    # obstacle specifications
    circ_centers_local = copy.deepcopy(circ_centers)
    
    # trafo of x for each of the obstacle circles
    x_transformed_values = [x - circ_center for circ_center in circ_centers_local]

    # compute the CBF values for each of the obstacle circles
    h_values_at_x = [bicycle_cbf_module_2p8.h(x_transformed_values[i]) - max_radii[i] for i in range(len(x_transformed_values))]

    return np.nanmin(h_values_at_x)

# controller settings for less agile system
controller_settings_2p8 = {}

controller_settings_2p8['cbf_grid_points'] = multi_circ_cbf_2p8_grid_points
controller_settings_2p8['cbf_function'] = multi_circ_cbf_2p8
controller_settings_2p8['alpha'] = lambda b: alpha(b, c=1.0, gamma=bicycle_cbf_module_2p8.gamma)
controller_settings_2p8['alpha_offset'] = 0.0                                               
controller_settings_2p8['dynamics'] = my_bicycle_2p8
controller_settings_2p8['dt'] = 0.5
controller_settings_2p8['step_size'] = 0.5

# controller settings for more agile system
controller_settings_1p12 = {}

controller_settings_1p12['cbf_grid_points'] = multi_circ_cbf_1p12_grid_points
controller_settings_1p12['cbf_function'] = multi_circ_cbf_1p12
controller_settings_1p12['alpha'] = lambda b: alpha(b, c=1.0, gamma=bicycle_cbf_module_1p12.gamma)
controller_settings_1p12['alpha_offset'] = 0.0                              
controller_settings_1p12['dynamics'] = my_bicycle_1p12
controller_settings_1p12['dt'] = 0.5
controller_settings_1p12['step_size'] = 0.5

########################################################################################
# Simulation

print("Simulation is running...")

# simulate less agile system
print("Simulation of less agile system started.")

# Create object for saving the baseline controller trajectory
u_baseline_sol_2p8 = np.zeros((0,my_bicycle_2p8.u_dim))

# Set initial time and state in dynamic systen
my_bicycle_2p8.t0 = t0
my_bicycle_2p8.x0 = x0
my_bicycle_2p8.reset()

current_state = x0
for current_time in np.arange(t0, T_sim, dt):
    # Compute the control input
    u_baseline = Bicycle.u_follow_straight_line(current_state,v_d,y_d)
    u_safe = sc.compute_safe_input(controller_settings_2p8,current_time,current_state,u_baseline)

    # simulate the system and store the solution automatically in the DynamicSystem object; the current state is updated
    current_state = my_bicycle_2p8.simulate(x=current_state, u=u_safe, dt=dt,saveSolution=True)

    # save the baseline controller
    u_baseline_sol_2p8 = np.vstack([u_baseline_sol_2p8,np.array(u_baseline).reshape(1,-1)])

print("Simulation of less agile system finished.")

# simulate more agile system
print("Simulation of more agile system started.")

# Create object for saving the baseline controller trajectory
u_baseline_sol_1p12 = np.zeros((0,my_bicycle_1p12.u_dim))

# Set initial time and state in dynamic systen
my_bicycle_1p12.t0 = t0
my_bicycle_1p12.x0 = x0
my_bicycle_1p12.reset()

current_state = x0
for current_time in np.arange(t0, T_sim, dt):
    # Compute the control input
    u_baseline = Bicycle.u_follow_straight_line(current_state,v_d,y_d)
    u_safe = sc.compute_safe_input(controller_settings_1p12,current_time,current_state,u_baseline)

    # simulate the system and store the solution automatically in the DynamicSystem object; the current state is updated
    current_state = my_bicycle_1p12.simulate(x=current_state, u=u_safe, dt=dt,saveSolution=True)

    # save the baseline controller
    u_baseline_sol_1p12 = np.vstack([u_baseline_sol_1p12,np.array(u_baseline).reshape(1,-1)])

print("Simulation of more agile system finished.")

print("Simulation finished.")

########################################################################################
# Visualization

# Plot obstacle
x_obstacle_grid = np.linspace(x0[0], -x0[0], 100)
y_obstacle_grid = np.linspace(controller_settings_2p8['cbf_grid_points'][1][0], controller_settings_2p8['cbf_grid_points'][1][-1], 100)

H_values_start = np.array([[multi_circ_h(0.0,[xi, yi,0]) for xi in x_obstacle_grid] for yi in y_obstacle_grid])
H_values_max = np.array([[multi_circ_h_max([xi, yi, 0]) for xi in x_obstacle_grid] for yi in y_obstacle_grid])

plt.figure(figsize=(6,4))  
plt.contour(x_obstacle_grid, y_obstacle_grid, H_values_start, levels=[0], colors='lightblue', linestyles='--')      # plot the boundary of the obstacle
plt.contour(x_obstacle_grid, y_obstacle_grid, H_values_max, levels=[0], colors='g', linestyles='--')

# Plot reference trajectory
plt.axhline(y=y_d, color='r', linestyle='-', label="_nolegend_")

# Plot trajectory
Bicycle.plot_trajectory_with_bicycle_markers(plt, my_bicycle_2p8, plotting_steps, marker_spacing=marker_spacing, label='less agile system', line_color='blue', linewidth=1, marker_size=2)
Bicycle.plot_trajectory_with_bicycle_markers(plt, my_bicycle_1p12, plotting_steps, marker_spacing=marker_spacing, label='more agile system', line_color='green', linewidth=1, marker_size=2)

# plt.legend()
plt.xlim([x0[0], -x0[0]])
plt.ylim([-5, 5])
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal', adjustable='box')

plt.show(block=False)

# Plot difference in the steering angles between the baseline and the safe controller and CBF values along the trajectory
cbf_values_along_trajectory_2p8 = np.array([multi_circ_cbf_2p8(my_bicycle_2p8.t_sol[i],my_bicycle_2p8.x_sol[i]) for i in range(len(my_bicycle_2p8.x_sol))])
cbf_values_along_trajectory_1p12 = np.array([multi_circ_cbf_1p12(my_bicycle_1p12.t_sol[i],my_bicycle_1p12.x_sol[i]) for i in range(len(my_bicycle_1p12.x_sol))])

distance_to_obstacle_along_trajectory_2p8 = np.array([multi_circ_h(my_bicycle_2p8.t_sol[i], my_bicycle_2p8.x_sol[i,:]) for i in range(len(my_bicycle_2p8.x_sol))])
distance_to_obstacle_along_trajectory_1p12 = np.array([multi_circ_h(my_bicycle_1p12.t_sol[i], my_bicycle_1p12.x_sol[i,:]) for i in range(len(my_bicycle_1p12.x_sol))])

valid_indices_2p8 = [i for i, dist in enumerate(distance_to_obstacle_along_trajectory_2p8) if dist is not None and not np.isnan(dist)]
valid_indices_1p12 = [i for i, dist in enumerate(distance_to_obstacle_along_trajectory_1p12) if dist is not None and not np.isnan(dist)]
t_filtered_2p8 = my_bicycle_2p8.t_sol[valid_indices_2p8]
t_filtered_1p12 = my_bicycle_1p12.t_sol[valid_indices_1p12]
distance_to_obstacle_along_trajectory_filtered_2p8 = distance_to_obstacle_along_trajectory_2p8[valid_indices_2p8]
distance_to_obstacle_along_trajectory_filtered_1p12 = distance_to_obstacle_along_trajectory_1p12[valid_indices_1p12]

fig, ax = plt.subplots(3,1,figsize=(6,4))

ax[0].plot(my_bicycle_2p8.t_sol[:-1], u_baseline_sol_2p8[:,1] - my_bicycle_2p8.u_sol[:,1], color='blue', label='less agile system')
ax[0].plot(my_bicycle_1p12.t_sol[:-1], u_baseline_sol_1p12[:,1] - my_bicycle_1p12.u_sol[:,1], color='green', label='more agile system')
# ax[0].legend()
ax[0].grid(True)
ax[0].set_ylabel(r'$\zeta_{baseline} - \zeta_{safe}$')
ax[0].set_xticklabels([])
ax[0].set_xlim([0,25])

ax[1].plot(my_bicycle_2p8.t_sol, cbf_values_along_trajectory_2p8, color='blue', label='less agile system')
ax[1].plot(my_bicycle_1p12.t_sol, cbf_values_along_trajectory_1p12, color='green', label='more agile system')
ax[1].grid(True)
ax[1].set_ylabel('CBF value')
ax[1].set_xticklabels([])
ax[1].set_xlim([0,25])
# ax[1].legend()

# Plot distance to obstacle
ax[2].plot(t_filtered_2p8, distance_to_obstacle_along_trajectory_filtered_2p8, color='blue', label='less agile system')
ax[2].plot(t_filtered_1p12, distance_to_obstacle_along_trajectory_filtered_1p12, color='green', label='more agile system')
ax[2].grid(True)
ax[2].set_ylabel('Distance')
ax[2].set_xlim([0,25])
ax[2].set_xlabel('Time [s]')
# ax[2].legend()

# Plot inputs
plt.figure(figsize=(6,2))
# Plot velocity in subplot with limits
plt.subplot(2,1,1)
plt.plot(my_bicycle_2p8.t_sol[:-1], my_bicycle_2p8.u_sol[:,0], label=r"less agile system", color='blue')
plt.plot(my_bicycle_1p12.t_sol[:-1], my_bicycle_1p12.u_sol[:,0], label=r"more agile system", color='green')
plt.axhline(my_bicycle_1p12.u_max[0], color='k', linestyle='--')
plt.axhline(my_bicycle_1p12.u_min[0], color='k', linestyle='--')
# plt.legend()
plt.grid(True)
plt.gca().set_xticklabels([])
plt.ylabel(r'$v$')
plt.xlim([0,25])

# Plot steering angle in subplot with limits
plt.subplot(2,1,2)
plt.plot(my_bicycle_2p8.t_sol[:-1], my_bicycle_2p8.u_sol[:,1], label=r"less agile system", color='blue')
plt.plot(my_bicycle_1p12.t_sol[:-1], my_bicycle_1p12.u_sol[:,1], label=r"more agile system", color='green')
plt.axhline(my_bicycle_2p8.u_min[1], color='blue', linestyle='--')
plt.axhline(my_bicycle_2p8.u_max[1], color='blue', linestyle='--')
plt.axhline(my_bicycle_1p12.u_min[1], color='green', linestyle='--')
plt.axhline(my_bicycle_1p12.u_max[1], color='green', linestyle='--')
# plt.legend()
plt.grid(True)
plt.xlim([0,25])
plt.xlabel('Time [s]')
plt.ylabel(r'$\zeta$')
plt.show(block=False)

########################################################################################

input("Press Enter to continue...")

########################################################################################
# Make a movie

print("Creating movie...")

import imageio.v2 as imageio

frames_dir = "frames_b114"
os.makedirs(frames_dir, exist_ok=True)

frames = []
for i in tqdm(range(0,len(my_bicycle_2p8.t_sol),data_steps), desc="Creating frames", unit="frame"):
    fig, ax = plt.subplots(figsize=(12,6))
    
    ax.set_xlim([x0[0], -x0[0]])
    ax.set_ylim([-5, 5])
    ax.set_aspect('equal')

    # Plot obstacle
    H_values_tv = np.array([[multi_circ_h(my_bicycle_2p8.t_sol[i],[xi, yi,0]) for xi in x_obstacle_grid] for yi in y_obstacle_grid])

    ax.contourf(x_obstacle_grid, y_obstacle_grid, H_values_tv, levels=[-np.inf,0], colors='gray', alpha=0.5)  # plot the obstacle
    ax.contour(x_obstacle_grid, y_obstacle_grid, H_values_tv, levels=[0], colors='k') 

    # Plot reference trajectory
    ax.axhline(y=y_d, color='r', linestyle='-')

    # Plot vehicle at current time step
    ax.plot(my_bicycle_2p8.x_sol[:i,0], my_bicycle_2p8.x_sol[:i,1], 'b-', linewidth=1,label='less agile system')  # Draw trajectory until current time step
    ax.plot(my_bicycle_1p12.x_sol[:i,0], my_bicycle_1p12.x_sol[:i,1], 'g-', linewidth=1, label='more agile system')  # Draw trajectory until current time step
    Bicycle.plot_trajectory_at_indices_with_bicycle_markers(plt, my_bicycle_2p8, [i], marker_size=1, dot_size=1)
    Bicycle.plot_trajectory_at_indices_with_bicycle_markers(plt, my_bicycle_1p12, [i], marker_size=1, dot_size=1)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.1), ncol=2)

    # Add current time to the upper right corner
    ax.text(0.95, 0.95, f"t = {my_bicycle_2p8.t_sol[i]:.2f}", transform=ax.transAxes, ha='right', va='top')

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

    



