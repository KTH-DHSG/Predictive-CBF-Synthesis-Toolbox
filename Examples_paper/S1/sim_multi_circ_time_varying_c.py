"""

    Simulation of single integrator 1 and 2 in a multi-circular environment with time-varying obstacles.

    The simulation is performed using the SafeController module and the CBF module. 

    (c) Adrian Wiltz, 2025
    
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
import Controller.SafeController as sc
import Auxiliaries.auxiliary_math as aux_math
from Dynamics.SingleIntegrator import SingleIntegrator
from CBF.CBFmodule import CBFmodule
import CBF.CBF as CBF
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

########################################################################################
# some parameters

# Load the CBF module for the single integrator with v_x in [-2,2]
cbf_module_filename_1 = "2025-03-22_19-13-02_s1_cbfm.json"
# Load the CBF module for the single integrator with v_x in [1,2]
cbf_module_filename_2 = "2025-03-24_17-33-42_s2_cbfm.json"

cbf_module_folder_path = r'Examples_paper\S1\Data'

movie_name = "s1_multi-circ-tv_movie_c"

# simulation paramters
T_sim = 35
dt = 0.02
plotting_steps = int(0.1 // dt)
marker_spacing = int(2 // dt)

t0 = 0
x0 = np.array([-20, 0])
y_d = 0       # desired lateral position of vehicle

v_d = 2     # desired forward speed of vehicle

data_steps = 5  # number of data points to skip for the movie
fps = 1/(dt*data_steps)  # frames per second for the movie

# obstacle positions (distance of obstacles at least 2 x turning radius)
circ_centers = [np.array([-12, 1]),
                    np.array([0, -1]),
                    np.array([12, 1])]
period = 18
circ_radii_max = 4.0*np.ones(len(circ_centers))
circ_radii = lambda t: [circ_radii_max[0]*np.abs(np.sin(np.pi*t/period)), circ_radii_max[1]*np.abs(np.sin(np.pi*t/period - 1/3 * np.pi)), circ_radii_max[2]*np.abs(np.sin(np.pi*t/period - 2/3 * np.pi))]


########################################################################################
# load system dynamics and cbf

sintegrator_cbf_module_1 = CBFmodule()
sintegrator_cbf_module_1.load(cbf_module_filename_1, cbf_module_folder_path)

my_sintegrator_1 = sintegrator_cbf_module_1.dynamics

########################################################################################
# load system dynamics and cbf

sintegrator_cbf_module_2 = CBFmodule()
sintegrator_cbf_module_2.load(cbf_module_filename_2, cbf_module_folder_path)

my_sintegrator_2 = sintegrator_cbf_module_2.dynamics

########################################################################################
# cbf preprocessing

#iterate over indices of cbf grid points
idym = (len(sintegrator_cbf_module_2.cbf.cbf_values[1])-1)//2
for idx in range(len(sintegrator_cbf_module_2.cbf.cbf_values[0])):
    for idy in range(1,idym+1):
        sintegrator_cbf_module_2.cbf.cbf_values[idx][idym-idy] = sintegrator_cbf_module_2.cbf.cbf_values[idx][idym+idy]

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

cbf_interpolator_1 = sintegrator_cbf_module_1.cbf.getCbfInterpolator(method='linear')
cbf_interpolator_2 = sintegrator_cbf_module_2.cbf.getCbfInterpolator(method='linear')

multi_circ_cbf_2p8_grid_points = CBF.computeGridPoints([-20,-10], [20,10], [81,41])
multi_circ_cbf_1p12_grid_points = CBF.computeGridPoints([-20,-10], [20,10], [81,41])

def multi_circ_cbf_1(t,x):
    # obstacle specifications
    circ_centers_local = copy.deepcopy(circ_centers)
    
    # trafo of x for each of the obstacle circles
    x_transformed_values = [x - circ_center for circ_center in circ_centers_local]

    # compute the CBF values for each of the obstacle circles
    cbf_values_at_x = [cbf_interpolator_1(x_transformed_values[i]) - circ_radii(t)[i] for i in range(len(x_transformed_values))]

    return np.nanmin(cbf_values_at_x)

def multi_circ_cbf_2(t,x):
    # obstacle specifications
    circ_centers_local = copy.deepcopy(circ_centers)
    
    # trafo of x for each of the obstacle circles
    x_transformed_values = [x - circ_center for circ_center in circ_centers_local]

    # compute the CBF values for each of the obstacle circles
    cbf_values_at_x = [cbf_interpolator_2(x_transformed_values[i]) - circ_radii(t)[i] for i in range(len(x_transformed_values))]

    return np.nanmin(cbf_values_at_x)

# identical for both examples
def multi_circ_h(t,x):
    # obstacle specifications
    circ_centers_local = copy.deepcopy(circ_centers)
    
    # trafo of x for each of the obstacle circles
    x_transformed_values = [x - circ_center for circ_center in circ_centers_local]

    # compute the CBF values for each of the obstacle circles
    h_values_at_x = [sintegrator_cbf_module_1.h(x_transformed_values[i]) - circ_radii(t)[i] for i in range(len(x_transformed_values))]

    return np.nanmin(h_values_at_x)

# identical for both examples
def multi_circ_h_max(x):
    # obstacle specifications
    circ_centers_local = copy.deepcopy(circ_centers)
    
    # trafo of x for each of the obstacle circles
    x_transformed_values = [x - circ_center for circ_center in circ_centers_local]

    # compute the CBF values for each of the obstacle circles
    h_values_at_x = [sintegrator_cbf_module_1.h(x_transformed_values[i]) - circ_radii_max[i] for i in range(len(x_transformed_values))]

    return np.nanmin(h_values_at_x)

controller_settings_1 = {}

controller_settings_1['cbf_grid_points'] = multi_circ_cbf_2p8_grid_points
controller_settings_1['cbf_function'] = multi_circ_cbf_1
controller_settings_1['alpha'] = lambda b: alpha(b, c=2, gamma=sintegrator_cbf_module_1.gamma)
controller_settings_1['alpha_offset'] = 0.2
controller_settings_1['dynamics'] = my_sintegrator_1
controller_settings_1['dt'] = 0.1
controller_settings_1['step_size'] = 0.5


controller_settings_2 = {}

controller_settings_2['cbf_grid_points'] = multi_circ_cbf_1p12_grid_points
controller_settings_2['cbf_function'] = multi_circ_cbf_2
controller_settings_2['alpha'] = lambda b: alpha(b, c=2, gamma=sintegrator_cbf_module_2.gamma)
controller_settings_2['alpha_offset'] = 0.2
controller_settings_2['dynamics'] = my_sintegrator_2
controller_settings_2['dt'] = 0.1
controller_settings_2['step_size'] = 0.5

########################################################################################
# Simulation

print("Simulation is running...")

########################################################################################
# Simulation example 1

print("Simulation example 1...")

# Create object for saving the baseline controller trajectory
u_baseline_sol_1 = np.zeros((0,my_sintegrator_1.u_dim))

# Set initial time and state in dynamic systen
my_sintegrator_1.t0 = t0
my_sintegrator_1.x0 = x0
my_sintegrator_1.reset()

current_state = x0
for current_time in np.arange(t0, T_sim, dt):
    # Compute the control input
    u_baseline = SingleIntegrator.u_follow_straight_line(current_state,v_d,y_d)
    u_safe = sc.compute_safe_input(controller_settings_1,current_time,current_state,u_baseline)

    # simulate the system and store the solution automatically in the DynamicSystem object; the current state is updated
    current_state = my_sintegrator_1.simulate(x=current_state, u=u_safe, dt=dt,saveSolution=True)

    # save the baseline controller
    u_baseline_sol_1 = np.vstack([u_baseline_sol_1,np.array(u_baseline).reshape(1,-1)])

print("Simulation example 1 finished.")

########################################################################################
# Simulation example 2

print("Simulation example 2...")

# Create object for saving the baseline controller trajectory
u_baseline_sol_2 = np.zeros((0,my_sintegrator_2.u_dim))

# Set initial time and state in dynamic systen
my_sintegrator_2.t0 = t0
my_sintegrator_2.x0 = x0
my_sintegrator_2.reset()

current_state = x0
for current_time in np.arange(t0, T_sim, dt):
    # Compute the control input
    u_baseline = SingleIntegrator.u_follow_straight_line(current_state,v_d,y_d)
    u_safe = sc.compute_safe_input(controller_settings_2,current_time,current_state,u_baseline)

    # simulate the system and store the solution automatically in the DynamicSystem object; the current state is updated
    current_state = my_sintegrator_2.simulate(x=current_state, u=u_safe, dt=dt,saveSolution=True)

    # save the baseline controller
    u_baseline_sol_2 = np.vstack([u_baseline_sol_2,np.array(u_baseline).reshape(1,-1)])

print("Simulation example 2 finished.")

print("Simulation finished.")

########################################################################################
# Visualization

# Plot obstacle
x_obstacle_grid = np.linspace(controller_settings_1['cbf_grid_points'][0][0], controller_settings_1['cbf_grid_points'][0][-1], 100)
y_obstacle_grid = np.linspace(controller_settings_1['cbf_grid_points'][1][0], controller_settings_1['cbf_grid_points'][1][-1], 100)

H_values_start = np.array([[multi_circ_h(0.0,[xi, yi]) for xi in x_obstacle_grid] for yi in y_obstacle_grid])
H_values_end = np.array([[multi_circ_h_max([xi, yi]) for xi in x_obstacle_grid] for yi in y_obstacle_grid])

plt.figure(figsize=(6,4)) 
plt.contour(x_obstacle_grid, y_obstacle_grid, H_values_start, levels=[0], colors='lightblue', linestyles='--')      # plot the boundary of the obstacle
plt.contour(x_obstacle_grid, y_obstacle_grid, H_values_end, levels=[0], colors='g', linestyles='--')

plt.axhline(0, color='black', linewidth=1, linestyle='--')
plt.axvline(0, color='black', linewidth=1, linestyle='--')
plt.xlim([controller_settings_1['cbf_grid_points'][0][0], controller_settings_1['cbf_grid_points'][0][-1]])
plt.ylim([-5, 5])
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal')  # Ensure aspect ratio is equal

# Plot reference trajectory
plt.axhline(y=y_d, color='r', linestyle='-', linewidth=2, label='Desired trajectory')

# Plot trajectory
SingleIntegrator.plot_trajectory_with_markers(plt, my_sintegrator_1, plotting_steps, marker_spacing=marker_spacing, line_color='blue', linewidth=2, marker_size=5, label=r"Single integrator with $v_{x} \in [-2,2]$")
SingleIntegrator.plot_trajectory_with_markers(plt, my_sintegrator_2, plotting_steps, marker_spacing=marker_spacing, line_color='green', linewidth=2, marker_size=5, label=r"Single integrator with $v_{x} \in [1,2]$")

# plt.legend()

plt.show(block=False)

# Plot difference in the steering angles between the baseline and the safe controller and CBF values along the trajectory
cbf_values_along_trajectory_1 = np.array([multi_circ_cbf_1(my_sintegrator_1.t_sol[i],my_sintegrator_1.x_sol[i]) for i in range(len(my_sintegrator_1.x_sol))])
cbf_values_along_trajectory_2 = np.array([multi_circ_cbf_2(my_sintegrator_2.t_sol[i],my_sintegrator_2.x_sol[i]) for i in range(len(my_sintegrator_2.x_sol))])

# Distance to obstacle along the trajectory
distance_to_obstacle_along_trajectory_1 = np.array([multi_circ_h(my_sintegrator_1.t_sol[i], my_sintegrator_1.x_sol[i,:]) for i in range(len(my_sintegrator_1.x_sol))])
distance_to_obstacle_along_trajectory_2 = np.array([multi_circ_h(my_sintegrator_2.t_sol[i], my_sintegrator_2.x_sol[i,:]) for i in range(len(my_sintegrator_2.x_sol))])

valid_indices_1 = [i for i, dist in enumerate(distance_to_obstacle_along_trajectory_1) if dist is not None and not np.isnan(dist)]
t_filtered_1 = my_sintegrator_1.t_sol[valid_indices_1]
distance_to_obstacle_along_trajectory_filtered_1 = distance_to_obstacle_along_trajectory_1[valid_indices_1]

valid_indices_2 = [i for i, dist in enumerate(distance_to_obstacle_along_trajectory_2) if dist is not None and not np.isnan(dist)]
t_filtered_2 = my_sintegrator_2.t_sol[valid_indices_2]
distance_to_obstacle_along_trajectory_filtered_2 = distance_to_obstacle_along_trajectory_2[valid_indices_2]

fig, ax = plt.subplots(3, 1, figsize=(6, 4))

# Plot difference in the steering angles between the baseline and the safe controller and CBF values along the trajectory
ax[0].plot(my_sintegrator_1.t_sol[:-1], u_baseline_sol_1[:,1] - my_sintegrator_1.u_sol[:,1], label=r"Single integrator with $v_{x} \in [-2,2]$", color='blue')
ax[0].plot(my_sintegrator_2.t_sol[:-1], u_baseline_sol_2[:,1] - my_sintegrator_2.u_sol[:,1], label=r"Single integrator with $v_{x} \in [1,2]$", color='green')
# ax[0].legend()
ax[0].grid(True)
ax[0].set_ylabel(r'$v_{y,baseline} - v_{y,safe}$')
ax[0].set_xticklabels([])
ax[0].set_xlim([0,30])

ax[1].plot(my_sintegrator_1.t_sol, cbf_values_along_trajectory_1, label=r"Single integrator with $v_{x} \in [-2,2]$", color='blue')
ax[1].plot(my_sintegrator_2.t_sol, cbf_values_along_trajectory_2, label=r"Single integrator with $v_{x} \in [1,2]$", color='green')

ax[1].grid(True)
ax[1].set_xticklabels([])
ax[1].set_ylabel('CBF value')
ax[1].set_xlim([0,30])
# ax[1].legend()

# Distance to obstacle along the trajectory
ax[2].axhline(0, color='black', linewidth=0.5, linestyle='--')
ax[2].plot(t_filtered_1, distance_to_obstacle_along_trajectory_filtered_1, label=r"Single integrator with $v_{x} \in [-2,2]$", color='blue')
ax[2].plot(t_filtered_2, distance_to_obstacle_along_trajectory_filtered_2, label=r"Single integrator with $v_{x} \in [1,2]$", color='green')
ax[2].set_xlabel('Time [s]')
ax[2].set_ylabel('Distance')
ax[2].set_xlim([0, 30])
# ax[2].legend()
ax[2].grid(True)

plt.show(block=False)

# Plot inputs
plt.figure(figsize=(6,3))
# Plot velocity in subplot with limits
plt.subplot(2,1,1)
plt.plot(my_sintegrator_1.t_sol[:-1], my_sintegrator_1.u_sol[:,0], label=r"Single integrator with $v_{x} \in [-2,2]$", color='blue')
plt.plot(my_sintegrator_2.t_sol[:-1], my_sintegrator_2.u_sol[:,0], label=r"Single integrator with $v_{x} \in [1,2]$", color='green')
plt.axhline(my_sintegrator_1.u_max[0], color='k', linestyle='--')
plt.axhline(my_sintegrator_1.u_min[0], color='blue', linestyle='--')
plt.axhline(my_sintegrator_2.u_min[0], color='green', linestyle='--')
plt.xlim([0, 30])
plt.ylim([-0.4, 2.1])
# plt.legend()
plt.gca().set_xticklabels([])
plt.grid(True)
plt.ylabel(r'$v_x$')
# Plot steering angle in subplot with limits
plt.subplot(2,1,2)
plt.plot(my_sintegrator_1.t_sol[:-1], my_sintegrator_1.u_sol[:,1], label=r"Single integrator with $v_{x} \in [-2,2]$", color='blue')
plt.plot(my_sintegrator_2.t_sol[:-1], my_sintegrator_2.u_sol[:,1], label=r"Single integrator with $v_{x} \in [1,2]$", color='green')
plt.axhline(my_sintegrator_1.u_min[1], color='k', linestyle='--')
plt.axhline(my_sintegrator_1.u_max[1], color='k', linestyle='--')
plt.xlim([0, 30])
# plt.legend()
plt.grid(True)
plt.xlabel('Time [s]')
plt.ylabel(r'$v_y$')
plt.show(block=False)

########################################################################################

input("Press Enter to continue...")

########################################################################################
# Make a movie

print("Creating movie...")

import imageio.v2 as imageio

frames_dir = "frames_s14"
os.makedirs(frames_dir, exist_ok=True)

frames = []
for i in tqdm(range(0,len(my_sintegrator_1.t_sol),data_steps), desc="Creating frames", unit="frame"):
    fig, ax = plt.subplots(figsize=(12,6))
    
    ax.set_xlim([x0[0], controller_settings_1['cbf_grid_points'][0][-1]])
    ax.set_ylim([-5, 5])
    ax.set_aspect('equal')

    # Plot obstacle
    H_values_tv = np.array([[multi_circ_h(my_sintegrator_1.t_sol[i],[xi, yi]) for xi in x_obstacle_grid] for yi in y_obstacle_grid])

    ax.contourf(x_obstacle_grid, y_obstacle_grid, H_values_tv, levels=[-np.inf,0], colors='gray', alpha=0.5)  # plot the obstacle
    ax.contour(x_obstacle_grid, y_obstacle_grid, H_values_tv, levels=[0], colors='k') 

    # Plot reference trajectory
    ax.axhline(y=y_d, color='r', linestyle='-')

    # Plot vehicle at current time step
    ax.plot(my_sintegrator_1.x_sol[:i,0], my_sintegrator_1.x_sol[:i,1], 'b-', linewidth=1, label=r"Single integrator with $v_{x} \in [-2,2]")  # Draw trajectory until current time step
    ax.plot(my_sintegrator_2.x_sol[:i,0], my_sintegrator_2.x_sol[:i,1], 'g-', linewidth=1, label=r"Single integrator with $v_{x} \in [1,2]")  # Draw trajectory until current time step
    SingleIntegrator.plot_trajectory_at_indices_withmarkers(plt, my_sintegrator_1, [i], marker_size=4, color='blue')
    SingleIntegrator.plot_trajectory_at_indices_withmarkers(plt, my_sintegrator_2, [i], marker_size=4, color='green')

    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.1), ncol=2)

    # Add current time to the upper right corner
    ax.text(0.95, 0.95, f"t = {my_sintegrator_1.t_sol[i]:.2f}", transform=ax.transAxes, ha='right', va='top')

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

    



