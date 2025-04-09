"""

    Simulation of single integrator 1 and 2 and a double integrator in an environment with one static obstacle.

    The simulation is performed using the SafeController module and the CBF module. 

    (c) Adrian Wiltz, 2025
    
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
import Controller.SafeController as sc
import Auxiliaries.auxiliary_math as aux_math
from Dynamics.DoubleIntegrator import DoubleIntegrator
from Dynamics.SingleIntegrator import SingleIntegrator
from CBF.CBFmodule import CBFmodule
from CBF.CBF import CBF
import matplotlib.pyplot as plt
from tqdm import tqdm

########################################################################################
# some parameters

cbf_module_filename_s1 = "2025-03-22_19-14-36_s1_cbfm.json"
cbf_module_filename_s2 = "2025-03-24_17-33-42_s2_cbfm.json"
cbf_module_filename_d = "2025-03-22_22-04-23_d2_cbfm.json"

cbf_module_folder_path = r'Examples_paper\SC\Data'

movie_name = "single-circ-static_movie"

cbf_offset = -2

# simulation paramters
T_sim = 20
dt = 0.01
marker_spacing = int(2 // dt)

t0 = 0
x0_s = np.array([-10, 0])        # initial state of the single integrator
x0_d = np.array([-10, 0, 2, 0])  # initial state of the double integrator
y_desired = 0       # desired lateral position of vehicle

v_desired = 2     # desired forward speed of vehicle

data_steps = 5  # number of data points to skip for the movie
fps = 1/(dt*data_steps)  # frames per second for the movie

P = np.diag([2,1])     # weighting matrix for the cost function of the safety controller
P_d = np.diag([5,1])

########################################################################################
# load system dynamics and cbf

# Single integrator cbf module and dynamics 1
single_integrator_cbf_module_s1 = CBFmodule()
single_integrator_cbf_module_s1.load(cbf_module_filename_s1, cbf_module_folder_path)

single_integrator_cbf_module_s1.cbf.cbf_values += cbf_offset

my_single_integrator_s1 = single_integrator_cbf_module_s1.dynamics

# Single integrator cbf module and dynamics 2
single_integrator_cbf_module_s2 = CBFmodule()
single_integrator_cbf_module_s2.load(cbf_module_filename_s2, cbf_module_folder_path)

single_integrator_cbf_module_s2.cbf.cbf_values += cbf_offset

my_single_integrator_s2 = single_integrator_cbf_module_s2.dynamics

# Double integrator cbf module and dynamics
dintegrator_cbf_module_d = CBFmodule()
dintegrator_cbf_module_d.load(cbf_module_filename_d, cbf_module_folder_path)

dintegrator_cbf_module_d.cbf.cbf_values += cbf_offset

my_dintegrator = dintegrator_cbf_module_d.dynamics


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

# Controller settings for single integrator 1
cbf_interpolator_s1 = single_integrator_cbf_module_s1.cbf.getCbfInterpolator(method='linear')

controller_settings_s1 = {}

controller_settings_s1['cbf_grid_points'] = single_integrator_cbf_module_s1.cbf.getCbfGridPoints()
controller_settings_s1['cbf_interpolator'] = cbf_interpolator_s1
controller_settings_s1['alpha'] = lambda b: alpha(b, c=2, gamma=single_integrator_cbf_module_s1.gamma)
controller_settings_s1['alpha_offset'] = 0.2
controller_settings_s1['dynamics'] = my_single_integrator_s1
controller_settings_s1['lambda_fun'] = lambda t: 0
controller_settings_s1['dt'] = 0.5
controller_settings_s1['step_size'] = 0.5

# Controller settings for single integrator 2
cbf_interpolator_s2 = single_integrator_cbf_module_s2.cbf.getCbfInterpolator(method='linear')

controller_settings_s2 = {}

controller_settings_s2['cbf_grid_points'] = single_integrator_cbf_module_s2.cbf.getCbfGridPoints()
controller_settings_s2['cbf_interpolator'] = cbf_interpolator_s2
controller_settings_s2['alpha'] = lambda b: alpha(b, c=2, gamma=single_integrator_cbf_module_s2.gamma)
controller_settings_s2['alpha_offset'] = 0.2
controller_settings_s2['dynamics'] = my_single_integrator_s2
controller_settings_s2['lambda_fun'] = lambda t: 0
controller_settings_s2['dt'] = 0.1
controller_settings_s2['step_size'] = 0.5

# Controller settings for double integrator
cbf_interpolator_d = dintegrator_cbf_module_d.cbf.getCbfInterpolator(method='linear')

controller_settings_d = {}

controller_settings_d['cbf_grid_points'] = dintegrator_cbf_module_d.cbf.getCbfGridPoints()
controller_settings_d['cbf_interpolator'] = cbf_interpolator_d
controller_settings_d['alpha'] = lambda b: alpha(b, c=0.5, gamma=dintegrator_cbf_module_d.gamma)
controller_settings_d['alpha_offset'] = 0.2
controller_settings_d['dynamics'] = my_dintegrator
controller_settings_d['lambda_fun'] = lambda t: 0 
controller_settings_d['dt'] = 0.1
controller_settings_d['step_size'] = 0.5

########################################################################################
# Simulation

print("Simulation is running...")

########################################################################################
# Simulation for single integrator 1

print("Simulation for single integrator 1 is running...")

# Create object for saving the baseline controller trajectory
u_baseline_sol_s1 = np.zeros((0,my_single_integrator_s1.u_dim))

# Set initial time and state in dynamic systen
my_single_integrator_s1.t0 = t0
my_single_integrator_s1.x0 = x0_s
my_single_integrator_s1.reset()

current_state = x0_s
for current_time in np.arange(t0, T_sim, dt):
    # Compute the control input
    u_baseline = SingleIntegrator.u_follow_straight_line(current_state,v_desired,y_desired)
    u_safe = sc.compute_safe_input(controller_settings_s1,current_time,current_state,u_baseline, P=P)

    # simulate the system and store the solution automatically in the DynamicSystem object; the current state is updated
    current_state = my_single_integrator_s1.simulate(x=current_state, u=u_safe, dt=dt,saveSolution=True)

    # save the baseline controller
    u_baseline_sol_s1 = np.vstack([u_baseline_sol_s1,np.array(u_baseline).reshape(1,-1)])

print("Simulation for single integrator 1 finished.")

########################################################################################
# Simulation for single integrator 2

print("Simulation for single integrator 2 is running...")

# Create object for saving the baseline controller trajectory
u_baseline_sol_s2 = np.zeros((0,my_single_integrator_s2.u_dim))

# Set initial time and state in dynamic systen
my_single_integrator_s2.t0 = t0
my_single_integrator_s2.x0 = x0_s
my_single_integrator_s2.reset()

current_state = x0_s
for current_time in np.arange(t0, T_sim, dt):
    # Compute the control input
    u_baseline = SingleIntegrator.u_follow_straight_line(current_state,v_desired,y_desired)
    u_safe = sc.compute_safe_input(controller_settings_s2,current_time,current_state,u_baseline, P=P)

    # simulate the system and store the solution automatically in the DynamicSystem object; the current state is updated
    current_state = my_single_integrator_s2.simulate(x=current_state, u=u_safe, dt=dt,saveSolution=True)

    # save the baseline controller
    u_baseline_sol_s2 = np.vstack([u_baseline_sol_s2,np.array(u_baseline).reshape(1,-1)])

print("Simulation for single integrator 2 finished.")

########################################################################################
# Simulation for double integrator

print("Simulation for double integrator is running...")

# Create object for saving the baseline controller trajectory
u_baseline_sol_d = np.zeros((0,my_dintegrator.u_dim))

# Set initial time and state in dynamic systen
my_dintegrator.t0 = t0
my_dintegrator.x0 = x0_d
my_dintegrator.reset()

current_state = x0_d
for current_time in np.arange(t0, T_sim, dt):
    # Compute the control input
    u_baseline = DoubleIntegrator.u_follow_straight_line(current_state,v_desired,y_desired)
    u_safe = sc.compute_safe_input(controller_settings_d,current_time,current_state,u_baseline, P=P_d)

    # simulate the system and store the solution automatically in the DynamicSystem object; the current state is updated
    current_state = my_dintegrator.simulate(x=current_state, u=u_safe, dt=dt,saveSolution=True)

    # save the baseline controller
    u_baseline_sol_d = np.vstack([u_baseline_sol_d,np.array(u_baseline).reshape(1,-1)])

print("Simulation for double integrator finished.")

print("Simulation finished.")

########################################################################################
# Visualization

# For illustration of location obstacle
def h2d(x):
    xc = 0
    yc = 0
    r = 0
    return np.sqrt((x[0]-xc)**2 + (x[1]-yc)**2) - r

# Plot obstacle
x_obstacle_grid = np.linspace(controller_settings_d['cbf_grid_points'][0][0], controller_settings_d['cbf_grid_points'][0][-1], 100)
y_obstacle_grid = np.linspace(controller_settings_d['cbf_grid_points'][1][0], controller_settings_d['cbf_grid_points'][1][-1], 100)

H_values = np.array([[h2d([xi, yi]) + cbf_offset for xi in x_obstacle_grid] for yi in y_obstacle_grid])

plt.figure(figsize=(8,4))  
plt.contourf(x_obstacle_grid, y_obstacle_grid, H_values, levels=[-np.inf,0], colors='gray', alpha=0.5)  # plot the obstacle
plt.contour(x_obstacle_grid, y_obstacle_grid, H_values, levels=[0], colors='k')      # plot the boundary of the obstacle

plt.axhline(0, color='red', linewidth=1.5, linestyle='-')
plt.axvline(0, color='black', linewidth=1, linestyle='--')
plt.xlim([controller_settings_d['cbf_grid_points'][0][0], 20])
plt.ylim([-10,10])
plt.xlim([-10, 20])
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal')  # Ensure aspect ratio is equal

# Plot trajectory
SingleIntegrator.plot_trajectory_with_markers(plt, my_single_integrator_s1, marker_spacing=marker_spacing, line_color='blue', linewidth=1.5, marker_size=3)
SingleIntegrator.plot_trajectory_with_markers(plt, my_single_integrator_s2, marker_spacing=marker_spacing, line_color='green', linewidth=1.5, marker_size=3)
DoubleIntegrator.plot_trajectory_with_markers(plt, my_dintegrator, marker_spacing=marker_spacing, line_color='gold', linewidth=1.5, marker_size=3)

plt.show(block=False)

# Plot difference in the steering angles between the baseline and the safe controller and CBF values along the trajectory
cbf_values_along_trajectory_s1 = np.array([cbf_interpolator_s1(x) for x in my_single_integrator_s1.x_sol])
cbf_values_along_trajectory_s2 = np.array([cbf_interpolator_s2(x) for x in my_single_integrator_s2.x_sol])
cbf_values_along_trajectory_d = np.array([cbf_interpolator_d(x) for x in my_dintegrator.x_sol])

plt.figure(figsize=(4,4))

plt.plot(my_single_integrator_s1.t_sol, cbf_values_along_trajectory_s1, label='CBF value S1', color='blue')
plt.plot(my_single_integrator_s2.t_sol, cbf_values_along_trajectory_s2, label='CBF value S2', color='green')
plt.plot(my_dintegrator.t_sol, cbf_values_along_trajectory_d, label='CBF value D', color='gold')
plt.grid(True)
# plt.legend()
plt.xlim([0, 12])
plt.ylim([-0.5, 10])
plt.xlabel('Time [s]')
plt.ylabel('CBF value')

plt.show(block=False)

########################################################################################

input("Press Enter to continue...")

########################################################################################
# Make a movie

print("Creating movie...")

import imageio.v2 as imageio

frames_dir = "frames_d11"
os.makedirs(frames_dir, exist_ok=True)

frames = []
for i in tqdm(range(0,len(my_dintegrator.t_sol),data_steps), desc="Creating frames", unit="frame"):
    fig, ax = plt.subplots(figsize=(8,4))
    
    plt.xlim([-10, 20])
    plt.ylim([controller_settings_d['cbf_grid_points'][1][0], controller_settings_d['cbf_grid_points'][1][-1]])
    ax.set_aspect('equal')

    # Plot obstacle
    H_values_tv = H_values + controller_settings_d['lambda_fun'](my_dintegrator.t_sol[i])

    ax.contourf(x_obstacle_grid, y_obstacle_grid, H_values, levels=[-np.inf,0], colors='gray', alpha=0.5)  # plot the obstacle
    ax.contour(x_obstacle_grid, y_obstacle_grid, H_values, levels=[0], colors='k') 

    # Plot reference trajectory
    ax.axhline(y=0, color='r', linestyle='-')

    # Plot single integrator 1 at current time step
    ax.plot(my_single_integrator_s1.x_sol[:i,0], my_single_integrator_s1.x_sol[:i,1], 'b-', linewidth=2, label='Single Integrator with $v_x\in[-2,2]$')  # Draw trajectory until current time step
    SingleIntegrator.plot_trajectory_at_indices_withmarkers(plt, my_single_integrator_s1, [i], marker_size=4, color='blue')

    # Plot single integrator 2 at current time step
    ax.plot(my_single_integrator_s2.x_sol[:i,0], my_single_integrator_s2.x_sol[:i,1], 'g-', linewidth=2, label='Single Integrator with $v_x\in[1,2]$')  # Draw trajectory until current time step
    SingleIntegrator.plot_trajectory_at_indices_withmarkers(plt, my_single_integrator_s2, [i], marker_size=4, color='green')

    # Plot double integrator at current time step
    ax.plot(my_dintegrator.x_sol[:i,0], my_dintegrator.x_sol[:i,1], color='gold', linewidth=2, label='Double Integrator')  # Draw trajectory until current time step
    DoubleIntegrator.plot_trajectory_at_indices_withmarkers(plt, my_dintegrator, [i], marker_size=4, color='gold')

    # Add current time to the upper right corner
    ax.text(0.95, 0.95, f"t = {my_dintegrator.t_sol[i]:.2f}", transform=ax.transAxes, ha='right', va='top')

    ax.legend(loc='lower left')

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

    



