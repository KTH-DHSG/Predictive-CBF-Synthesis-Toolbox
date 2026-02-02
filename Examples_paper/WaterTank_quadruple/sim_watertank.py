"""

    Code for simulating a quadruple water tank system with a time-varying Control Barrier Function (CBF).

    (c) Adrian Wiltz, 2025

"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
import Controller.SafeController as sc
import Auxiliaries.auxiliary_math as aux_math
import Auxiliaries.auxiliary as aux
from Dynamics.WaterTank_quadruple import WaterTank
from CBF.CBFmodule import CBFmodule
from CBF.CBF import CBF
import matplotlib.pyplot as plt
import casadi as ca
from tqdm import tqdm
import copy

########################################################################################
# Specify the system dynamics, the CBF computation parameters, and initialize the CBF module   

# some parameters

cbf_module_filename = "2025-11-14_19-17-07_b_watertank.json"

cbf_module_folder_path = r'Examples_paper\WaterTank_quadruple\Data'

# simulation paramters
T_sim =600.0
dt = 0.1
plotting_steps = int(0.1 // dt)
marker_spacing = int(2 // dt)

t0 = 0
x0 = np.array([4, 4, 4, 4])   # initial state

def smooth_fmax(a, b, k=10.0):
    # smooth approximation of max(a,b)
    return (1.0 / k) * ca.log(ca.exp(k * a) + ca.exp(k * b))

# shift
def lambda_func(t):

    T_1 = 150.0
    T_2 = 400.0
    T_3 = T_2+150.0

    min_val = 2.0
    max_val = 10.0

    start_val = min_val
    end_val_1 = max_val
    end_val_2 = min_val
    end_val_3 = max_val

    m = 0.1

    if t < T_1:
        max_decrease_func = max_val - start_val - m*t
        return smooth_fmax(max_decrease_func, max_val-end_val_1)
    elif t < T_2:
        return max_val-end_val_2
    elif t < T_3:
        t = t - T_2
        max_decrease_func = max_val - end_val_2 - m*t
        return smooth_fmax(max_decrease_func, max_val-end_val_3)
    else:
        return max_val - end_val_3

########################################################################################
# load system dynamics and cbf

# load less agile system
cbf_module_watertank = CBFmodule()
cbf_module_watertank.load(cbf_module_filename, cbf_module_folder_path)

my_watertank = cbf_module_watertank.dynamics

########################################################################################
# Setup system and controller for simulation

def alpha(b):
    
    return 0.5*b

cbf_interpolator = cbf_module_watertank.cbf.getCbfInterpolator(method='linear')

def tv_cbf(t,x):
    
    return cbf_interpolator(x) + lambda_func(t)

grid_points = CBF.computeGridPoints([0.5,0.5,0.5,0.5], [14.0,14.0,14.0,14.0], [20,20,20,20])

# controller settings for less agile system
controller_settings = {}

controller_settings['cbf_grid_points'] = grid_points
controller_settings['cbf_function'] = tv_cbf
controller_settings['alpha'] = lambda b: alpha(b)
controller_settings['alpha_offset'] = 0.2       
controller_settings['dynamics'] = my_watertank
controller_settings['dt'] = 1.0
controller_settings['step_size'] = 0.5

########################################################################################
# Simulation

print("Simulation is running...")

# simulate cart pole
print("Simulation of watertank started.")

# Create object for saving the baseline controller trajectory
u_baseline_sol = np.zeros((0,my_watertank.u_dim))

# Set initial time and state in dynamic systen
my_watertank.t0 = t0
my_watertank.x0 = x0
my_watertank.reset()

current_state = x0
steps = int((T_sim - t0) / dt)
for i in tqdm(range(steps), desc="Simulating watertank", unit="step"):
    current_time = t0 + i*dt

    # Compute the control input
    u_baseline = np.array([0, 0])
    u_safe = sc.compute_safe_input(controller_settings, current_time, current_state, u_baseline)

    # simulate the system and store the solution automatically in the DynamicSystem object; the current state is updated
    current_state = my_watertank.simulate(x=current_state, u=u_safe, dt=dt, saveSolution=True)

    # save the baseline controller
    u_baseline_sol = np.vstack([u_baseline_sol, np.array(u_baseline).reshape(1, -1)])

print("Simulation of watertank finished.")

print("Simulation finished.")

########################################################################################
# Visualization

print("Creating plots...")

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(x0)]
color_cycle_light = [aux.lighten(color, amount=0.5) for color in color_cycle]

# Minumum height tankwise
h_min_1 = 5.0 - np.array([lambda_func(t)/2.0 for t in my_watertank.t_sol])
h_min_2 = 5.0 - np.array([lambda_func(t)/2.0 for t in my_watertank.t_sol])
h_min_3 = 10.0 - np.array([lambda_func(t) for t in my_watertank.t_sol])
h_min_4 = 10.0 - np.array([lambda_func(t) for t in my_watertank.t_sol])

plt.figure()
plt.plot(my_watertank.t_sol, my_watertank.x_sol[:,0], color=color_cycle[0], linestyle='-', label='$h_1$')
plt.plot(my_watertank.t_sol, my_watertank.x_sol[:,1], color=color_cycle[1], linestyle='-', label='$h_2$')
plt.plot(my_watertank.t_sol, my_watertank.x_sol[:,2], color=color_cycle[2], linestyle='-', label='$h_3$')
plt.plot(my_watertank.t_sol, my_watertank.x_sol[:,3], color=color_cycle[3], linestyle='-', label='$h_4$')
plt.plot(my_watertank.t_sol, h_min_1, color=color_cycle_light[0], linestyle='--', label='$h_{min,1}$, $h_{min,2}$')
plt.plot(my_watertank.t_sol, h_min_3, color=color_cycle_light[2], linestyle='--', label='$h_{min,3}$, $h_{min,4}$')
plt.xlabel('Time [s]')
plt.legend()
plt.grid()
plt.show(block=False)

plt.figure(figsize=(6,2))
b_values = [tv_cbf(my_watertank.t_sol[k], my_watertank.x_sol[k]) for k in range(len(my_watertank.t_sol))]
plt.plot(my_watertank.t_sol, b_values, label='CBF value')
plt.xlabel('Time [s]')
plt.axhline(y=0, color='r', linestyle='--', label='CBF boundary')
plt.legend()
plt.grid()
plt.show(block=False)

########################################################################################

input("Press Enter to continue...")
print("Done.")