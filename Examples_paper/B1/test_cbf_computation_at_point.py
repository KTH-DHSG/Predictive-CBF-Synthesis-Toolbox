"""

    This script illustrates the usage of the CBF module for computing the CBF value at one particular sample point. It also allows to test warm start input trajectories for the CBF computation.

    The script initializes the CBF module, sets the warm start input trajectories and computes the CBF value at a sample point.
    The script visualizes the obstacle and the optimal trajectory for the sample point in comparision to the chosen warm start trajectory.

    (c) Adrian Wiltz, 2025
    
"""

########################################################################################
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
from Dynamics.Bicycle import Bicycle
from CBF.CBFmodule import CBFmodule
import CBF.CBFcomputation as CBFcomputation
import casadi as ca
import matplotlib.pyplot as plt
import time


########################################################################################
    
# some parameters
num_of_batches_factor = 40      # determines the number of batches for parallel computation
cbf_file_name = "b1_1_cbfm_2p8.json"

# create a dynamic system
t0 = 0
x0 = np.array([0,0,0])
L = 1
u_min = np.array([1, -20/180*np.pi])    # [minimum speed, minimum steering angle] in [m/s, rad/s]
u_max = np.array([2, 20/180*np.pi])     # [maximum speed, maximum steering angle] in [m/s, rad/s]
myBike = Bicycle(x0=x0,
                L=1,
                u_min=u_min,
                u_max=u_max)

#create a state constraint function

def h(x):
    """State constraint function"""
    xc = 0
    yc = 0
    r = 0
    return np.sqrt((x[0]-xc)**2 + (x[1]-yc)**2) - r

# create a terminal constraint function
def cf(x):
    """Terminal constraint function for casadi type arguments"""

    xc = 0
    yc = 0
    r = 15
    h = lambda x: (x[0]-xc)**2 + (x[1]-yc)**2 - r**2
    turning_radius = 2.8
    h_grad = lambda x: ca.vertcat(2*(x[0]-xc), 2*(x[1]-yc))
    delta = (2*turning_radius)**2
    orientation = ca.vertcat(ca.cos(x[2]), ca.sin(x[2]))

    return ca.vertcat(ca.dot(h_grad(x), orientation), h(x) - delta)

# set parameters for the CBF module
T = 12
gamma = 2

# set domain bounds
domain_lower_bound = np.array([-20,-20,-np.pi])
domain_upper_bound = np.array([20,20,np.pi])
discretization = np.array([40,40,40])

print("Number of grid points to be computed: ", np.prod(discretization))

# create a CBF module
myCBFmodule = CBFmodule(h=h, 
                        dynamicSystem=myBike, 
                        cf=cf, 
                        T=T, 
                        N =30,
                        gamma=gamma, 
                        domain_lower_bound=domain_lower_bound, 
                        domain_upper_bound=domain_upper_bound, 
                        discretization=discretization,
                        p_norm=50,
                        p_norm_decrement=10,
                        p_norm_min=40)

########################################################################################
# Initialize the cbf value optimization and compute the cbf value at a selction of sample points

curve_steps = 10

# 1. Initialize the warm start input trajectories and assign them to the cbf module
warmStartInputTrajectory_0 = np.array([u_max[0]*np.ones(myCBFmodule.N), 
                                        u_min[1]*np.ones(myCBFmodule.N)])   # max speed and turn to the right
warmStartInputTrajectory_0[1,curve_steps:] = np.zeros(myCBFmodule.N-curve_steps) # set the steering angle to zero after curve
warmStartInputTrajectory_1 = np.array([u_max[0]*np.ones(myCBFmodule.N),
                                        np.zeros(myCBFmodule.N)])   # go straight at max speed
warmStartInputTrajectory_2 = np.array([u_max[0]*np.ones(myCBFmodule.N),
                                        u_max[1]*np.ones(myCBFmodule.N)])   # max speed and turn to the left
warmStartInputTrajectory_2[1,curve_steps:] = np.zeros(myCBFmodule.N-curve_steps) # set the steering angle to zero after curve
warmStartInputTrajectories = np.array([warmStartInputTrajectory_0, warmStartInputTrajectory_1, warmStartInputTrajectory_2])


myCBFmodule.setWarmStartInputTrajectories(warmStartInputTrajectories)


# 2. Initialize the opti_object
opt_specs_with_dynamics = myCBFmodule.__getOptSpecsWithoutSerialization__() # get the opti specs

opti_object = CBFcomputation.initializeCbfComputation(opt_specs_with_dynamics) # initialize the opti object

# 3. Compute the cbf value at sample point
sample_point = np.array([2,1,np.pi])

tic = time.time()
cbf_value, u_opt = CBFcomputation.computeCbfAtPoint(opti_object, sample_point, warmStartInputTrajectories)
toc = time.time()
_, x_opt = myBike.simulateOverHorizon(sample_point, u_opt, myCBFmodule.dt)
print("CBF value at point ", sample_point, " is: ", cbf_value, " Computation time: ", toc-tic)

# 4. Visualize the obstacle and trajectory of first sample point
x_obstacle_grid = np.linspace(domain_lower_bound[0], domain_upper_bound[0], 100)
y_obstacle_grid = np.linspace(domain_lower_bound[1], domain_upper_bound[1], 100)

H_values = np.array([[h([xi, yi]) for xi in x_obstacle_grid] for yi in y_obstacle_grid])

plt.figure(figsize=(6,6))
plt.contourf(x_obstacle_grid, y_obstacle_grid, H_values, levels=[-np.inf,0], colors='gray', alpha=0.5)  # plot the obstacle
plt.contour(x_obstacle_grid, y_obstacle_grid, H_values, levels=[0], colors='k')      # plot the boundary of the obstacle

if u_opt is not None:
    _, x_opt = myBike.simulateOverHorizon(sample_point, u_opt, myCBFmodule.dt)
    plt.plot(x_opt[0,:], x_opt[1,:], 'r-', label='Optimal trajectory')

# plot the warm start state trajectories
for i in range(warmStartInputTrajectories.shape[0]):
    _, x_warmstart = myBike.simulateOverHorizon(sample_point, warmStartInputTrajectories[i], myCBFmodule.dt)
    plt.plot(x_warmstart[0,:], x_warmstart[1,:], 'b--', label='Warm start trajectory ' + str(i))

plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.xlim(domain_lower_bound[0], domain_upper_bound[0])
plt.ylim(domain_lower_bound[1], domain_upper_bound[1])
plt.gca().set_aspect('equal')  # Ensure aspect ratio is equal

plt.show()


