"""

    Computation of the CBF for the unicycle model with a circular obstacle.

    The CBF is computed in parallel using the CBFcomputation module and using all the available cores on the computer.
    The results are saved in a json file.

    (c) Adrian Wiltz, 2025
    
"""

if __name__ == '__main__':

    import sys
    import os

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

    import numpy as np
    from Dynamics.Unicycle import Unicycle
    from CBF.CBFmodule import CBFmodule
    import CBF.CBFcomputation as CBFcomputation
    import casadi as ca
    import time

    ########################################################################################
    # Specify the system dynamics, the CBF computation parameters, and initialize the CBF module   

    # some parameters
    num_of_batches_factor = 40      # determines the number of batches for parallel computation
    cbf_file_name = "u1_2_cbfm_1p12.json"
    
    # create a dynamic system
    t0 = 0
    x0 = np.array([0,0,0])
    L = 1
    u_min = np.array([1, -0.9])    # [minimum speed, minimum steering angle] in [m/s, rad/s]
    u_max = np.array([2, 0.9])     # [maximum speed, maximum steering angle] in [m/s, rad/s]
    myUnicycle = Unicycle(x0=x0,
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
        r = 10
        h = lambda x: (x[0]-xc)**2 + (x[1]-yc)**2 - r**2
        turning_radius = 1.12
        h_grad = lambda x: ca.vertcat(2*(x[0]-xc), 2*(x[1]-yc))
        delta = (2*turning_radius)**2
        orientation = ca.vertcat(ca.cos(x[2]), ca.sin(x[2]))

        return ca.vertcat(ca.dot(h_grad(x), orientation), h(x) - delta)

    # set parameters for the CBF module
    T = 10
    gamma = 2

    # set domain bounds
    domain_lower_bound = np.array([-10,-10,-np.pi])
    domain_upper_bound = np.array([10,10,np.pi])
    discretization = np.array([41,41,41])

    print("Number of grid points to be computed: ", np.prod(discretization))

    # create a CBF module
    myCBFmodule = CBFmodule(h=h, 
                            dynamicSystem=myUnicycle, 
                            cf=cf, 
                            T=T, 
                            N=30,
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
    warmStartInputTrajectory_0 = np.array([u_min[0]*np.ones(myCBFmodule.N), 
                                            u_max[1]*np.ones(myCBFmodule.N)])   # max speed and turn to the right
    warmStartInputTrajectory_0[0,curve_steps:] = u_max[0]*np.ones(myCBFmodule.N-curve_steps) # set the steering angle to zero after curve
    warmStartInputTrajectory_0[1,curve_steps:] = np.zeros(myCBFmodule.N-curve_steps) # set the steering angle to zero after curve
    warmStartInputTrajectory_1 = np.array([u_max[0]*np.ones(myCBFmodule.N),
                                            np.zeros(myCBFmodule.N)])   # go straight at max speed
    warmStartInputTrajectory_2 = np.array([u_min[0]*np.ones(myCBFmodule.N),
                                            u_min[1]*np.ones(myCBFmodule.N)])   # max speed and turn to the left
    warmStartInputTrajectory_2[0,curve_steps:] = u_max[0]*np.ones(myCBFmodule.N-curve_steps) # set the steering angle to zero after curve
    warmStartInputTrajectory_2[1,curve_steps:] = np.zeros(myCBFmodule.N-curve_steps) # set the steering angle to zero after curve

    warmStartInputTrajectories = np.array([warmStartInputTrajectory_0, warmStartInputTrajectory_1, warmStartInputTrajectory_2])

    myCBFmodule.setWarmStartInputTrajectories(warmStartInputTrajectories)

    # 2. Compute the CBF on the domain
    tic = time.time()
    CBFcomputation.computeCbfParallelized(myCBFmodule, processes=None, timeout_per_sample=300, num_of_batches_factor=num_of_batches_factor)
    toc = time.time()
    print("CBF computation took ", toc-tic, " seconds.")

    # 3. Save the CBF module to a file
    myCBFmodule.cbf.computation_time = toc-tic

    # 4. Save the CBF module to a file
    myCBFmodule.save(cbf_file_name, folder_name="Data")

    print("CBF computation finished.")


