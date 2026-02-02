"""

    Code for computing a Control Barrier Function (CBF) for a 2D double integrator system.

    (c) Adrian Wiltz, 2025
    
"""

if __name__ == "__main__":

    import sys
    import os

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

    import numpy as np
    from Dynamics.DoubleIntegrator import DoubleIntegrator
    from CBF.CBFmodule import CBFmodule
    import CBF.CBFcomputation as CBFcomputation
    import casadi as ca
    import time

    ########################################################################################

    # some parameters
    num_of_batches_factor = 50      # determines the number of batches for parallel computation
    cbf_file_name = "d1_cbfm.json"

    # create a dynamic system
    x0 = np.array([0,0,0,0])
    u_min = np.array([-1.0,-1.0])    # [minimum speed, minimum steering angle] in [m/s, rad/s]
    u_max = np.array([1.0,1.0])     # [maximum speed, maximum steering angle] in [m/s, rad/s]
    myDoubleIntegrator = DoubleIntegrator(x0=x0,
                    u_min=u_min,
                    u_max=u_max)

    #create a state constraint function

    def h(x):
        """State constraint function for both numpy and casadi type arguments"""

        v_min = np.array([-2,-2])
        v_max = np.array([2,2])

        xc = 0
        yc = 0
        r = 0

        factor = 100

        if isinstance(x[0], (np.ndarray, float, int)):
            h0 = np.sqrt((x[0]-xc)**2 + (x[1]-yc)**2) - r
            h1 = factor * np.minimum(v_max[0] - x[2], x[2] - v_min[0])
            h2 = factor * np.minimum(v_max[1] - x[3], x[3] - v_min[1])
        else:
            h0 = ca.sqrt((x[0]-xc)**2 + (x[1]-yc)**2) - r
            h1 = factor * ca.fmin(v_max[0] - x[2], x[2] - v_min[0])
            h2 = factor * ca.fmin(v_max[1] - x[3], x[3] - v_min[1])

        h = ca.fmin(ca.fmin(h0, h1), h2) if isinstance(x[0], ca.MX) else np.min([h0, h1, h2])

        return h

    # create a terminal constraint function
    def cf(x):
        """Terminal constraint function for casadi type arguments"""

        xc = 0
        yc = 0
        r = 10
        h0 = lambda x: (x[0]-xc)**2 + (x[1]-yc)**2 - r**2
        h0_grad = lambda x: ca.vertcat(2*(x[0]-xc), 2*(x[1]-yc))
        orientation = x[2:]/ca.norm_2(x[2:])

        return ca.vertcat(ca.dot(h0_grad(x), orientation),h0(x))

    # set parameters for the CBF module
    T = 12
    gamma = 1

    # set domain bounds
    domain_lower_bound = np.array([-14,-14,-2.5,-2.5])
    domain_upper_bound = np.array([14,14,2.5,2.5])
    discretization = np.array([29,29,15,15])

    print("Number of grid points to be computed: ", np.prod(discretization))

    # create a CBF module
    myCBFmodule = CBFmodule(h=h, 
                            dynamicSystem=myDoubleIntegrator, 
                            cf=cf, 
                            T=T, 
                            N=30,
                            gamma=gamma, 
                            domain_lower_bound=domain_lower_bound, 
                            domain_upper_bound=domain_upper_bound, 
                            discretization=discretization,
                            p_norm=40,
                            p_norm_decrement=10,
                            p_norm_min=40)

    ########################################################################################
    # Initialize the cbf value optimization and compute the cbf value at a selction of sample points

    # 1. Initialize the warm start input trajectories and assign them to the cbf module
    warmStartInputTrajectory_0 = np.array([u_min[0]*np.ones(myCBFmodule.N), 
                                            np.zeros(myCBFmodule.N)])   # max acceleration backward
    warmStartInputTrajectory_1 = np.array([u_max[0]*np.ones(myCBFmodule.N),
                                            np.zeros(myCBFmodule.N)])   # max accelreation forward
    warmStartInputTrajectory_2 = np.array([np.zeros(myCBFmodule.N),
                                            u_min[1]*np.ones(myCBFmodule.N)])   # max acceleration down
    warmStartInputTrajectories_3 = np.array([np.zeros(myCBFmodule.N),
                                            u_max[1]*np.ones(myCBFmodule.N)])   # max acceleration up

    warmStartInputTrajectories = np.array([warmStartInputTrajectory_0, warmStartInputTrajectory_1, warmStartInputTrajectory_2, warmStartInputTrajectories_3])

    myCBFmodule.setWarmStartInputTrajectories(warmStartInputTrajectories)

    # 2. Compute the CBF on the domain
    tic = time.time()
    CBFcomputation.computeCbfParallelized(myCBFmodule, processes=None, timeout_per_sample=10000, num_of_batches_factor=num_of_batches_factor)
    toc = time.time()
    print("CBF computation took ", toc-tic, " seconds.")

    # 3. Save the CBF module to a file
    myCBFmodule.cbf.computation_time = toc-tic

    # 4. Save the CBF module to a file
    myCBFmodule.save(cbf_file_name, folder_name="Data")

    print("CBF computation finished.")


