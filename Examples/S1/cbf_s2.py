"""

    Computation of the CBF for single integrator 2 (input constraints are U=[1,2]x[-2,2]) with a circular obstacle.

    The CBF is computed in parallel using the CBFcomputation module and using all the available cores on the computer.
    The results are saved in a json file.

    (c) Adrian Wiltz, 2025
    
"""

if __name__ == '__main__':

    import sys
    import os

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

    import numpy as np
    from Dynamics.SingleIntegrator import SingleIntegrator
    from CBF.CBFmodule import CBFmodule
    import CBF.CBFcomputation as CBFcomputation
    import casadi as ca
    import time

    ########################################################################################
    # Specify the system dynamics, the CBF computation parameters, and initialize the CBF module   

    # some parameters
    num_of_batches_factor = 40      # determines the number of batches for parallel computation
    cbf_file_name = "s2_cbfm.json"

    # create a dynamic system
    x0 = np.array([0,0])
    u_min = np.array([1,-2])   # [minimum speed, minimum steering angle] in [m/s, rad/s]
    u_max = np.array([2,2])     # [maximum speed, maximum steering angle] in [m/s, rad/s]
    mySingleIntegrator = SingleIntegrator(x0=x0,
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

        delta = 0

        return ca.vertcat(h(x) - delta)

    # set parameters for the CBF module
    T = 10
    gamma = 2

    # set domain bounds
    domain_lower_bound = np.array([-10,-10])
    domain_upper_bound = np.array([10,10])
    discretization = np.array([41,41])

    print("Number of grid points to be computed: ", np.prod(discretization))

    # create a CBF module
    myCBFmodule = CBFmodule(h=h, 
                            dynamicSystem=mySingleIntegrator, 
                            cf=cf, 
                            T=T, 
                            N=25,
                            gamma=gamma, 
                            domain_lower_bound=domain_lower_bound, 
                            domain_upper_bound=domain_upper_bound, 
                            discretization=discretization,
                            p_norm=50,
                            p_norm_decrement=10,
                            p_norm_min=40)

    ########################################################################################
    # Initialize the cbf value optimization and compute the cbf value at a selction of sample points

    # 1. Initialize the warm start input trajectories and assign them to the cbf module
    warmStartInputTrajectory_0 = np.array([u_min[0]*np.ones(myCBFmodule.N), 
                                        np.zeros(myCBFmodule.N)])   # max acceleration backward
    warmStartInputTrajectory_1 = np.array([u_max[0]*np.ones(myCBFmodule.N),
                                            np.zeros(myCBFmodule.N)])   # max accelreation forward
    warmStartInputTrajectory_2 = np.array([u_min[0]*np.ones(myCBFmodule.N),
                                            u_min[1]*np.ones(myCBFmodule.N)])   # max acceleration down
    warmStartInputTrajectories_3 = np.array([u_min[0]*np.ones(myCBFmodule.N),
                                            u_max[1]*np.ones(myCBFmodule.N)])   # max acceleration up

    warmStartInputTrajectories = np.array([warmStartInputTrajectory_0, warmStartInputTrajectory_1, warmStartInputTrajectory_2, warmStartInputTrajectories_3])

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


