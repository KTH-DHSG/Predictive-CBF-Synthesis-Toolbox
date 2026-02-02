"""

    Example code for computing a Control Barrier Function (CBF) for a quadruple water tank system.

    (c) Adrian Wiltz, 2025
    
"""

from sympy import zeta


if __name__ == '__main__':

    import sys
    import os

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

    import numpy as np
    from Dynamics.WaterTank_quadruple import WaterTank
    from CBF.CBFmodule import CBFmodule
    import CBF.CBFcomputation as CBFcomputation
    import casadi as ca
    import time

    ########################################################################################
    
    # some parameters
    num_of_batches_factor = 40      # determines the number of batches for parallel computation
    cbf_file_name = "b_watertank.json"

    # create a dynamic system
    t0 = 0
    x0 = np.array([5,5,5,5])

    h_max = np.infty

    box_constraints_min = np.array([0,0,0,0])   # [minimum steering angle in rad, maximum steering angle in rad, minimum position in m, minimum velocity in m/s, minimum steering angle rate in rad/s]
    box_constraints_max = np.array([h_max, h_max, h_max, h_max])      # [maximum steering angle in rad, maximum steering angle in rad, maximum position in m, maximum velocity in m/s, maximum steering angle rate in rad/s]

    A1 = 20
    a1 = 0.6
    A2 = 20
    a2 = 0.4
    A3 = 20
    a3 = 0.4
    A4 = 20
    a4 = 0.4

    u_min = np.array([0.0,0.0])
    u_max = np.array([20, 10])

    myWaterTank = WaterTank(x0=x0,
                    A_1=A1,
                    a_1=a1,
                    A_2=A2,
                    a_2=a2,
                    A_3=A3,
                    a_3=a3,
                    A_4=A4,
                    a_4=a4,
                    u_min=u_min,
                    u_max=u_max)

    #create a state constraint function

    def h(x):
        """State constraint function that supports numpy and casadi types"""
        smoothness = 10.0
        h_max = 10.0

        smoothmin = -1/smoothness * ca.log(ca.exp(-smoothness*2.0*x[0]) + ca.exp(-smoothness*2.0*x[1]) + ca.exp(-smoothness*x[2]) + ca.exp(-smoothness*x[3]))

        return smoothmin - h_max + 0.1

    # create a terminal constraint function
    def cf(x):
        """Terminal constraint function for casadi type arguments"""

        A1 = 20
        a1 = 0.6
        A2 = 20
        a2 = 0.4
        A3 = 20
        a3 = 0.4
        A4 = 20
        a4 = 0.4

        h_min = 10.0

        return ca.vertcat(2.0*x[0]-h_min,
                        2.0*x[1]-h_min,
                        x[2]-h_min,
                        x[3]-h_min,
                        a1**2/A1**2 * x[0] - a3**2/A3**2 * x[2],
                        1.5**2 * a2**2/A2**2 * x[1] - a4**2/A4**2 * x[3])

    # set parameters for the CBF module
    T = 60
    def alpha_bar(zeta):
        """Class K function for encoding the CBF condition into the CBF construction. Specify the class K function for the CBF condition here as alpha."""

        gamma = 0.1

        sigmoid_func = 2*gamma*(1/(1 + ca.exp(-10.0*zeta)) - 0.5)

        if isinstance(zeta, (ca.SX, ca.MX, ca.DM)):
            return ca.if_else(zeta <= 0, 0, 0)
        else:
            # Normal Python numeric behavior
            return sigmoid_func if zeta <= 0 else 0

    # set domain bounds
    domain_lower_bound = np.array([1.0,1.0,2.0,2.0])
    domain_upper_bound = np.array([12.0,12.0,12.0,12.0])
    discretization = np.array([22,22,21,21])

    print("Number of grid points to be computed: ", np.prod(discretization))

    # warm start feedback controller (LQR based)
    def u_ws(x):
        """Warm start feedback controller for casadi type arguments"""

        a1 = 0.6
        a2 = 0.6
        u_max = np.array([20,10])

        u = u_max

        return u

    # create a CBF module
    myCBFmodule = CBFmodule(h=h, 
                            dynamicSystem=myWaterTank, 
                            box_constraints_min=box_constraints_min,
                            box_constraints_max=box_constraints_max,
                            cf=cf, 
                            T=T, 
                            N=30,
                            alpha_bar=alpha_bar, 
                            domain_lower_bound=domain_lower_bound, 
                            domain_upper_bound=domain_upper_bound, 
                            discretization=discretization,
                            p_norm=40,
                            p_norm_decrement=10,
                            p_norm_min=40,
                            warm_start_controller=u_ws)

    dt = myCBFmodule.dt
    
    ########################################################################################
    
    # 1. Compute the CBF on the domain
    tic = time.time()
    CBFcomputation.computeCbfParallelized(myCBFmodule, processes=None, timeout_per_sample=1000, num_of_batches_factor=num_of_batches_factor)
    toc = time.time()
    print("CBF computation took ", toc-tic, " seconds.")

    # 2. Save the CBF module to a file
    myCBFmodule.cbf.computation_time = toc-tic

    # 3. Save the CBF module to a file
    myCBFmodule.save(cbf_file_name, folder_name="Data")

    print("CBF computation finished.")


