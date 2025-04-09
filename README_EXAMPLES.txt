README on the provided examples
===============================

The implementation of the predicitive CBF synthesis scheme is accompanied by examples. The examples illustrate the applicability of the proposed approach on the one hand, but is also intended to show how the implementation is practically applied. 

This brief note shall give an overview on the provided examples. 

The example folder contains four subfolders:

-- S1: Examples for the single integrator. 

    * The computation of a CBF for input constrained single integrators with respect a circular obstacle is demonstrated (see files "cbf_s1.py" and "cbf_s2.py").  

    * The computation of a CBF value at a particular sampling point is demonstrated exemplarily in "test_cbf_computation_at_point.py". 

    * Simulation examples apply the computed CBFs in a set up with multiple static circular obstacles (see "sim_multi_circ_static_combined.py") and with multiple time-varying circular obstacles (see "sim_multi_circ_time_varying_combined.py"). 

    * A script for visualizing the computed CBFs and analyzing various of its features is provided in "visualize_cbf.py". 

    * Videos illustrate the simulation results presented in the paper with animations.

    * The folder "Data" contains precomputed CBFs. 


-- SC: Examples for single and double integrators in the presence of a single obstacle. The included files illustrate the application of the toolbox in a basic setting. 

    * Simulation examples apply the computed CBFs in a set up with a single static circular obstacle (see "sim_single_circ_static.py") and with a single time-varying circular obstacle (see "sim_single_circ_time_varying.py"). For scripts on the computation of the employed CBFs, refer to the other folders. 

    * Videos illustrate the simulation results presented in the paper with animations.

    * The folder "Data" contains precomputed CBFs. 

-- B1: Examples for the bicycle model. 

    * The computation of a CBF for the input constrained kinematic bicycle model with respect a circular obstacle is demonstrated (see files "cbf_b1.py" for the less agile bicycle and "cbf_b2.py" for the more agile bicycle).

    * The computation of a CBF value at a particular sampling point is demonstrated exemplarily in "test_cbf_computation_at_point.py". 

    * Simulation examples apply the computed CBFs in a set up with multiple static circular obstacles (see "sim_multi_circ_static_combined.py") and with multiple time-varying circular obstacles (see "sim_multi_circ_time_varying_combined.py"). 

    * A script for visualizing the computed CBFs and analyzing various of its features is provided in "visualize_cbf.py". 

    * Videos illustrate the simulation results presented in the paper with animations.

    * The folder "Data" contains precomputed CBFs. 

-- U1: Examples for the unicycle model.

    * The computation of a CBF for input constrained unicycle model with respect a circular obstacle is demonstrated (see file "cbf_u1.py").

    * Simulation examples apply the computed CBFs in a set up with multiple static circular obstacles (see "sim_multi_circ_static.py") and with multiple time-varying circular obstacles (see "sim_multi_circ_time_varying.py"). 

    * Videos illustrate the simulation results presented in the paper with animations.

    * The folder "Data" contains precomputed CBFs. 


These examples demonstrate how to use the main functionality of the toolbox and guide through its application. 