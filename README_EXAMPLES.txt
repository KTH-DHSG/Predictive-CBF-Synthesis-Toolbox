README on the provided examples
===============================

The implementation of the predicitive CBF synthesis scheme is accompanied by examples. The examples illustrate the applicability of the proposed approach on the one hand, but is also intended to show how the implementation is practically applied. 

This brief note shall give an overview on the provided examples. 

The example folder contains four subfolders:


-- SC: Examples for single and double integrators in the presence of a single obstacle. The included files illustrate the application of the toolbox in a basic setting. 

    * The computation of a CBF for input constrained single and double integrators with respect a circular obstacle is demonstrated (see files "cbf_s1.py", "cbf_s2.py" and "cbf_d1.py").  

    * Simulation examples apply the computed CBFs in a set up with a single static circular obstacle (see "sim_single_circ_static.py") and with a single time-varying circular obstacle (see "sim_single_circ_time_varying.py"). For scripts on the computation of the employed CBFs, refer to the other folders. 

    * Videos illustrate the simulation results presented in the paper with animations.

    * A script for visualizing the computed CBFs and analyzing various of its features is provided in "visualize_cbf.py". 

    * The folder "Data" contains precomputed CBFs. 

-- B1: Examples for the bicycle model. 

    * The computation of a CBF for the input constrained kinematic bicycle model with respect a circular obstacle is demonstrated (see file "cbf_b1.py").

    * Simulation examples apply the computed CBFs in a set up with multiple static circular obstacles (see "sim_multi_circ_static_campaign.py") and with multiple time-varying circular obstacles (see "sim_multi_circ_tv_campaign.py"). 

    * A script for visualizing the computed CBFs and analyzing various of its features is provided in "visualize_cbf.py". 

    * Videos illustrate the simulation results presented in the paper with animations.

    * The folder "Data" contains precomputed CBFs. 

-- U1: Examples for the unicycle model.

    * The computation of a CBF for input constrained unicycle model with respect a circular obstacle is demonstrated (see file "cbf_u1.py").

    * Simulation examples apply the computed CBFs in a set up with multiple static circular obstacles (see "sim_multi_circ_static_campaign.py") and with multiple time-varying circular obstacles (see "sim_multi_circ_tv_campaign.py"). 

    * Videos illustrate the simulation results presented in the paper with animations.

    * The folder "Data" contains precomputed CBFs. 

-- Watertank_quadruple: Examples for the interconnected tank system with coupling constraint function.

    * The computation of a CBF for the interconnected tank system with respect a constraint function coupling the fill levels of the various tanks is demonstrated (see file "b_watertank.py").

    * Simulation example applying the computed CBF to a setting with time-varying constraint (see "sim_watertank.py"). 

    * The folder "Data" contains precomputed CBFs. 

 -- Watertank_quadruple: Examples for the interconnected tank system with coupling constraint function.

    * The computation of a CBF for the interconnected tank system with respect a constraint function coupling the fill levels of the various tanks is demonstrated (see file "b_watertank.py").

    * Simulation example applying the computed CBF to a setting with time-varying constraint (see "sim_watertank.py"). 

    * The folder "Data" contains precomputed CBFs. 

 
These examples demonstrate how to use the main functionality of the toolbox and guide through its application. 