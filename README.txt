README
======

What is the code about
======================

The code povides an elaborate implementation of the predictive CBF synthesis scheme proposed in:

Adrian Wiltz, Dimos V. Dimarogonas. "Predictive Synthesis of Control Barrier Functions and its Application to Time-Varying Constraints", 2025.

The toolbox allows for the parallelized computation of CBFs that can accound for changing and time-varying constraints. The CBFs can be computed for any input constraint system. For details, please refer to the paper. 

If you find this code useful, please reference the above paper. 


Prerequistes:
=============
-- Casadi, at least version 3.6.0 recommended (optimal control toolbox)
-- dask, recommended version 2023.11.0 (parallel computation toolbox)

other used toolboxes are standard


Structure of the code:
======================

The repository contains the following subfolders:

-- Auxiliaries: implementations of various auxiliary functions used through out the implementation

-- CBF: core module for computing CBFs, parallelizing the CBF computation, initialization of the CBF computation and modules with functionalities for saving, loading and handling computed CBFs. The CBF modules also save all settings and constraint specifications relevant to the computation of the CBF. 

-- Controller: based on the computed CBFs; computes control inputs that are safe with respect to the constraints encoded in the CBFs. 

-- Dynamics: provides system dynamics in the format required for the CBF computation and the computation of safe control inputs. The folder also contains functionalities for simulating dynamic systems under given control input trajectories or a feedback control law. Each of the provided dynamic systems is equipped with a basic line following controller. 

-- Examples: application examples; illustrate the application of the toolbox. 


How to get started
==================

The examples in the folder "Examples" provide a good starting point to get acquainted with the functionalities provided. For details, refer to the file README_EXAMPLES.txt


Further documentation
=====================

Further documentation is provided via comments directly in the code. An elaborate documentation is given in the beginning of each class and function. 


Related paper
=============

Adrian Wiltz, Dimos V. Dimarogonas. "Predictive Synthesis of Control Barrier Functions and its Application to Time-Varying Constraints", 2025. 