# FlowPy
A 2D single-phase finite difference fluid flow solver written in python with numpy vectorization for fast performance. Currently it supports single phase flow in the laminar regime. 

The Navier-Stokes equations and the continuity equation are discreitzed using a second order finite difference method and solved numerically. To ensure a divergence free velocity at the end of each time step, the process of calculating velocity and pressure is as follows:
1. Calculate intermediate starred velocities that may not necessarily be divergence free by solving the momentum equation without the pressure term. This is the predictor step.
2. Differentitate the momentum equation (with the pressure term) and apply continuity to eliminate next time-step velocities. Thus, we are left with a Poisson equation for pressure in terms of the starred velocities. This step gives the pressure field for the next time-step.
3. Calculate divergence-free velocities for the next time-step using the newly calculated pressure field and the starred velocities in a corrector step. 

The program requires 2 files to work: FlowPy.py and FlowPy_Input.py. The first file contains all the classes and functions utilized in the main run. The second file contains all the parameter inputs, object definitions and the main run which calls all the requisite functions.

This program has been benchmarked against the Lid Cavity Test results from Ghia et al (1982).

Currently, two different flow domains have been tested on this program: the lid cavity and Poiseuille flow. 

There is a provision to print the resultant pressures and velocities at each time instant in separate files which is created in a new Results folder (automatically created). The files in that folder can be read by using the ReadAndPlot.py file provided.

The codebase for this was developed by referring partly to Prof. Lorena Barba’s (Dept. of Mechanical and Aerospace Engineering, George Washington University) excellent tutorial titled “12 steps to Navier-Stokes” and partly to Prof. Mark Owkes’ (Dept. of Mechanical and Industrial Engineering, Montana State University) guide titled “A guide to writing your first CFD solver”.



