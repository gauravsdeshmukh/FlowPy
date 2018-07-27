# FlowPy
A fluid flow solver written in python with a highly object oriented approach. Currently it supports single phase flow in the laminar regime. 

The program requires 2 files to work: FlowPy.py and FlowPy_Input.py. The first file contains all the classes and functions utilized in the main run. The second file contains all the parameter inputs, object definitions and the main run which calls all the requisite functions.

This program has been benchmarked against the Lid Cavity Test results from Ghia et al (1982).

Currently, two different flow domains have been tested on this program: the lid cavity and Poiseuille flow. 

There is a provision to print the resultant pressures and velocities at each time instant in a separate file which is created in a new Results folder (automatically created). The files in that folder can be read by using the ReadAndPlot.py file provided.
