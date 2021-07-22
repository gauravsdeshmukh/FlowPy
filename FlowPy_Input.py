# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 19:25:12 2018

@author: Gaurav
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
from FlowPy import *

###############################################################################
###########################USER INPUT BEGINS###################################
############################DEFINE SPATIAL AND TEMPORAL PARAMETERS#############
length=4
breadth=4
colpts=257
rowpts=257
time=150
###############################MISC############################################
CFL_number=0.8 #Do not touch this unless solution diverges
file_flag=1 #Keep 1 to print results to file
interval=100 #Record values in file per interval number of iterations
plot_flag=1 #Keep 1 to plot results at the end
###########################DEFINE PHYSICAL PARAMETERS##########################
rho=1
mu=0.01
##########################DEFINE INITIAL MOMENTUM PARAMETERS###################
u_in=1
v_wall=0
p_out=0

###############################################################################
########################CREATE SPACE OBJECT####################################
cavity=Space()
cavity.CreateMesh(rowpts,colpts)
cavity.SetDeltas(breadth,length)
water=Fluid(rho,mu)

###############################################################################
#########################BOUNDARY DEFINITIONS##################################
########################CREATE BOUNDARY OBJECTS################################
###########################VELOCITY############################################
flow=Boundary("D",u_in)
noslip=Boundary("D",v_wall)
zeroflux=Boundary("N",0)
############################PRESSURE###########################################
pressureatm=Boundary("D",p_out)

#######################USER INPUT ENDS#########################################
###############################################################################
#############################INITIALIZATION####################################
t=0
i=0
############################THE RUN############################################
print("######## Beginning FlowPy Simulation ########")
print("#############################################")
print("# Simulation time: {0:.2f}".format(time))
print("# Mesh: {0} x {1}".format(colpts,rowpts))
print("# Re/u: {0:.2f}\tRe/v:{1:.2f}".format(rho*length/mu,rho*breadth/mu))
print("# Save outputs to text file: {0}".format(bool(file_flag)))
MakeResultDirectory(wipe=True)

while(t<time):
    sys.stdout.write("\rSimulation time left: {0:.2f}".format(time-t))
    sys.stdout.flush()

    CFL=CFL_number
    SetTimeStep(CFL,cavity,water)
    timestep=cavity.dt
    
    
    SetUBoundary(cavity,noslip,noslip,flow,noslip)
    SetVBoundary(cavity,noslip,noslip,noslip,noslip)
    SetPBoundary(cavity,zeroflux,zeroflux,pressureatm,zeroflux)
    GetStarredVelocities(cavity,water)
    
    
    SolvePressurePoisson(cavity,water,zeroflux,zeroflux,pressureatm,zeroflux)
    SolveMomentumEquation(cavity,water)
    
    SetCentrePUV(cavity)
    if(file_flag==1):
        WriteToFile(cavity,i,interval)

    t+=timestep
    i+=1
    

###########################END OF RUN##########################################
###############################################################################
#######################SET ARRAYS FOR PLOTTING#################################
x=np.linspace(0,length,colpts)
y=np.linspace(0,breadth,rowpts)
[X,Y]=np.meshgrid(x,y)

u=cavity.u
v=cavity.v
p=cavity.p
u_c=cavity.u_c
v_c=cavity.v_c
p_c=cavity.p_c

#Ghia et al. Cavity test benchmark
y_g=[0,0.0547,0.0625,0.0703,0.1016,0.1719,0.2813,0.4531,0.5,0.6172,0.7344,0.8516,0.9531,0.9609,0.9688,0.9766]
u_g=[0,-0.08186,-0.09266,-0.10338,-0.14612,-0.24299,-0.32726,-0.17119,-0.11477,0.02135,0.16256,0.29093,0.55892,0.61756,0.68439,0.75837]

x_g=[0,0.0625,0.0703,0.0781,0.0983,0.1563,0.2266,0.2344,0.5,0.8047,0.8594,0.9063,0.9453,0.9531,0.9609,0.9688]
v_g=[0,0.1836,0.19713,0.20920,0.22965,0.28124,0.30203,0.30174,0.05186,-0.38598,-0.44993,-0.23827,-0.22847,-0.19254,-0.15663,-0.12146]

y_g=[breadth*y_g[i] for i in range(len(y_g))]
x_g=[length*x_g[i] for i in range(len(x_g))]

######################EXTRA PLOTTING CODE BELOW################################
if(plot_flag==1):
    plt.figure(figsize=(20,20))
    plt.contourf(X,Y,p_c,cmap=cm.viridis)
    plt.colorbar()
    plt.quiver(X,Y,u_c,v_c)
    plt.title("Velocity and Pressure Plot")
    
    plt.figure(figsize=(20,20))
    plt.plot(y,u_c[:,int(np.ceil(colpts/2))],"darkblue")
    plt.plot(y_g,u_g,"rx")
    plt.xlabel("Vertical distance along center")
    plt.ylabel("Horizontal velocity")
    plt.title("Benchmark plot 1")
    
    plt.figure(figsize=(20,20))
    plt.plot(x,v_c[int(np.ceil(rowpts/2)),:],"darkblue")
    plt.plot(x_g,v_g,"rx")
    plt.xlabel("Horizontal distance along center")
    plt.ylabel("Vertical velocity")
    plt.title("Benchmark plot 2")
    plt.show()
    
#########################END OF FILE###########################################