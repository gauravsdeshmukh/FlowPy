# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 23:20:56 2018

@author: Gaurav
"""

import scipy as sci
import numba as nb
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
############################CLASS SPACE########################################
class Boundary:
    def __init__(self,boundary_type,boundary_value):
        self.DefineBoundary(boundary_type,boundary_value)
        
    def DefineBoundary(self,boundary_type,boundary_value):
        self.type=boundary_type
        self.value=boundary_value

class Space:
    def __init__(self):
        pass
    
    def CreateMesh(self,rowpts,colpts):
        self.rowpts=rowpts
        self.colpts=colpts
        self.u=sci.zeros((self.rowpts+2,self.colpts+2))
        self.v=sci.zeros((self.rowpts+2,self.colpts+2))
        self.p=sci.zeros((self.rowpts+2,self.colpts+2))
        self.r=sci.zeros((self.rowpts+2,self.colpts+2))
        self.SetEmptyMesh()
        
    def SetDeltas(self,breadth,length):
        self.dx=length/self.colpts
        self.dy=breadth/self.rowpts

    def SetEmptyMesh(self):
        self.u_next=sci.zeros((self.rowpts+2,self.colpts+2))
        self.v_next=sci.zeros((self.rowpts+2,self.colpts+2))
        self.p_c=sci.zeros((self.rowpts,self.colpts))
        self.u_c=sci.zeros((self.rowpts,self.colpts))
        self.v_c=sci.zeros((self.rowpts,self.colpts))

class Fluid:
    def __init__(self,rho,mu):
        self.SetFluidProperties(rho,mu)
    
    def SetFluidProperties(self,rho,mu):
        self.rho=rho
        self.mu=mu
        
##########################BOUNDARY SPACE#######################################

def SetUBoundary(space,left,right,top,bottom):
    if(left.type=="D"):
        space.u[:,0]=2*left.value-space.u[:,1]
    elif(left.type=="N"):
        space.u[:,0]=-left.value*space.dx+space.u[:,1]
    
    if(right.type=="D"):
        space.u[:,-1]=2*right.value-space.u[:,-2]
    elif(right.type=="N"):
        space.u[:,-1]=right.value*space.dx+space.u[:,-2]
        
    if(top.type=="D"):
        space.u[-1,:]=2*top.value-space.u[-2,:]
    elif(top.type=="N"):
        space.u[-1,:]=-top.value*space.dy+space.u[-2,:]
     
    if(bottom.type=="D"):
        space.u[0,:]=2*bottom.value-space.u[1,:]
    elif(bottom.type=="N"):
        space.u[0,:]=bottom.value*space.dy+space.u[1,:]
        

def SetVBoundary(space,left,right,top,bottom):
    if(left.type=="D"):
        space.v[:,0]=2*left.value-space.v[:,1]
    elif(left.type=="N"):
        space.v[:,0]=-left.value*space.dx+space.v[:,1]
    
    if(right.type=="D"):
        space.v[:,-1]=2*right.value-space.v[:,-2]
    elif(right.type=="N"):
        space.v[:,-1]=right.value*space.dx+space.v[:,-2]
        
    if(top.type=="D"):
        space.v[-1,:]=2*top.value-space.v[-2,:]
    elif(top.type=="N"):
        space.v[-1,:]=-top.value*space.dy+space.v[-2,:]
     
    if(bottom.type=="D"):
        space.v[0,:]=2*bottom.value-space.v[1,:]
    elif(bottom.type=="N"):
        space.v[0,:]=bottom.value*space.dy+space.v[1,:]
    
def SetPBoundary(space,left,right,top,bottom):
    if(left.type=="D"):
        space.p[:,0]=left.value
    elif(left.type=="N"):
        space.p[:,0]=-left.value*space.dx+space.p[:,1]
    
    if(right.type=="D"):
        space.p[1,-1]=right.value
    elif(right.type=="N"):
        space.p[:,-1]=right.value*space.dx+space.p[:,-2]
        
    if(top.type=="D"):
        space.p[-1,:]=top.value
    elif(top.type=="N"):
        space.p[-1,:]=-top.value*space.dy+space.p[-2,:]
     
    if(bottom.type=="D"):
        space.p[0,:]=bottom.value
    elif(bottom.type=="N"):
        space.p[0,:]=bottom.value*space.dy+space.p[1,:]
    
    
########################FUNCTION SPACE#########################################
def SetTimeStep(CFL,space,fluid):
    if(space.u.any()!=0):
        dt_hyper=CFL/max(sci.amax(space.u)/space.dx,sci.amax(space.v)/space.dy)
    else:
        dt_hyper=CFL*space.dx 
        
    dt_para=min(space.dx**2/(2*fluid.mu),space.dy**2/(2*fluid.mu))
    dt_min=min(dt_hyper,dt_para)
    space.dt=dt_min
 
@nb.jit    
def SolvePressurePoisson(space,fluid,left,right,top,bottom):
    rows=int(space.rowpts)
    cols=int(space.colpts)
    u=space.u.astype(float)
    v=space.v.astype(float)
    p=space.p.astype(float,copy=False)
    dx=float(space.dx)
    dy=float(space.dy)
    dt=float(space.dt)
    rho=float(fluid.rho)
    factor=1/(2/dx**2+2/dy**2)
    
    error=1
    tol=1e-3

    i=0
    while(error>tol):
        i+=1
        p_old=p.astype(float,copy=True)
        
        term_1=(1/dt)*(((v[2:,1:cols+1]-v[0:rows,1:cols+1])/(2*dy))+((u[1:rows+1,2:]-u[1:rows+1,0:cols])/(2*dx)))
        term_2=((u[1:rows+1,2:]-u[1:rows+1,0:cols])/(2*dx))**2
        term_3=((v[2:,1:cols+1]-v[0:rows,1:cols+1])/(2*dy))**2
        term_4=2*((u[2:,1:cols+1]-u[0:rows,1:cols+1])/(2*dy))*((v[1:rows+1,2:]-v[1:rows+1,0:cols])/(2*dx))
        term_5=(p_old[2:,1:cols+1]+p_old[0:rows,1:cols+1])/dy**2+(p_old[1:rows+1,2:]+p_old[1:rows+1,0:cols])/dx**2
        p[1:rows+1,1:cols+1]=factor*term_5-(factor*rho)*(term_1-term_2-term_3-term_4)
        error=sci.amax(abs(p-p_old))
        #Apply Boundary Conditions
        SetPBoundary(space,left,right,top,bottom)
        
        if(i>500):
            tol*=10
            
    
@nb.jit
def SolveMomentumEquation(space,fluid):
    rows=int(space.rowpts)
    cols=int(space.colpts)
    u=space.u.astype(float,copy=False)
    v=space.v.astype(float,copy=False)
    p=space.p.astype(float)
    dx=float(space.dx)
    dy=float(space.dy)
    dt=float(space.dt)
    rho=float(fluid.rho)
    mu=float(fluid.mu)
    u_next=space.u_next.astype(float,copy=False)
    v_next=space.v_next.astype(float,copy=False)

    u1_y=(u[2:,1:cols+1]-u[0:rows,1:cols+1])/(2*dy)
    u1_x=(u[1:rows+1,2:]-u[1:rows+1,0:cols])/(2*dx)
    p1_x=(p[1:rows+1,2:]-p[1:rows+1,0:cols])/(2*dx)
    u2_y=(u[2:,1:cols+1]-2*u[1:rows+1,1:cols+1]+u[0:rows,1:cols+1])/(dy**2)
    u2_x=(u[1:rows+1,2:]-2*u[1:rows+1,1:cols+1]+u[1:rows+1,0:cols])/(dx**2)
    v_face=(v[1:rows+1,1:cols+1]+v[1:rows+1,0:cols]+v[2:,1:cols+1]+v[2:,0:cols])/4
    u_next[1:rows+1,1:cols+1]=u[1:rows+1,1:cols+1]-dt*(u[1:rows+1,1:cols+1]*u1_x+v_face*u1_y)-(dt/rho)*p1_x+(dt*(mu/rho)*(u2_x+u2_y))   

    v1_y=(v[2:,1:cols+1]-v[0:rows,1:cols+1])/(2*dy)
    v1_x=(v[1:rows+1,2:]-v[1:rows+1,0:cols])/(2*dx)
    p1_y=(p[2:,1:cols+1]-p[0:rows,1:cols+1])/(2*dy)
    v2_y=(v[2:,1:cols+1]-2*v[1:rows+1,1:cols+1]+v[0:rows,1:cols+1])/(dy**2)
    v2_x=(v[1:rows+1,2:]-2*v[1:rows+1,1:cols+1]+v[1:rows+1,0:cols])/(dx**2)
    u_face=(u[1:rows+1,1:cols+1]+u[1:rows+1,2:]+u[0:rows,1:cols+1]+u[0:rows,2:])/4
    v_next[1:rows+1,1:cols+1]=v[1:rows+1,1:cols+1]-dt*(u_face*v1_x+v[1:rows+1,1:cols+1]*v1_y)-(dt/rho)*p1_y+(dt*(mu/rho)*(v2_x+v2_y))
            
def AdjustUV(space):
    space.u[1:-1,1:-1]=space.u_next[1:-1,1:-1].copy()
    space.v[1:-1,1:-1]=space.v_next[1:-1,1:-1].copy()
    
def SetCentrePUV(space):
    space.p_c=space.p[1:-1,1:-1]
    space.u_c=space.u[1:-1,1:-1]
    space.v_c=space.v[1:-1,1:-1]
    
def WriteToFile(space,iteration):
    if(iteration%10==0):
        cwdir=os.getcwd()
        if(iteration==0):
            if(os.path.isdir("Result")==False):
                os.mkdir("Result")
                os.chdir("Result")
                cwdir=os.getcwd()
            elif(os.path.isdir("Result")==True):
                os.chdir("Result")
                cwdir=os.getcwd()
                filelist=os.listdir()
                for file in filelist:
                    os.remove(file)
            if(os.path.basename(os.getcwd())!="Result"):
                os.chdir("Result")
            cwdir=os.getcwd()
        filename=f"PUV{iteration}.txt"
        path=os.path.join(cwdir,filename)
        with open(path,"w") as f:
            for i in range(space.rowpts):
                for j in range(space.colpts):
                    f.write("{}\t{}\t{}\n".format(space.p_c[i,j],space.u_c[i,j],space.v_c[i,j]))
     
#################################END OF FILE###################################
