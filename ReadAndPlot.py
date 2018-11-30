# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 22:33:14 2018

@author: Gaurav
"""

import scipy as sci
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation


os.chdir("Result")
cwdir=os.getcwd()

final_iter=3995
inter=10
rowpts=200
colpts=200
length=1
breadth=1

number_of_frames=int(final_iter/inter)+1

x=sci.linspace(0,length,colpts)
y=sci.linspace(0,breadth,rowpts)
[X,Y]=sci.meshgrid(x,y)
fig=plt.figure(figsize=(16,8))
ax=plt.axes(xlim=(0,length),ylim=(0,breadth))

#for i in range(0,final_iter+inter,inter):
#    arr=sci.loadtxt(f"PUV{i}.txt",delimiter="\t")
#    rows,cols=arr.shape
#    p_p=sci.zeros((rowpts,colpts))
#    u_p=sci.zeros((rowpts,colpts))
#    v_p=sci.zeros((rowpts,colpts))
#    p_arr=arr[:,0]
#    u_arr=arr[:,1]
#    v_arr=arr[:,2]
#    
#    p_p=p_arr.reshape((rowpts,colpts))
#    u_p=u_arr.reshape((rowpts,colpts))
#    v_p=v_arr.reshape((rowpts,colpts))

    
def animate(i):
    filename=f"PUV{inter*i}.txt"
    path=os.path.join(cwdir,filename)
    arr=sci.loadtxt(path,delimiter="\t")
    rows,cols=arr.shape
    p_p=sci.zeros((rowpts,colpts))
    u_p=sci.zeros((rowpts,colpts))
    v_p=sci.zeros((rowpts,colpts))
    p_arr=arr[:,0]
    u_arr=arr[:,1]
    v_arr=arr[:,2]
    
    p_p=p_arr.reshape((rowpts,colpts))
    u_p=u_arr.reshape((rowpts,colpts))
    v_p=v_arr.reshape((rowpts,colpts))
    
    ax.clear()
    ax.set_xlim([0,length])
    ax.set_ylim([0,breadth])
    ax.set_title(f"Frame No: {i}")
    cont=ax.contourf(X,Y,p_p)
    stream=ax.streamplot(X,Y,u_p,v_p,color="k")
    return cont,stream

anim=animation.FuncAnimation(fig,animate,frames=number_of_frames,interval=50,blit=False)
anim.save("LidCavity_Benchmark.mp4")