#Author Parham Aram
#Date 12-10-2011
'''This module simulates the  neural field model and uses the generated data
in the state-space model to estimate states and 
the connectivity kernel parameters.'''

#Standard library imports
#~~~~~~~~~~~~~~~~~~~~~~~~
from __future__ import division
import pylab as pb
import numpy as np
import scipy as sp
#My modules
#~~~~~~~~~~~~~~~~~~~~
import IDEComponents
import BsplineBases
from MRIDE import *
from NF import *

dimension=1
spacewidth=8
#distance between adjacent sensors
Delta_s=0.05
obs_locns=pb.arange(-spacewidth/2.,spacewidth/2.+Delta_s,Delta_s)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
sensor_kernel=scale(4,-2)
#time properties
#~~~~~~~~~~~~~~~
#sampling rate  
Fs =1e3                                       
#sampling period
Ts = 1/Fs   
t_end= 1
NSamples = t_end*Fs
T = pb.linspace(0,t_end,NSamples)
#Define synaptic time constant
zeta=100
#space properties
#~~~~~~~~~~~~~~~~
spacestep=0.01
simulation_space=pb.arange(-spacewidth/2.,spacewidth/2.+spacestep,spacestep)	
#Define connectivity kernel for data generation
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
NF_k_1=pb.array([BsplineBases.scale(j,l-3) for j in [1] for l in [1]],ndmin=2).T 
NF_k_0=pb.array([BsplineBases.scale(j,l-3) for j in [0] for l in [1]],ndmin=2).T

NF_kernel_basis_functions=pb.concatenate([NF_k_1,NF_k_0])
NF_kernel_weights = pb.array([[200,-100]]).T



NF_kernel=IDEComponents.Kernel(NF_kernel_weights,NF_kernel_basis_functions)
#define connectivity kernel for estimation (reduced model)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MRIDE_kernel_basis_functions_scale=pb.array([scale(j,l-3) for j in [1] for l in [-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7]],ndmin=2).T
MRIDE_kernel_basis_functions_wavelet=pb.array([wavelet(j,l-3) for j in [1] for l in [-6,-5,-4,-3,-2,-1,0,1,2,3,4,5]],ndmin=2).T

MRIDE_kernel_basis_functions=pb.concatenate([MRIDE_kernel_basis_functions_scale,MRIDE_kernel_basis_functions_wavelet])

MRIDE_kernel_weights=pb.ones_like(MRIDE_kernel_basis_functions)

MRIDE_kernel=IDEComponents.Kernel(MRIDE_kernel_weights,MRIDE_kernel_basis_functions)


#Define field basis functions at each level such that they cover the spatial domain of interest.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#level0
f_1=pb.array([scale(j,l) for j in [0] for l in range(-6,3)],ndmin=2).T
f_2=pb.array([wavelet(j,l) for j in [0] for l in range(-7,1)],ndmin=2).T
#level1
f_3=pb.array([wavelet(j,l) for j in [1] for l in range(-11,5)],ndmin=2).T
#level2
f_4=pb.array([wavelet(j,l) for j in [2] for l in range(-19,13)],ndmin=2).T
#level3
f_5=pb.array([wavelet(j,l) for j in [3] for l in range(-36,30)],ndmin=2).T
#level4
#f_6=pb.array([wavelet(j,l) for j in [4] for l in range(-69,63)],ndmin=2).T



field_basis_functions=pb.concatenate([f_1,f_2,f_3,f_4,f_5])
field=IDEComponents.Field(f_1,f_2,f_3,f_4,f_5)
#define disturbance covariance function
#~~~~~~~~~~~~~~~~~~~~
eta=scale(3,-2)
eta_weight=1.*(1./eta(0))
#define observation noise
#~~~~~~~~~~~~~~~~~~~~~~~~~~
ny=len(obs_locns)
Sigma_epsilon=.1*pb.eye(ny,ny)
#~~~~~~~~~~~~~~~~~~~~~~~~
#firing rate slope
#~~~~~~~~~~~~~~~~~~~~~~~~
varsigma=0.56 

#inisialization
#~~~~~~~~~~~~~~~~~~~~~~~~~~
mean=[0]*len(field_basis_functions)
P0=10*pb.eye(len(mean))
x0=pb.multivariate_normal(mean,P0,[1]).T
NF_model=NF(NF_kernel,sensor_kernel,obs_locns,eta,eta_weight,Sigma_epsilon,Ts,zeta,varsigma,simulation_space,Delta_s,spacestep)
NF_model.gen_ssmodel()
V,Y=NF_model.simulate(T)
MRIDE_model=IDE(MRIDE_kernel,field,sensor_kernel,obs_locns,eta,eta_weight,Sigma_epsilon,Ts,zeta,varsigma,x0,P0)
MRIDE_model.gen_ssmodel()
#perform the state and parameter estimation for 20 iterations, ignoring first 100 observations
First_n_observations=100
number_of_iterations=20
ps_estimate=para_state_estimation(MRIDE_model)
ps_estimate.itrerative_state_parameter_estimation(Y[First_n_observations:],number_of_iterations)

