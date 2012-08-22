#Author Parham Aram
#Date 12-10-2011
"""
This module provides the full neural field class, which describes a linear
integro-difference equation model and methods to simulate
the model
"""

#Standard library imports
from __future__ import division
import pylab as pb
import numpy as np
import time
import os
from scipy import signal
import scipy as sp

#My modules
from BsplineBases import *


class NF():

	"""class defining full neural model.

	Arguments
	----------
	kernel: 
			connectivity kernel.

	sensor_kernel:

			output kernel, governs the sensor pick-up geometry
		
	obs_locns: ndarray
				sensors locations

	eta : B-spline function
				covariance function of the field disturbance
	eta_weight: float
				amplitude of the field disturbance covariance function

	Sigma_epsilon: ndarray
				observation noise covariance

	Ts: float
				sampling step


	zeta: float
				inverse synaptic time constant


	simulation_space: ndarray
				spatiel field

	spacestep: float
			spatial step size for descritization 


	
	Attributes
	----------
	ny: int
		number of sensors

	xi: float
		time constant parameter

	gen_ssmodel:
		generate the full neural field model

	simulate:
		simulate the full neural field model
	"""



	def __init__(self,kernel,sensor_kernel,obs_locns,eta,eta_weight,Sigma_epsilon,Ts,zeta,varsigma,simulation_space,sensorspace,spacestep):

		self.kernel = kernel
		self.sensor_kernel=sensor_kernel
		self.obs_locns=obs_locns
		self.eta=eta
		self.eta_weight=eta_weight
		self.Sigma_epsilon=Sigma_epsilon
		self.Ts=Ts
		self.zeta=zeta
		self.varsigma=varsigma
		self.simulation_space=simulation_space
		self.sensorspace=sensorspace
		self.spacestep=spacestep
		self.ny=len(self.obs_locns)
		self.ntheta=len(self.kernel.Lambda)
		self.xi=1-self.zeta*self.Ts


	def gen_ssmodel(self):


		"""
		generates full neural model

		Attributes:
		----------
		K: ndarray
			array of connectivity kernel evaluated over the spatial domain of the kernel

		Sigma_e: ndarray
			field disturbance covariance matrix
		Sigma_e_c: ndarray
			Cholesky decomposiotion of field disturbance covariance matrix
		Sigma_epsilon_c: ndarray
			cholesky decomposiotion of observation noise covariance matrix
		C: ndarray
			matrix of sensors evaluated at each spatial location
		"""
        
		print "generating full neural model"


		K=0
		for i in range(len(self.kernel.Lambda)):
			K_temp=pb.vectorize(self.kernel.Lambda[i,0].__call__)
			K+=self.kernel.weights[i]*K_temp(self.simulation_space)

		K.shape=K.shape[0],1
		self.K=K

		#calculate field disturbance covariance matrix and its Cholesky decomposition
		#initialisation

		if hasattr(self,'Sigma_e'):
			pass
		else:

		
			Sigma_e=pb.zeros([self.simulation_space.shape[0],self.simulation_space.shape[0]])

			offset=pb.absolute(self.eta.Supp()[1]-self.eta.Supp()[0])/2. 			
			eta_translation=(2**self.eta.j)*pb.arange(-pb.absolute(self.simulation_space[0])-offset,(pb.absolute(self.simulation_space[0])-offset)+self.spacestep,self.spacestep)

			for m,n in enumerate(eta_translation):
				eta_temp=pb.vectorize(scale(self.eta.j,n).__call__)
				Sigma_e[m]=eta_temp(self.simulation_space)
			
			
		
			self.Sigma_e=self.eta_weight*Sigma_e


		if hasattr(self,'Sigma_e_c'):
			pass
		else:
			self.Sigma_e_c=sp.linalg.cholesky(self.Sigma_e,lower=1)    

		#calculate Cholesky decomposition of observation noise covariance matrix
		Sigma_epsilon_c=sp.linalg.cholesky(self.Sigma_epsilon,lower=1)
		self.Sigma_epsilon_c=Sigma_epsilon_c

		#Calculate sensors at each spatial locations
		C=pb.zeros([self.ny,self.simulation_space.shape[0]])
		offset=pb.absolute(self.sensor_kernel.Supp()[1]-self.sensor_kernel.Supp()[0])/2. 
		
		sensor_kernel_translation=(2**self.sensor_kernel.j)*pb.arange(-pb.absolute(self.obs_locns[0])-offset,(pb.absolute(self.obs_locns[0])-offset)+self.sensorspace,self.sensorspace)
		for m,n in enumerate(sensor_kernel_translation):
			sensor_kernel_temp=pb.vectorize(scale(self.sensor_kernel.j,n).__call__)	
			C[m]=sensor_kernel_temp(self.simulation_space)
		self.C=C


	def simulate(self,T):

		"""Simulates the full neural field model

		Arguments
		----------

		T: ndarray
				simulation time instants
		Returns
		----------
		V: list of ndarray
			each ndarray is the neural field at a time instant

		Y: list of ndarray
			each ndarray is the observation vector corrupted with noise at a time instant
		"""

		Y=[]
		V=[]  
		spatial_location_num=(len(self.simulation_space))


		#initial field
		v_membrane=pb.dot(self.Sigma_e_c,np.random.randn(spatial_location_num,1))

		for t in T[1:]:
			v = pb.dot(self.Sigma_epsilon_c,np.random.randn(self.ny,1))
			w = pb.dot(self.Sigma_e_c,np.random.randn(spatial_location_num,1))

			g=signal.fftconvolve(self.K,v_membrane,mode='same') 
			g*=(self.spacestep)
			v_membrane=self.Ts*self.varsigma*g +self.xi*v_membrane+w
			#Observation
			Y.append((self.spacestep)*(pb.dot(self.C,v_membrane))+v)
			V.append(v_membrane)

		return V,Y



