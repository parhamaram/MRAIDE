#Author Parham Aram
#Date 12-10-2011
from __future__ import division
import pylab as pb
import numpy as np

class para_state_estimation():

	def __init__(self,model):

		self.model=model



	def estimate_kernel(self,X,P,M):

		"""estimate the ide model's kernel weights from data stored in the ide object"""

		# form Xi variables
		Xi_0 = pb.zeros([self.model.nx,self.model.nx])
		Xi_1 = pb.zeros([self.model.nx,self.model.nx])
		for t in range(1,len(X)):
			Xi_0 += pb.dot(X[t-1,:].reshape(self.model.nx,1),X[t,:].reshape(self.model.nx,1).T) + M[t,:].reshape(self.model.nx,self.model.nx)
			Xi_1 += pb.dot(X[t-1,:].reshape(self.model.nx,1),X[t-1,:].reshape(self.model.nx,1).T) + P[t-1,:].reshape(self.model.nx,self.model.nx)

		# form Upsilon and upsilons
		Upsilon = pb.zeros([self.model.ntheta,self.model.ntheta])
		upsilon0 = pb.zeros([1,self.model.ntheta])
		upsilon1 = pb.zeros([1,self.model.ntheta])
		for i in range(self.model.nx):
			for j in range(self.model.nx):
				Upsilon += Xi_1[i,j] * self.model.Delta_Upsilon[j,i]
				upsilon0 += Xi_0[i,j]*self.model.Delta_upsilon[j,i]
				upsilon1 += Xi_1[i,j]*self.model.Delta_upsilon[j,i]
		upsilon1=upsilon1*self.model.xi
		Upsilon=Upsilon*self.model.Ts*self.model.varsigma

		

		weights = pb.dot(pb.inv(Upsilon.T),upsilon0.T-upsilon1.T)


		return weights

#define matrix multiplication
def dots(*args):
	lastItem = 1.
	for arg in args:
		lastItem = pb.dot(lastItem, arg)
	return lastItem		


