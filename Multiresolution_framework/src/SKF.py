#Author Parham Aram
#Date 12-10-2011
# built-ins
from __future__ import division
import numpy as np
import pylab as pb
import scipy as sp
import copy


class SKF():


	''' class defining the RTS smoother.
        
        
        
        
		Arguments:
		----------
		x0: ndarray
			initial state
		P0: ndarray
			initial state covariance matrix
		A: ndarray
			transition matrix
		C: ndarray
			observation matrix

		Sigma_e: ndarray
			process noise covariance matrix

		Sigma_epsilon: ndarray
			observation noise covariance matrix

		
'''




	def __init__(self,x0,P0,A,C,Sigma_e,Sigma_epsilon):
		
		assert 	type(x0) is np.ndarray, 'Initial state must be ndarray'
		assert 	type(P0) is np.ndarray, 'Initial state covariance matrix must be ndarray'
		assert 	type(A) is np.ndarray, 'Transition matrix must be ndarray'
		assert 	type(C) is np.ndarray, 'Observation matrix must be ndarray'
		assert 	type(Sigma_e) is np.ndarray, 'process noise covariance matrix must be ndarray'
		assert 	type(Sigma_epsilon) is np.ndarray, 'observation noise covariance matrix must be ndarray'

		self.x0=x0
		self.P0=P0
		self.A=A
		self.C=C
		self.Sigma_e=Sigma_e
		self.Sigma_epsilon=Sigma_epsilon
		self.ny,self.nx=self.C.shape
		



	def kfilter(self,Y):
		
		
		"""Standard Kalman Filter
		
		Arguments
		----------
		Y : list of ndarray
			A list of observation vectors
			
		Returns
		----------	
		X : list of ndarray
			A list of state estimates
		P : list of ndarray
			A list of state covariance matrices

		"""
		
		
		# prediction
        
		def Kpred(A,P,Sigma_e,x):
					
			if (np.rank(A)!=2 or type(self.A)!=np.ndarray):raise TypeError('A should be rank two array')
			if (np.rank(Sigma_e)!=2 or type(self.Sigma_e)!=np.ndarray):raise TypeError('Sigma_e should be rank two array')
			if (np.rank(P)!=2 or type(self.Sigma_e)!=np.ndarray):raise TypeError('P should be rank two array')
			if type(x)!=np.ndarray:raise TypeError('x0 should be rank two array')
			
			# predict state
			x_ = pb.dot(A,x) 
			# predict state covariance matrix
			P_ =dots(A,P,A.T) + Sigma_e
			return x_,P_

		# correction
		def Kupdate(A,C,P_,Sigma_epsilon,x_,y):
								
			if (np.rank(A)!=2 or type(self.A)!=np.ndarray):raise TypeError('A should be rank two array')
			if (np.rank(C)!=2 or type(self.C)!=np.ndarray):raise TypeError('C should be rank two array')
			if (np.rank(Sigma_epsilon)!=2 or type(self.Sigma_epsilon)!=np.ndarray):raise TypeError('Sigma_epsilon should be rank two array')
			if (np.rank(P_)!=2 or type(P_)!=np.ndarray):raise TypeError('P should be rank two array')
			if type(x_)!=np.ndarray:raise TypeError('x0 should be rank two array')
			if type(y)!=np.ndarray:raise TypeError('y should be rank two array')

			
			# calculte Kalman gain
			K =pb.dot(pb.dot(P_,C.T),pb.inv(dots(C,P_,C.T) + Sigma_epsilon))
			# update estimate with model output measurement
			x = x_ + pb.dot(K,(y-pb.dot(C,x_)))
			# update the state error covariance
			P = pb.dot((np.eye(self.nx)-pb.dot(K,C)),P_);
			return x,P

		# filter quantities
		XStore = []
		PStore = []
		# initialise the filter
		x_,P_ = Kpred(self.A,self.P0,self.Sigma_e,self.x0)
		# filter
		for y in Y :
			# update state estimate using measurement and Kalman gain
			x,P= Kupdate(self.A,self.C,P_,self.Sigma_epsilon,x_,y)
			# store corrected values of state and covariance matrix
			XStore.append(x)
			PStore.append(P)
			# predict new state
			x_, P_ = Kpred(self.A,P,self.Sigma_e,x);
			
	
		return XStore, PStore
	
	def rtssmooth(self,Y):
		
		
		"""Rauch Tung Streibel(RTS) smoother
		
		
		Arguments
		----------
		Y : list of ndarray
			 list of observation vectors
			
		Returns
		----------	
		XStore : list of ndarray
			 list of forward state estimates
		PStore : list of ndarray
			 list of forward state covariance matrices
			
		XbStore : list of ndarray
			 list of backward state estimates
		PbStore : list of ndarray
			 list of backward state covariance matrices	

		"""
		
		
		
		# prediction
		def Kpred(A,P,Sigma_e,x):
					
			if (np.rank(A)!=2 or type(self.A)!=np.ndarray):raise TypeError('A should be rank two array')
			if (np.rank(Sigma_e)!=2 or type(self.Sigma_e)!=np.ndarray):raise TypeError('Sigma_e should be rank two array')
			if (np.rank(P)!=2 or type(self.Sigma_e)!=np.ndarray):raise TypeError('P should be rank two array')
			if type(x)!=np.ndarray:raise TypeError('x0 should be rank two array')
			
			# predict state
			x_ = pb.dot(A,x) 
			# predict state covariance
			P_ =dots(A,P,A.T) + Sigma_e
			return x_,P_

		# correction
		def Kupdate(A,C,P_,Sigma_epsilon,x_,y):
								
			if (np.rank(A)!=2 or type(self.A)!=np.ndarray):raise TypeError('A should be rank two array')
			if (np.rank(C)!=2 or type(self.C)!=np.ndarray):raise TypeError('C should be rank two array')
			if (np.rank(Sigma_epsilon)!=2 or type(self.Sigma_epsilon)!=np.ndarray):raise TypeError('Sigma_epsilon should be rank two array')
			if (np.rank(P_)!=2 or type(P_)!=np.ndarray):raise TypeError('P should be rank two array')
			if type(x_)!=np.ndarray:raise TypeError('x0 should be rank two array')
			if type(y)!=np.ndarray:raise TypeError('y should be rank two array')

			
			# calculte Kalman gain
			
			K =pb.dot(pb.dot(P_,C.T),pb.inv(dots(C,P_,C.T) + Sigma_epsilon))
			# update estimate with model output measurement
			x = x_ + pb.dot(K,(y-pb.dot(C,x_)))
			# update the state error covariance
			P = pb.dot((np.eye(self.nx)-pb.dot(K,C)),P_);
			return x,P,K

		# smoother quantities
		XStore_ =pb.zeros([len(Y),self.nx])
		PStore_ =pb.zeros([len(Y),self.nx**2])
		XStore = pb.zeros([len(Y),self.nx])
		PStore = pb.zeros([len(Y),self.nx**2])
		x_,P_ = Kpred(self.A,self.P0,self.Sigma_e,self.x0)
		# filter
		for i in range(len(Y)):
			#store predicted states
			PStore_[i,:]=P_.ravel()
			XStore_[i,:]=x_.ravel()
			# update state estimate with measurement and stored Kalman gain
			x,P,K= Kupdate(self.A,self.C,P_,self.Sigma_epsilon,x_,Y[i])
			# store corrected states and covariance matrices
			XStore[i,:]=x.ravel()
			PStore[i,:]=P.ravel()
			# predict new state
			x_, P_ = Kpred(self.A,P,self.Sigma_e,x)


			
		# initialise the smoother
		T=len(Y)
		XbStore = pb.zeros([T,self.nx])
		PbStore = pb.zeros([T,self.nx**2])
		S = pb.zeros([T,self.nx**2])
		XbStore[-1,:], PbStore[-1,:] = XStore[-1,:], PStore[-1,:]
		# RTS smoother
		for t in range(T-2,-1,-1):
			S_temp= dots(PStore[t,:].reshape(self.nx,self.nx),self.A.T,pb.inv(PStore_[t+1,:].reshape(self.nx,self.nx)))
			S[t,:]=S_temp.ravel()
			XbStore_temp= XStore[t,:].reshape(self.nx,1) + pb.dot(S_temp,(XbStore[t+1].reshape(self.nx,1) - XStore_[t+1,:].reshape(self.nx,1)))
			XbStore[t,:]=XbStore_temp.ravel()
			PbStore_temp = PStore[t,:].reshape(self.nx,self.nx) + dots(S_temp,PbStore[t+1,:].reshape(self.nx,self.nx)-PStore_[t+1,:].reshape(self.nx,self.nx),S_temp.T)
			PbStore[t,:]=PbStore_temp.ravel()
		# iterate a final time to calucate the cross covariance matrices 		
		M = pb.zeros([T,self.nx**2])
 		M[-1,:]=pb.dot(np.eye(self.nx)-pb.dot(K,self.C), pb.dot(self.A,PStore[-2,:].reshape(self.nx,self.nx))).ravel()
		for t in range(T-2,0,-1):
			M_temp=pb.dot(PStore[t,:].reshape(self.nx,self.nx),S[t-1,:].reshape(self.nx,self.nx).T) + dots(S[t,:].reshape(self.nx,self.nx),M[t+1,:].reshape(self.nx,self.nx) - pb.dot(self.A,PStore[t,:].reshape(self.nx,self.nx)),S[t-1].reshape(self.nx,self.nx).T)
			M[t,:]=M_temp.ravel()
		
		return XbStore,PbStore,M

#Define matrix multiplication
def dots(*args):
	lastItem = 1.
	for arg in args:
		lastItem = pb.dot(lastItem, arg)
	return lastItem		
			




