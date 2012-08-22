#Author Parham Aram
#Date 12-10-2011

#Built-in modules
#~~~~~~~~~~~~~~~~~
from __future__ import division
import pylab as pb
import numpy as np
import time
from scipy import signal
import scipy as sp


#My modules
#~~~~~~~~~
import SKF
import ML

class IDE():

	def __init__(self,kernel,field,sensor_kernel,obs_locns,eta,eta_weight,Sigma_epsilon,Ts,zeta,varsigma,x0,P0):

		self.kernel = kernel
		self.field = field
		self.sensor_kernel=sensor_kernel
		self.obs_locns=obs_locns
		self.eta=eta
		self.eta_weight=eta_weight
		self.Sigma_epsilon=Sigma_epsilon
		self.x0=x0
		self.P0=P0
		self.nx=len(self.field.Mu)
		self.ny=len(obs_locns)
		self.ntheta=len(self.kernel.Lambda)
		self.Ts=Ts
		self.zeta=zeta
		self.varsigma=varsigma
		self.xi=1-self.zeta*self.Ts
    
    
	def FroNorm(self):
		'''returns Frobenius Norm'''
		fn= pb.sqrt(pb.trace(self.A.T*self.A))
		return fn



	def gen_ssmodel(self):


		#calculate U
		#~~~~~~~~~~~

		if not(hasattr(self,'U')):

			t0=time.time()
			self.U_Index=[]
			U=pb.empty((self.nx,self.nx),dtype=object) 
			U_T=pb.empty((self.nx,self.nx),dtype=object)
			for i in range(self.nx):				
				for j in range(self.nx):
					u_temp=pb.vectorize(self.field.Mu[j,0].conv)(self.kernel.Lambda) 
					U[i,j]=pb.dot(u_temp.T,self.field.Mu[i,0]).astype(float) 
					U_T[j,i]=U[i,j].T

					if U[i,j].any():
						self.U_Index.append((i,j))	

			self.U=U
			self.U_T=U_T
			print 'Elapsed time in seconds to calculate U is', time.time()-t0

		#calculate the inverse of inner product of field basis functions
		if not(hasattr(self,'Lambda_x_blocks_inverse')):
			t0=time.time()
			Lambda_x_blocks_inverse=pb.empty((1,len(self.field.NoatEachLevel)),dtype=object)
			Lambda_x_inverse_temp=0
			for i in range(Lambda_x_blocks_inverse.shape[1]):
				Lambda_x_blocks_inverse_temp=pb.reshape(self.field.Mu[Lambda_x_inverse_temp:Lambda_x_inverse_temp+self.field.NoatEachLevel[i],0],(1,self.field.NoatEachLevel[i]))
				Lambda_x_blocks_inverse[0,i]=pb.inv(pb.dot(Lambda_x_blocks_inverse_temp.T,Lambda_x_blocks_inverse_temp)).astype(float)
				Lambda_x_inverse_temp+=self.field.NoatEachLevel[i]
			self.Lambda_x_blocks_inverse=Lambda_x_blocks_inverse
			print 'Elapsed time in seconds to calculate Lambda_x_inverse is',time.time()-t0

		#calculate Lambda_x 
		if not(hasattr(self,'Lambda_x')):
			t0=time.time()
			Lambda_x=pb.dot(self.field.Mu,self.field.Mu.T)
			self.Lambda_x=Lambda_x
			print 'Elapsed time in seconds to calculate Lambda_x is',time.time()-t0

		t0=time.time()
		Lambda_theta = pb.zeros([self.nx,self.nx]) 
		for i in self.U_Index:
			Lambda_theta[i] = pb.dot(self.U[i],self.kernel.weights)			
		self.Lambda_theta = Lambda_theta

		#calculate A
		A_blocks=pb.empty((1,len(self.field.NoatEachLevel)),dtype=object)
		A_temp=0
		for i in range(A_blocks.shape[1]):
			A_blocks_temp=self.Lambda_theta[A_temp:self.field.NoatEachLevel[i]+A_temp]
			A_blocks[0,i]=pb.dot(self.Lambda_x_blocks_inverse[0,i],A_blocks_temp)	
			A_temp+=self.field.NoatEachLevel[i]
		self.A=self.Ts*self.varsigma*pb.vstack(A_blocks[0,:])+self.xi*pb.eye(self.nx)
		print 'Elapsed time in seconds to calculate Lambda_theta and A is',time.time()-t0

		# form the observation matrix 
		if not(hasattr(self,'C')):
			t0=time.time()
            

			t_observation_matrix=time.time()
			sensor_kernel_convolution_vecrorized=pb.vectorize(self.sensor_kernel.conv)
			sensor_kernel_conv_Mu=sensor_kernel_convolution_vecrorized(self.field.Mu).T  
			C=pb.empty(([self.ny,self.nx]),dtype=float)
			for i in range(self.nx):
				c_temp=pb.vectorize(sensor_kernel_conv_Mu[0,i].__call__)
				C[:,i]=c_temp(self.obs_locns)

			self.C=C
			print 'Elapsed time in seconds to calculate observation matrix C is',time.time()-t_observation_matrix	


		#calculate Sigma_e_c
		if not(hasattr(self,'Sigma_e_c')):

			t0=time.time()
			eta_convolution_vecrorized=pb.vectorize(self.eta.conv)
			eta_conv_Mu=eta_convolution_vecrorized(self.field.Mu).T
			Pi=pb.dot(eta_conv_Mu.T,self.field.Mu.T).astype(float).T
			self.Pi=Pi
			Sigma_e_blocks=pb.empty((1,len(self.field.NoatEachLevel)),dtype=object)
			Sigma_e_temp=0
			#calculate (LAmbda_x )^-1* Sigma_e)
			for i in range(Sigma_e_blocks.shape[1]):
				Sigma_e_blocks_temp=Pi[Sigma_e_temp:self.field.NoatEachLevel[i]+Sigma_e_temp]
				Sigma_e_blocks[0,i]=pb.dot(self.Lambda_x_blocks_inverse[0,i],Sigma_e_blocks_temp)	
				Sigma_e_temp+=self.field.NoatEachLevel[i]
			Sigma_e=pb.vstack(Sigma_e_blocks[0,:])
			
			Sigma_e_temp=0
			for i in range(Sigma_e_blocks.shape[1]):
				Sigma_e_blocks_temp=Sigma_e[:,Sigma_e_temp:self.field.NoatEachLevel[i]+Sigma_e_temp]
				Sigma_e_blocks[0,i]=pb.dot(Sigma_e_blocks_temp,self.Lambda_x_blocks_inverse[0,i].T)
				Sigma_e_temp+=self.field.NoatEachLevel[i]
			Sigma_e=pb.hstack(Sigma_e_blocks[0,:])
			self.Sigma_e=self.eta_weight*Sigma_e
			print 'Elapsed time in seconds to calculate Sigma_e is',time.time()-t0
			self.Sigma_e_c=sp.linalg.cholesky(self.Sigma_e,lower=1)


		#calculate Sigma_epsilon_c

		if not(hasattr(self,'Sigma_epsilon_c')):
			Sigma_epsilon_c=sp.linalg.cholesky(self.Sigma_epsilon,lower=1)
			self.Sigma_epsilon_c=Sigma_epsilon_c

		#calculate EM components for speed
		if not(hasattr(self,'Delta_upsilon')):					
			t0=time.time()
			self.Delta_upsilon=dots(pb.inv(self.Sigma_e),directsum(self.Lambda_x_blocks_inverse),self.U)
			print 'Elapsed time in seconds to calculate Delta_upsilon is',time.time()-t0

		if not(hasattr(self,'Delta_Upsilon')):
			t0=time.time()
			self.Delta_Upsilon=dots(self.U_T,directsum(self.Lambda_x_blocks_inverse),self.Delta_upsilon)
			print 'Elapsed time in seconds to calculate Delta_Upsilon is',time.time()-t0

	def simulate(self,T):

		Y = []		
		X = []		
		x=self.x0


		print "iterating"
		for t in T[1:]:

			v = np.random.randn(self.ny,1)
			w = np.random.randn(self.nx,1)
			x = pb.dot(self.A,x)+pb.dot(self.Sigma_e_c,w)
			X.append(x)
			Y.append(pb.dot(self.C,x)+ pb.dot(self.Sigma_epsilon_c,v))

		return X,Y



class para_state_estimation():

	def __init__(self,model):

		'''this is to estimate state and connectivity kernel parameters

		Arguments:
		----------
			model: IDE instance
			order: 1 or 2
				specify the zero or first Taylor approximation to the non-linearity '''

		self.model=model





	def itrerative_state_parameter_estimation(self,Y,max_it):

		"""Two part iterative algorithm, consisting of a state estimation step followed by a
		parameter estimation step
		
		Arguments:
		---------
		Y: list of matrices
			Observation vectors
		max_it: int
			maximum number of iterations """


		xi_est=[]
		kernel_weights_est=[]
		# generate a random state sequence
		Xb=np.random.rand(len(Y),self.model.nx)
		Pb=pb.zeros([len(Y),self.model.nx**2])
		Mb=pb.zeros([len(Y),self.model.nx**2])

		# iterate
		keep_going = 1
		it_count = 0
		print " Estimatiing IDE's kernel and field weights"
		t0=time.time()
		while keep_going:
			ML_instance=ML.para_state_estimation(self.model)
			temp=ML_instance.estimate_kernel(Xb,Pb,Mb)
			kernel_weights_est.append(temp)
			self.model.kernel.weights=temp
			self.model.gen_ssmodel();FrobNorm=self.model.FroNorm()
			skf_instance=SKF.SKF(self.model.x0,self.model.P0,self.model.A,self.model.C,self.model.Sigma_e,self.model.Sigma_epsilon)
			Xb,Pb,Mb=skf_instance.rtssmooth(Y)
			self.model.Xb=Xb
			self.model.Pb=Pb 
			self.model.kernel_weights_est=kernel_weights_est
			self.model.x0=Xb[0,:].reshape(self.model.nx,1)
			self.model.P0=Pb[0,:].reshape(self.model.nx,self.model.nx)

			print it_count, " Kernel current estimate: ", self.model.kernel.weights
			print it_count,"current estimate Frobenius norm: ",FrobNorm 

			if it_count == max_it:
				keep_going = 0
			it_count += 1
		print "Elapsed time in seconds is", time.time()-t0




#define matrix multiplication
def dots(*args):
	lastItem = 1.
	for arg in args:
		lastItem = pb.dot(lastItem, arg)
	return lastItem		
			
#define direc sum			
def directsum(arg):

	#check if all matrices are square
	for i in range(arg.shape[1]):
		assert arg[0,i].shape[0]==arg[0,i].shape[1], 'Matrix '+str(arg[i])+ ' is not square.'

	#calculate the shape of the result of the directsum operation
	temp=(0,0)
	for i in range(arg.shape[1]):
		temp=pb.add(temp,arg[0,i].shape)
	dsum =pb.zeros(temp)

	dsum[:arg[0,0].shape[0],:arg[0,0].shape[1]]=arg[0,0]

	rc_Indices=0   
	for i in range(1,arg.shape[1]):
		rc_Indices+=arg[0,i-1].shape[0]
		dsum[rc_Indices:rc_Indices+arg[0,i].shape[0],rc_Indices:rc_Indices+arg[0,i].shape[1]]=arg[0,i]
	return dsum







