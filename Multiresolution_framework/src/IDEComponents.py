#Author Parham Aram
#Date 12-10-2011
'''This is to generate IDE components: connectivity kernal and the field'''
import pylab as pb

class Kernel():

	"""class defining the Connectivity kernel of the brain.

	Arguments
	----------

	Lambda:array
		array of connectivity kernel basis functions
	weights: array
		list of connectivity kernel weights	

	"""
	
	def __init__(self,weights,*Lambda):

		
		self.Lambda=pb.concatenate(Lambda)
		self.weights = weights
		self.NoatEachLevel=[len(f) for f in Lambda]
	def plot(self,width,color='k'):
		temp1=[]
		space=pb.linspace(-width,width,1000)
		for i in range(self.Lambda.shape[0]):
			temp2=[]
			for s in space:
				temp2.append(self.weights[i]*self.Lambda[i,0](s))
			temp1.append(temp2)
		pb.plot(space,pb.sum(pb.squeeze(temp1),axis=0),color)
		
		
		
		



class Field():	

	"""class defining the field.

	Arguments
	----------

	Mu:array
		array of field basis functions; must be of class Bases

	"""

	def __init__(self, *Mu):

		self.Mu=pb.concatenate(Mu)
		self.NoatEachLevel=[len(f) for f in Mu]

		

class spline8():
	"""Clasee defining a spline (m=8) in order to calculate convolution in terms of N_8(s)"""
	def __init__(self,dimension,bases,weights):
		self.dimension = dimension
		assert type(bases) is list, 'the bases must be in a list (even if there is only one)'			
		self.bases = bases
		self.weights = weights
		
	def __call__(self,s):
		return sum([self.bases[i].evaluate_at_real(s)*self.weights[i] for i in range(len(self.bases))])

