#Author Parham Aram
#Date 12-10-2011
from __future__ import division
#built-in modules
#standard library imports
#~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import pylab as pb
import operator
from scipy import io
#my modules
#~~~~~~~~~~~~~~~~~~~~
import spline
import IDEComponents

# load coefficients mat file for to tranform B-spline scaling andwavelet function to different scales

TransformCoefficients=io.loadmat('TransformCoefficients')

"""Each basis that is to be used in the integrodifference equation model 
should be defined here.

Each basis needs to have __mul__ and __conv__ operators that define inner products and convolution, respectively."""

class basis():
	"""Class defining scale and wavelet function properties"""
	def isbasis(self):
		return True


class convolution:

	"""Defines convolution of two basis functions."""


	def __init__(self,j1,j2,k1,k2,name1,name2,Supp,base=None):
		self.base='convolution'
		self.j1=j1
		self.j2=j2
		self.k1=k1
		self.k2=k2
		self.name1=name1
		self.name2=name2
		self.Supp=Supp

	def __repr__(self):

		if self.name1=='P':name_1='Scaling function;'
		else:name_1='Wavelet basis function;'

		if self.name2=='P':name_2='Scaling function;'
		else:name_2='Wavelet basis function;'

		return 'Convolution of '+name_1+'j='+str(self.j1)+',k='+str(self.k1)+' with '+name_2 +'j='+str(self.j2)+',k='+str(self.k2)

		

	def __mul__(self,g): 	
	

		'''Inner product of two  basis functions.'''

		#Multiplication is allowed from right.

		assert (g.base == 'scale' or g.base=='wavelet'), 'inner product error'
		if max((self.Supp[0],g.Supp()[0])) >= min((self.Supp[1],g.Supp()[1])):temp_mul=0; return temp_mul	
		#Finding highest and lowest resolutions between all basis functions  for inner product.
		else:
			DilationTranslationList=[(self.j1,self.k1,self.name1),(self.j2,self.k2,self.name2),(g.j,g.k,g.name)] 


			SortedDilationTranslationList=sorted(DilationTranslationList, key=operator.itemgetter(0))

			InnerProduct_j_min,InnerProduct_k_max=SortedDilationTranslationList[0][0],SortedDilationTranslationList[0][1]
			InnerProduct_j_med,InnerProduct_k_med=SortedDilationTranslationList[1][0],SortedDilationTranslationList[1][1]
			InnerProduct_j_max,InnerProduct_k_min=SortedDilationTranslationList[2][0],SortedDilationTranslationList[2][1]


			Convolution_j_max=np.max((self.j1,self.j2)) 					   
			Convolution_j_min=np.min((self.j1,self.j2)) 					   
			if Convolution_j_min==self.j1:Convolution_k_max=self.k1;Convolution_k_min=self.k2  
			else: Convolution_k_min=self.k1;Convolution_k_max=self.k2			   

	
		
			Convolution_temp_1=TransformCoefficients['T'+self.name1+str(InnerProduct_j_max+1-self.j1)]	
			Convolution_temp_2=TransformCoefficients['T'+self.name2+str(InnerProduct_j_max+1-self.j2)]

		
			Convolution_temp_len_min=np.min((Convolution_temp_1.shape[1],Convolution_temp_2.shape[1]))
			Convolution_temp_len_max=np.max((Convolution_temp_1.shape[1],Convolution_temp_2.shape[1]))


			if Convolution_temp_len_min==Convolution_temp_1.shape[1]:Convolution_Matrix_In=Convolution_temp_1;Convolution_Matrix_out=Convolution_temp_2 		
			else:Convolution_Matrix_In=Convolution_temp_2;Convolution_Matrix_out=Convolution_temp_1 

			Convolution_coefs_temp=np.zeros((Convolution_temp_len_min+Convolution_temp_len_max-1,Convolution_temp_len_max))
			for n in range(Convolution_temp_len_max):
				m=0
				for i in range(n,n+Convolution_temp_len_min):
					Convolution_coefs_temp[i,n]=Convolution_Matrix_In[0,m]
					m+=1
		
			Convolution_coefs=np.dot(Convolution_coefs_temp,Convolution_Matrix_out.T)


			Convolution_shift=(2**(InnerProduct_j_max+1-Convolution_j_min))*(Convolution_k_max)+(2**(InnerProduct_j_max+1-Convolution_j_max))*Convolution_k_min 


			InnerProduct_temp_2=Convolution_coefs                                              
 			InnerProduct_temp_1=TransformCoefficients['T'+g.name+str(InnerProduct_j_max+1-g.j)] 	
		

			InnerProduct_temp_len_1=InnerProduct_temp_1.shape[1]
			InnerProduct_temp_len_2=InnerProduct_temp_2.shape[0]

		

			InnerProduct_Matrix_In=InnerProduct_temp_2
			InnerProduct_Matrix_out=InnerProduct_temp_1 

			InnerProduct_coefs_temp=np.zeros((InnerProduct_temp_len_2+InnerProduct_temp_len_1-1,InnerProduct_temp_len_1))		
			for n in range(InnerProduct_temp_len_1):
				m=0
				for i in range(n,n+InnerProduct_temp_len_2):
					InnerProduct_coefs_temp[i,n]=InnerProduct_Matrix_In[m,0]
					m+=1
			InnerProduct_coefs=np.dot(InnerProduct_coefs_temp,InnerProduct_Matrix_out.T)

			InnerProduct_shift=(2**(InnerProduct_j_max+1-g.j))*(g.k)

		

			shift_minus=Convolution_shift; shift_plus=InnerProduct_shift
		

			start=5-InnerProduct_Matrix_In.shape[0]
			stop=4+InnerProduct_Matrix_out.shape[1]  
			temp_mul=0
			b12=spline.spline12(0,0)
	

			for i in range(start,stop):
				temp_mul+=InnerProduct_coefs[i-start]*b12(i+shift_plus-shift_minus)
			return float(temp_mul*(1./(2**(InnerProduct_j_max+1)))*(1./(2**(InnerProduct_j_max+1)))*(2**(self.j1/2.))*(2**(self.j2/2.))*(2**(g.j/2.)))


				
	def __call__(self,s):

		#This is to find maximum resolution.
		j_max=np.max((self.j1,self.j2)) 
		#This is to find minimum resolution.
		j_min=np.min((self.j1,self.j2)) 

		if j_min==self.j1:k_max=self.k1;k_min=self.k2          
		else: k_min=self.k1;k_max=self.k2                     


		temp_1=TransformCoefficients['T'+self.name1+str(j_max+1-self.j1)]	
		temp_2=TransformCoefficients['T'+self.name2+str(j_max+1-self.j2)]

		

		temp_len_min=np.min((temp_1.shape[1],temp_2.shape[1]))
		temp_len_max=np.max((temp_1.shape[1],temp_2.shape[1]))


		if temp_len_min==temp_1.shape[1]:Matrix_In=temp_1;Matrix_out=temp_2 
		else:Matrix_In=temp_2;Matrix_out=temp_1 

		Concoefs_temp=np.zeros((temp_len_min+temp_len_max-1,temp_len_max))
		for n in range(temp_len_max):
			m=0
			for i in range(n,n+temp_len_min):
				Concoefs_temp[i,n]=Matrix_In[0,m]
				m+=1
		Concoefs=np.dot(Concoefs_temp,Matrix_out.T)

		
		weights=Concoefs					    
		l=(2**(j_max+1-j_min))*(k_max)+2*k_min  		     
		f=[spline.spline8(j_max+1,j+l) for j in range(weights.shape[0])] 
		k=IDEComponents.spline8(1,f,weights)  			     
		return (1./2**(j_max+1))*k(s)*(2**(self.j1/2.))*(2**(self.j2/2.)) 


	def plot(self,show=1,color='color'):
		
		u=np.linspace(self.Supp[0],self.Supp[1],300) 
		z=np.zeros(len(u))
		for i in range(len(u)):
			z[i]=self(u[i])
		pb.plot(u,z,str(color)+'-') 
		if show==1: pb.show()

		
class wavelet(basis):
	"""Wavelet"""		

	def __init__(self,j,k):
		self.base='wavelet'
		self.j=j
		self.k=k
		self.name='Q'

	def __repr__(self):
		return 'Wavelet basis function;'+'j='+str(self.j) +',k='+str(self.k) 
	
	def Supp(self):
		'''Compute Support of the basis function'''
		return [(0+self.k)*2**-self.j,(7+self.k)*2**-self.j]

	def __call__(self,u):		
		x=2**self.j*u-self.k
		if u>=(0+self.k)*2**-self.j and u< (0.5+self.k)*2**-self.j: output=(4./3.)*x**3
		elif u>=(0.5+self.k)*2**-self.j and u< (1+self.k)*2**-self.j: output=(64./3.)-128*x+256*x**2-(508./3.)*x**3
		elif u>=(1+self.k)*2**-self.j and u< (1.5+self.k)*2**-self.j: output=-2884+8588*x-8460*x**2+2736*x**3
		elif u>=(1.5+self.k)*2**-self.j and u< (2+self.k)*2**-self.j: output=66236-129652*x+83700*x**2-17744*x**3
		elif u>=(2+self.k)*2**-self.j and u< (2.5+self.k)*2**-self.j: output=-580772+840860*x-401556*x**2+63132*x**3
		elif u>=(2.5+self.k)*2**-self.j and u< (3+self.k)*2**-self.j: output=2595228-2970340*x+1122924*x**2-140132*x**3
		elif u>=(3+self.k)*2**-self.j and u< (3.5+self.k)*2**-self.j: output=-6754800+6379688*x-1993752*x**2+(618496./3.)*x**3
		elif u>=(3.5+self.k)*2**-self.j and u< (4+self.k)*2**-self.j: output=(32771632./3.)-8773464*x+2335720*x**2-(618496./3.)*x**3
		elif u>=(4+self.k)*2**-self.j and u< (4.5+self.k)*2**-self.j: output=-11239152+7848808*x-1819848*x**2+140132*x**3
		elif u>=(4.5+self.k)*2**-self.j and u< (5+self.k)*2**-self.j: output=7283280-4499480*x+924216*x**2-63132*x**3
		elif u>=(5+self.k)*2**-self.j and u< (5.5+self.k)*2**-self.j: output=-2826220+1566220*x-288924*x**2+17744*x**3
		elif u>=(5.5+self.k)*2**-self.j and u< (6+self.k)*2**-self.j: output=581140-292340*x+48996*x**2-2736*x**3
		elif u>=(6+self.k)*2**-self.j and u< (6.5+self.k)*2**-self.j: output=-46412+21436*x-3300*x**2+(508./3.)*x**3
		elif u>=(6.5+self.k)*2**-self.j and u< (7+self.k)*2**-self.j: output=(1372./3.)-196*x+28*x**2-(4./3.)*x**3
		else: output=0
		return (2**(self.j/2.))*(output/40320.)

	def __mul__(self,g):

		'''inner product of two  basis functions'''

		# calculate the inner product.

		assert (g.base == 'scale' or g.base=='wavelet'), 'inner product error'
		
		if ((g.base is 'scale' and self.j>=g.j) or (g.base is 'wavelet' and self.j<>g.j)):temp_mul=0; return temp_mul	
		elif max((self.Supp()[0],g.Supp()[0])) >= min((self.Supp()[1],g.Supp()[1])):temp_mul=0; return temp_mul	

		else:	


			j_max=np.max((self.j,g.j)) 
			j_min=np.min((self.j,g.j)) 

			if j_min==self.j:k_max=self.k;k_min=g.k
			else: k_min=self.k;k_max=g.k		


			self_temp=TransformCoefficients['TQ'+str(j_max+1-self.j)]	
			g_temp=	TransformCoefficients['T'+g.name+str(j_max+1-g.j)]

		

			temp_len_min=np.min((self_temp.shape[1],g_temp.shape[1]))
			temp_len_max=np.max((self_temp.shape[1],g_temp.shape[1]))
			

			if temp_len_min==self_temp.shape[1]:Matrix_In=self_temp;Matrix_out=g_temp 
			else:Matrix_In=g_temp;Matrix_out=self_temp 

			Incoefs_temp=np.zeros((temp_len_min+temp_len_max-1,temp_len_max))
			for n in range(temp_len_max):
				m=0
				for i in range(n,n+temp_len_min):
					Incoefs_temp[i,n]=Matrix_In[0,m]
					m+=1
			Incoefs=np.dot(Incoefs_temp,Matrix_out.T)
			#Calculate inner product
			start=5-Matrix_In.shape[1]
			stop=4+Matrix_out.shape[1]		
			temp_mul=0
			b8=spline.spline8(0,0)	

	

			for i in range(start,stop):
				temp_mul+=Incoefs[i-start]*b8(i+(2**(j_max+1-j_min))*(k_max)-2*k_min)
			return float(temp_mul*(1./(2**(j_max+1)))*(2**(self.j/2.))*(2**(g.j/2.)))





	def plot(self,show=1,color='color'):
		u=np.linspace(self.Supp()[0],self.Supp()[1],1000)
		z=np.zeros_like(u)
		for i in range(len(u)):
			z[i]=self(u[i])
		pb.plot(u,z,str(color)+'-')
		if show==1: pb.show()
	

	def conv(self,g):
		"""Convolution of two bases"""
		assert (g.base=='wavelet' or g.base =='scale'), 'convolution error'
		Supp=np.add(self.Supp(),g.Supp())
		h=convolution(self.j,g.j,self.k,g.k,self.name,g.name,Supp)
		return h



class scale(basis):
	"""scale"""
	def __init__(self,j,k):
		self.base='scale'		
		self.j=j
		self.k=k
		self.name='P'

	def __repr__(self):
		return 'Scaling function;'+'j='+str(self.j) +',k='+str(self.k) 

	def Supp(self):
		'''Compute Support of the basis function'''
		return [(0+self.k)*2**-self.j,(4+self.k)*2**-self.j]

	def __call__(self,u):	
		x=2**self.j*u-self.k
		if u>= (0+self.k)*2**-self.j and u< (1+self.k)*2**-self.j: output=x**3
		elif u>= (1+self.k)*2**-self.j and u< (2+self.k)*2**-self.j: output=4-12*x+12*x**2-3*x**3
		elif u>= (2+self.k)*2**-self.j and u< (3+self.k)*2**-self.j: output=-44+60*x-24*x**2+3*x**3
		elif u>= (3+self.k)*2**-self.j and u<= (4+self.k)*2**-self.j: output=64-48*x+12*x**2-x**3 
		else: output=0
		return (2**(self.j/2.))*(output/6.)


	def __mul__(self,g):

		'''inner product of two  basis functions'''

		# calculate the inner product.

		assert (g.base == 'scale' or g.base=='wavelet'), 'inner product error'


		if (g.base is 'wavelet' and self.j<=g.j):temp_mul=0; return temp_mul		
		elif max((self.Supp()[0],g.Supp()[0])) >= min((self.Supp()[1],g.Supp()[1])):temp_mul=0; return temp_mul	
		else:
			j_max=np.max((self.j,g.j)) 
			j_min=np.min((self.j,g.j)) 

			if j_min==self.j:k_max=self.k;k_min=g.k 	
			else: k_min=self.k;k_max=g.k			


			self_temp=TransformCoefficients['TP'+str(j_max+1-self.j)]	
			g_temp=	TransformCoefficients['T'+g.name+str(j_max+1-g.j)]

		
			temp_len_min=np.min((self_temp.shape[1],g_temp.shape[1]))
			temp_len_max=np.max((self_temp.shape[1],g_temp.shape[1]))


			if temp_len_min==self_temp.shape[1]:Matrix_In=self_temp;Matrix_out=g_temp 
			else:Matrix_In=g_temp.shape[1];Matrix_out=self_temp 


			Incoefs_temp=np.zeros((temp_len_min+temp_len_max-1,temp_len_max))
			for n in range(temp_len_max):
				m=0
				for i in range(n,n+temp_len_min):
					Incoefs_temp[i,n]=Matrix_In[0,m]
					m+=1
			Incoefs=np.dot(Incoefs_temp,Matrix_out.T)
			#Calculate inner product
			start=5-Matrix_In.shape[1]
			stop=4+Matrix_out.shape[1]		
			temp_mul=0
			b8=spline.spline8(0,0)	

	
	
			for i in range(start,stop):
				temp_mul+=Incoefs[i-start]*b8(i+(2**(j_max+1-j_min))*(k_max)-2*k_min)
			return float(temp_mul*(1./(2**(j_max+1)))*(2**(self.j/2.))*(2**(g.j/2.)))



	def plot(self,show=1,color='color'):
		u=np.linspace(self.Supp()[0],self.Supp()[1],1000)
		z=np.zeros_like(u)
		for i in range(len(u)):
			z[i]=self(u[i])
		pb.plot(u,z,str(color)+'-')
		if show==1: pb.show()


	def conv(self,g):
		"""Convolution of two bases"""
		assert (g.base=='wavelet' or g.base =='scale'), 'convolution error'
		Supp=np.add(self.Supp(),g.Supp())
		h=convolution(self.j,g.j,self.k,g.k,self.name,g.name,Supp)
		return h


	
		

