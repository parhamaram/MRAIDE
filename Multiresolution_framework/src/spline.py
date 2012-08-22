#Author Parham Aram
#Date 12-10-2011
# calculate spline order 8 and 12 required to compute convolution and inner product of B-spline scaling and wavelet functions
from __future__ import division
import numpy as np
import pylab as pb

class spline8():
	"""BSpline order 8  to evaluate convolution of two bases"""
	def __init__(self,j,k):
		self.j=j
		self.k=k

	def __repr__(self):
		return 'Bspline 8;'+'j='+str(self.j) +',k='+str(self.k) 

	def Supp(self):
		'''Compute Support of the basis function'''
		return [(0+self.k)*2**-self.j,(8+self.k)*2**-self.j]

	def evaluate_at_real(self,u):			
		x=2**self.j*u-self.k
		if u>= (0+self.k)*2**-self.j and u< (1+self.k)*2**-self.j: output=x**7
		elif u>= (1+self.k)*2**-self.j and u< (2+self.k)*2**-self.j: output=8-56*x+168*x**2-280*x**3+280*x**4-168*x**5+56*x**6-7*x**7
		elif u>= (2+self.k)*2**-self.j and u< (3+self.k)*2**-self.j: output=-3576+12488*x-18648*x**2+15400*x**3-7560*x**4+2184*x**5-336*x**6+21*x**7
		elif u>= (3+self.k)*2**-self.j and u< (4+self.k)*2**-self.j: output=118896-273280*x+267120*x**2-143360*x**3+45360*x**4-8400*x**5+840*x**6-35*x**7
		elif u>= (4+self.k)*2**-self.j and u< (5+self.k)*2**-self.j: output=-1027984+1733760*x-1238160*x**2+483840*x**3-111440*x**4+15120*x**5-1120*x**6+35*x**7
		elif u>= (5+self.k)*2**-self.j and u< (6+self.k)*2**-self.j: output=3347016-4391240*x+2436840*x**2-741160*x**3+133560*x**4-14280*x**5+840*x**6-21*x**7
		elif u>= (6+self.k)*2**-self.j and u< (7+self.k)*2**-self.j: output=-4491192+4753336*x-2135448*x**2+528920*x**3-78120*x**4+6888*x**5-336*x**6+7*x**7
		elif u>= (7+self.k)*2**-self.j and u<= (8+self.k)*2**-self.j: output=-(-8+x)**7
		else: output=0
		return output/5040.


	def __call__(self,u):
		
		if u==(1+self.k)*2**-self.j or u==(7+self.k)*2**-self.j :output=1
		elif u==(2+self.k)*2**-self.j or u==(6+self.k)*2**-self.j:output=120
		elif u==(3+self.k)*2**-self.j or u==(5+self.k)*2**-self.j:output=1191
		elif u==(4+self.k)*2**-self.j:output=2416
		else: output=0
		return output/5040.

	def plot(self):
		u=np.linspace(self.Supp()[0],self.Supp()[1],1000)
		z=np.zeros_like(u)
		for i in range(len(u)):
			z[i]=self.evaluate_at_real(u[i])
		pb.plot(u,z)
		pb.show()

class spline12():
	"""BSpline order 12 using to evaluate inner product of two bases"""
	def __init__(self,j,k,wsrange=None):
		self.j=j
		self.k=k		

	def __repr__(self):
		return 'Bspline 12;'+'j='+str(self.j) +',k='+str(self.k) 

	def Supp(self):
		'''Compute Support of the basis function'''
		return [(0+self.k)*2**-self.j,(12+self.k)*2**-self.j]
	
	def evaluate_at_real(self,u):			
		x=2**self.j*u-self.k
		if u>= (0+self.k)*2**-self.j and u< (1+self.k)*2**-self.j: output=x**11
		elif u>= (1+self.k)*2**-self.j and u< (2+self.k)*2**-self.j: output=12-132*x+660*x**2-1980*x**3+3960*x**4-5544*x**5+5544*x**6-3960*x**7+1980*x**8-660*x**9+132*x**10-11*x**11
		elif u>= (2+self.k)*2**-self.j and u< (3+self.k)*2**-self.j: output=-135156+743292*x-1857900*x**2+2785860*x**3-2783880*x**4+1945944*x**5-970200*x**6+344520*x**7-85140*x**8+13860*x**9-1320*x**10+55*x**11
		elif u>= (3+self.k)*2**-self.j and u< (4+self.k)*2**-self.j: output=38837184-142155288*x+236306400*x**2-235378440*x**3+155992320*x**4-72149616*x**5+23728320*x**6-5536080*x**7+894960*x**8-95040*x**9+5940*x**10-165*x**11
		elif u>= (4+self.k)*2**-self.j and u< (5+self.k)*2**-self.j: output=-2037343296+5567341032*x-6900564000*x**2+5117274360*x**3-2520334080*x**4+864564624*x**5-210450240*x**6+36281520*x**7-4332240*x**8+340560*x**9-15840*x**10+330*x**11
		elif u>= (5+self.k)*2**-self.j and u< (6+self.k)*2**-self.j: output=36634531704-79510783968*x+78177561000*x**2-45929600640*x**3+17898415920*x**4-4852685376*x**5+932999760*x**6-127068480*x**7+12002760*x**8-748440*x**9+27720*x**10-462*x**11
		elif u>= (6+self.k)*2**-self.j and u< (7+self.k)*2**-self.j: output=-298589948040+535067428896*x-433970949720*x**2+210144654720*x**3-67459669200*x**4+15064201152*x**5-2386481328*x**6+268107840*x**7-20928600*x**8+1081080*x**9-33264*x**10+462*x**11
		elif u>= (7+self.k)*2**-self.j and u< (8+self.k)*2**-self.j: output=1267452832416-1925856940392*x+1323832171200*x**2-543199539960*x**3+147781529280*x**4-27984038544*x**5+3763267200*x**6-359417520*x**7+23894640*x**8-1053360*x**9+27720*x**10-330*x**11
		elif u>= (8+self.k)*2**-self.j and u< (9+self.k)*2**-self.j: output=-2984564790624+3920667291288*x-2330245473600*x**2+827079576840*x**3-194788249920*x**4+31965672816*x**5-3730446720*x**6+309664080*x**7-17922960*x**8+689040*x**9-15840*x**10+165*x**11
		elif u>= (9+self.k)*2**-self.j and u< (10+self.k)*2**-self.j: output=3919268323356-4517350959132*x+2357542443300*x**2-735516395460*x**3+152455299480*x**4-22049990424*x**5+2271293640*x**6-166664520*x**7+8539740*x**8-291060*x**9+5940*x**10-55*x**11
		elif u>= (10+self.k)*2**-self.j and u< (11+self.k)*2**-self.j: output=-2680731676644+2742649040868*x-1272457556700*x**2+353483604540*x**3-65344700520*x**4+8442009576*x**5-777906360*x**6+51135480*x**7-2350260*x**8+71940*x**9-1320*x**10+11*x**11
		elif u>= (11+self.k)*2**-self.j and u<= (12+self.k)*2**-self.j: output=-(-12+x)**11
		else: output=0
		return output/39916800.

	def __call__(self,u):
		if u==(1+self.k)*2**-self.j or u==(11+self.k)*2**-self.j :output=1
		elif u==(2+self.k)*2**-self.j or u==(10+self.k)*2**-self.j:output=2036
		elif u==(3+self.k)*2**-self.j or u==(9+self.k)*2**-self.j:output=152637
		elif u==(4+self.k)*2**-self.j or u==(8+self.k)*2**-self.j:output=2203488
		elif u==(5+self.k)*2**-self.j or u==(7+self.k)*2**-self.j:output=9738114
		elif u==(6+self.k)*2**-self.j :output=15724248
		else: output=0
		return output/39916800.
		
	def plot(self):
		u=np.linspace(self.Supp()[0],self.Supp()[1],1000)
		z=np.zeros_like(u)
		for i in range(len(u)):
			z[i]=self.evaluate_at_real(u[i])
		pb.plot(u,z)
		pb.show()



