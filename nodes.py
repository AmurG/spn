import numpy as np
from scipy.stats import multivariate_normal as mn

class Node:
	def __init__(self, n, scope):
		self.scope = []
		self.children = []
		self.parent = None
		self.value = 0
	
	def retval(self):
		return self.value		

class prodNode(Node):
	def evaluate(self):
		Logval = 0
		for i in self.children:
				Logval = Logval + retval(i)
		self.value = Logval

class sumNode(Node):
	def __init__(self, n, scope):
		self.scope = []
		self.children = []
		self.wts = []	
		self.parent = None
		self.value = 0

	
	def setwts(self,arr):
		for i in arr:
			self.wts.append(i)

	def evaluate(self):
		Rawval = 0
		for i in self.children:
			Rawval = Rawval + (wts[i])*(np.exp(retval(i)))
		self.value = np.exp(Rawval)

class leafNode(Node):
	def __init__(self):
		self.value = 0
		
	def create(self,mean,cov):
		self.pdf = mn(mean=mean,cov=cov)

	def setval(self,val):
		self.value = self.pdf.logpdf(val)



#test

t = leafNode()
mean = [0, 0]
cov = [[1, 0],[0,1]]
t.create(mean,cov)
t.setval([5.0,0.0])
print(t.retval())



