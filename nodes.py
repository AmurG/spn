import numpy as np
from scipy.stats import multivariate_normal as mn

class Node:
	def __init__(self):
		self.scope = []
		self.children = []
		self.parent = None
		self.value = 0
		self.det = []
	
	def passon(self,arr):
		self.det = arr
		for i in self.children:
			i.passon(arr)		

class prodNode(Node):
	def retval(self):
		Logval = 0
		for i in self.children:
				Logval = Logval + i.retval()
		self.value = Logval
		return (self.value)

class sumNode(Node):
	def __init__(self):
		self.scope = []
		self.children = []
		self.wts = np.zeros(10)	
		self.parent = None
		self.value = 0
		self.det = []

	
	def setwts(self,arr):
		j = 0
		for i in arr:
			self.wts[j] = i
			j = j+1

	def retval(self):
		Rawval = 0
		j = 0
		for i in self.children:
			Rawval = Rawval + (self.wts[j])*(np.exp(i.retval()))
			j = j+1
		self.value = np.log(Rawval)
		return (self.value)

class leafNode(Node):
	def __init__(self):
		self.value = 0
		self.flag = 1
		
	def create(self,mean,cov):
		self.pdf = mn(mean=mean,cov=cov)

	def setval(self,val):
		self.value = self.pdf.logpdf(val)

	def passon(self,arr):
		self.setval(arr)

	def retval(self):
		return(self.value)	



#test


t1 = leafNode()
mean = [0, 0]
cov = [[1, 0],[0,1]]
t1.create(mean,cov)


t2 = leafNode()
mean = [1, 0]
cov = [[1, -0.5],[-0.5,1]]
t2.create(mean,cov)

p = prodNode()
p.children.append(t1)
p.children.append(t2)

s = sumNode()
s.setwts([1])
s.children.append(p)

s.passon([1, 0])
print(s.retval())



