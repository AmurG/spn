import numpy as np
from data import *
from scipy.stats import multivariate_normal as mn

class Node:
	def __init__(self):
		self.scope = set()
		self.children = []
		self.parent = None
		self.value = 0
		self.det = []
	
	def passon(self,arr):
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
		self.wts = []
		self.parent = None
		self.value = 0
		self.det = []

	
	def setwts(self,arr):
		for i in arr:
			self.wts.append(i)

	def retval(self):
		Rawval = 0
		j = 0
		sum = 0
		for i in self.children:
			Rawval = Rawval + float((self.wts[j])*(np.exp(i.retval())))
			sum = sum + self.wts[j]
			j = j+1
		self.value = np.log(float(Rawval)/(float(sum)+1e-11) + 1e-11)
		return (self.value)

class leafNode(Node):
	def __init__(self):
		self.value = 0
		self.flag = 1
		self.scope = []
		
	def create(self,mean,cov):
		self.pdf = mn(mean=mean,cov=cov)

	def setval(self,val):
		self.value = self.pdf.logpdf(submean(val,self.scope))

	def passon(self,arr):
		self.setval(arr)

	def retval(self):
		return(self.value)	



#test

'''
t = leafNode()
mean = [0, 0]
cov = [[1, 0],[0,1]]
t.create(mean,cov)

p = prodNode()
p.children.append(t)


t = leafNode()
mean = [1, 0]
cov = [[1, -0.5],[-0.5,1]]
t.create(mean,cov)

p.children.append(t)

s = sumNode()
s.setwts([1])
s.children.append(p)

s.passon([1, 0])
print(s.retval())
'''


