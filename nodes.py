import numpy as np
from data import *
from scipy.stats import multivariate_normal as mn

def bintodec(arr):
	wt = np.rint(np.power(2,len(arr)-1))
	cnt = 0
	for i in range(0,len(arr)):
		cnt = cnt + wt*arr[i]
		wt = wt/2
	return int(np.rint(cnt))


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

	def update(self):
		for i in self.children:
			i.update()

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
		Rawval = 0.0
		j = 0
		sum = 0.0
		for i in self.children:
			Rawval = Rawval + float((self.wts[j])*(np.exp(i.retval())))
			sum = sum + self.wts[j]
			j = j+1
		self.value = np.log(float(Rawval)/(float(sum)+1e-11) + 1e-11)
		return (self.value)

	def update(self):
		inf = -10000000000
		j = 0
		for i in self.children:
			if((i.value)>inf):
				inf = (i.value)
				winnode = i
				winidx = j
			j = j+1
		self.wts[winidx] = self.wts[winidx]+1
		winnode.update()


class leafNode(Node):
	def __init__(self):
		self.value = 0
		self.flag = 1
		self.mean = []
		self.cov = []
		self.rec = []
		self.scope = []
		self.counter = 5
		
	def create(self,mean,cov):
		self.pdf = mn(mean=mean,cov=cov)
		self.mean = mean
		self.cov = cov

	def passon(self,arr):
		self.rec = submean(arr,self.scope)
		self.value = self.pdf.logpdf(self.rec)

	def retval(self):
		return(self.value)
	
	def update(self):
		
		tempmean = np.zeros(len(self.mean))
		for i in range(0,len(self.mean)):
			tempmean[i] = self.mean[i] + float((self.rec[i] - self.mean[i])/(float(self.counter)))
		for i in range(0,len(self.mean)):
			for j in range(0,len(self.mean)):
				self.cov[i][j] = float((self.cov[i][j]*(self.counter-1) + (self.rec[i]-tempmean[i])*(self.rec[j] - self.mean[j]))/(self.counter))
		self.mean = tempmean
		self.pdf = mn(mean=self.mean,cov=self.cov)
		self.counter = self.counter+1
		
		return	

class discNode(Node):
	def __init__(self):
		self.value = 0
		self.flag = 1
		self.rec = []
		self.scope = []
		self.arr = []
		self.size = 0
		self.counter = 5.0

	def create(self,pdfarr):
		self.arr = pdfarr
		self.size = len(pdfarr)

	def passon(self,arr):
		self.rec = submean(arr,self.scope)
		self.value = np.log(self.arr[bintodec(self.rec)]+1e-11)

	def retval(self):
		return (self.value)

	def update(self):
		idx = bintodec(self.rec)
		self.arr[idx] = float(self.arr[idx]) + float((1.0)/float(self.counter))
		for i in range(0,self.size):
			self.arr[i] = float(float(self.arr[i])/(1.0+float((1.0)/float(self.counter))))
		self.counter = self.counter+1
		


