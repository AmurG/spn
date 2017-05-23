import math
import numpy as np
import scipy
import networkx as nx
from scipy.stats import multivariate_normal as mn
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hcluster
from nodes import *
from data import *

k = 7

cov = np.zeros(k*k)
cov = np.reshape(cov,(k,k))

for i in range(0,k):
	for j in range(0,k):
		if(j<i):
			cov[i,j] = cov[j,i]			
		if(i==j):
			cov[i,j] = 1
		if(j>i):
			cov[i,j] = np.random.uniform(0,1)


cov = np.matmul(cov,np.transpose(cov))

print(cov)

mean = np.random.uniform(-1,1,k)

gen = np.random.multivariate_normal(mean,cov,1000)

data = np.transpose(np.asarray(gen))

operdat = np.transpose(data)

corrmat = np.corrcoef(data)
covmat = np.cov(data)

def returnarr(arr,scope):
	q = []
	te = list(scope)
	te = sorted(te)
	for i in arr:
		q.append(te[i])
	return set(q)

def induce(idxst,idxend,maxsize,scope,indsize):
	print("inducecall")
	print(scope)
	tempdat = operdat[idxst:idxend,:]
	effdat = np.zeros(len(tempdat)*len(scope))
	effdat = np.reshape(effdat,(len(tempdat),len(scope)))
	for i in range(0,len(tempdat)):
		temp = submean(tempdat[i],scope)
		for j in range(0,len(scope)):
			effdat[i][j] = temp[j]
	effcorr = np.corrcoef(np.transpose(effdat))
	effcov = np.cov(np.transpose(effdat))
	print(np.shape(effcorr))
	print(np.shape(effcov))
	empmean = np.mean(effdat,axis=0)
	print(np.shape(empmean))

	G = nx.from_numpy_matrix(-abs(effcorr))
	G = G.to_undirected()

	Dec = []

	T=nx.minimum_spanning_tree(G)
	Order = np.asarray(T.edges(data='weight'))
	k = len(Order)
	wts = np.zeros(k)
	Order = Order[Order[:,2].argsort()]
	Dec = []
	Gc = max(nx.connected_component_subgraphs(T), key=len)
	n = Gc.number_of_nodes()
	if(n<=maxsize):
		Dec.append(list(nx.connected_components(T)))

	for i in range(0,k):
		sum = 0
		for j in range(0,len(Order)-i):
			sum = sum - Order[j,2]
		wts[i] = sum
		idx = int(Order[len(Order)-i-1,0])
		idx2 = int(Order[len(Order)-i-1,1])
		T.remove_edge(idx,idx2)
		Gc = max(nx.connected_component_subgraphs(T), key=len)
		n = Gc.number_of_nodes()
		if(n<=maxsize):
			Dec.append(list(nx.connected_components(T)))

	#wts[k-1]=0.1
	effwts = np.zeros(len(Dec))
	for i in range(0,len(Dec)):
		effwts[i] = wts[i+k-len(Dec)]

	s = sumNode()
	s.setwts(effwts)

	print(Dec)

	for i in range(0,len(Dec)):
		p = prodNode()
		s.children.append(p)
		for j in (Dec[i]):
			print(j)
			sub = returnarr(j,scope)
			print(sub)
			if (len(j)<=indsize):
				l = leafNode()
				tempmean = submean(empmean,j)
				tempcov = submat(effcov,j)
				l.scope = sub
				l.create(tempmean,tempcov)
				p.children.append(l)
			else:
				p.children.append(induce(idxst,idxend,maxsize-1,sub,indsize))
		

	return s

#test

s = set(xrange(k))

Tst = induce(0,1000,5,s,3)

print(Tst)

Tst.passon(mean)
print(Tst.retval())
print(Tst.wts)

ref = mn(mean=mean,cov=covmat)
print(ref.logpdf(mean))



