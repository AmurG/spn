import math
import numpy as np
import scipy
import networkx as nx
from scipy.stats import multivariate_normal as mn
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hcluster
from nodes import *
from data import *
from sklearn import mixture
import matplotlib.pyplot as plt
from scipy.cluster.vq import vq, kmeans, whiten

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

def induce(tempdat,maxsize,scope,indsize,flag):
	full = len(tempdat)
	if (flag==0):
		if (full>=30*len(scope)):
			tempdat2 = split(tempdat,3)
			s = sumNode()
			arr = []
			for i in range(0,len(tempdat2)):
				if(len(tempdat2[i])>=(len(scope))):
					arr.append(len(tempdat2[i]))
					s.children.append(induce(np.asarray(tempdat2[i]),maxsize,scope,indsize,1))
			s.setwts(arr)
			return s
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
		wts[i] = sum - np.log(k-i) + np.log(k)
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
	print(effwts)

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
				p.children.append(induce(tempdat,maxsize-1,sub,indsize,0))
		

	return s

#test

s = set(xrange(4))

ab = np.genfromtxt('QU.dat',delimiter=",")
ab = np.asarray(ab[:,:4])
print(ab[0])
ab = whiten(ab)
ab = np.random.permutation(ab)

gmix = mixture.GMM(n_components=3, covariance_type='full')
gmix.fit(ab[:1600,:])

Tst = induce(ab[:1600,:],3,s,2,0)


for i in range(0,7000):
	idx = np.random.randint(0,1600)
	Tst.passon(ab[idx])
	placeholder = Tst.retval()
	Tst.update()

sum = 0

plot1 = np.zeros(400)

for i in range(1600,2000):
	Tst.passon(ab[i])
	sum = sum + Tst.retval()
	plot1[i-1600] = Tst.retval()

print(Tst.wts)
print(sum/400)
print(np.mean((gmix.score(ab[1600:2000,:]))))
plt.plot(plot1)
plt.plot(gmix.score(ab[1600:2000,:]))
plt.show()

