import math
import numpy as np
import scipy
import networkx as nx
from scipy.stats import multivariate_normal as mn

def submat(mat,subset):
	q = len(subset)
	print(q)
	ret = np.zeros(q*q)
	ret = np.reshape(ret,(q,q))
	w = 0
	for i in subset:
		z = 0
		for j in subset:
			ret[w,z] = mat[i,j]
			z=z+1
		w=w+1
	return ret

def submean(mean,subset):
	q = len(subset)
	m = np.zeros(q)
	w = 0
	for i in subset:
		m[w] = mean[i]
		w=w+1
	return m

k = 4

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

gen = np.random.multivariate_normal(mean,cov,100)

#print(np.shape(gen))

estcov = np.corrcoef(np.transpose(gen))
estcov2 = np.cov(np.transpose(gen))

print(-estcov)

G = nx.from_numpy_matrix(-estcov)
G = G.to_undirected()

T=nx.minimum_spanning_tree(G)
Dec = []

for i in range(0,k-1):
	Order = np.asarray(T.edges(data='weight'))
	Dec.append(list(nx.connected_components(T)))
	print(np.shape(Order))
	Order = Order[Order[:,2].argsort()]
	iter = len(Order)
	print(Order)
	idx = int(Order[len(Order)-1,0])
	idx2 = int(Order[len(Order)-1,1])
	print(idx,idx2)
	T.remove_edge(idx,idx2)

cset = Dec[1][0]

print(cset)

print(submat(estcov,cset))








