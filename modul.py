import math
import numpy as np
import scipy
import networkx as nx
from scipy.stats import multivariate_normal as mn
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hcluster
import nodes
import data

k = 10

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

def induce(idx1,idx2,maxsize):
	effdat = operdat[idx1:idx2]
	effcorr = np.corrcoef(np.transpose(effdat))
	effcov = np.cov(np.transpose(effdat))

	G = nx.from_numpy_matrix(-abs(estcov))
	G = G.to_undirected()

	Dec = []

	T=nx.minimum_spanning_tree(G)
	Order = np.asarray(T.edges(data='weight'))
	k = len(Order)
	wts = np.zeros(k)
	Order = Order[Order[:,2].argsort()]
	Dec = []
	Dec.append(list(nx.connected_components(T)))

	for i in range(0,k-1):
		sum = 0
		for j in range(0,len(Order)):
			sum = sum + Order[j,2]
		wts[i] = sum
		idx = int(Order[len(Order)-i-1,0])
		idx2 = int(Order[len(Order)-i-1,1])
		T.remove_edge(idx,idx2)
		Dec.append(list(nx.connected_components(T)))


	PDF = []
	for i in range(0,len(Dec)):
		subpdf = []
		for j in (Dec[i]):
			m = submean(mean,j)
			#print(m)
			c = submat(estcov2,j)
			#print(c)
			subpdf.append(mn(mean=m,cov=c))
		PDF.append(subpdf)

	



