import math
import numpy as np
import scipy
import networkx as nx
from scipy.stats import multivariate_normal as mn
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hcluster
import nodes

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

def induce(idx1,idx2):
	effdat = operdat[idx1:idx2]
	effcorr = np.corrcoef(np.transpose(effdat))
	effcov = np.cov(np.transpose(effdat))



