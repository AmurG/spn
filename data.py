import math
import numpy as np
import scipy
import networkx as nx
from scipy.stats import multivariate_normal as mn
import matplotlib.pyplot as plt

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

k = 5

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

ref = mn(mean=mean,cov=cov)

gen = np.random.multivariate_normal(mean,cov,10000)

#print(np.shape(gen))

estcov = np.corrcoef(np.transpose(gen))
estcov2 = np.cov(np.transpose(gen))

wts = np.zeros(k)

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
	sum = 0
	for j in range(0,len(Order)):
		sum = sum + Order[j,2]
	wts[i] = sum
	iter = len(Order)
	print(Order)
	idx = int(Order[len(Order)-1,0])
	idx2 = int(Order[len(Order)-1,1])
	print(idx,idx2)
	T.remove_edge(idx,idx2)

Dec.append(list(nx.connected_components(T)))
PDF = []
print(Dec)

for i in range(0,len(Dec)):
	subpdf = []
	for j in (Dec[i]):
		m = submean(mean,j)
		print(m)
		c = submat(estcov2,j)
		print(c)
		subpdf.append(mn(mean=m,cov=c))
	PDF.append(subpdf)
	print(PDF)

wts[k-1] = 0.1

run = 0

#wts[0] = 0

for i in range(0,k-1):
	run = run+wts[i]

for i in range(0,k-1):
	wts[i] = (wts[i]/run)*0.9

print(wts)
print(len(PDF))
print(PDF)

def comppdf(x):
	pdf = 0
	for i in range(0,k):
		wt = wts[i]
		pd = 1
		for j in range(0,i+1):
			fix = PDF[i][j]
			print(Dec[i][j])
			pd = pd*(fix.pdf(submean(x,Dec[i][j])))
			print(pd)
		pdf = pdf + pd*wt
		print(pdf)
	return pdf

samples = np.random.multivariate_normal(mean,estcov2,1000)

pdf1 = np.zeros(1000)
pdf2 = np.zeros(1000)

for i in range(0,1000):
	pdf1[i] = ref.pdf(samples[i])
	pdf2[i] = comppdf(samples[i])

plt.plot(pdf1)
plt.plot(pdf2)
plt.show()






