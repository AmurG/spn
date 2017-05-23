import math
import numpy as np
import scipy
import networkx as nx
from scipy.stats import multivariate_normal as mn
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hcluster

#gen = np.random.multivariate_normal(mean,cov,10)

#print((gen))
#print(np.shape((gen)))

def split(arr,thresh):
	clusters = hcluster.fclusterdata(arr, thresh, criterion="distance")
	print(clusters)
	big = []
	for i in range(0,len(set(clusters))):
		small = []
		for j in range(0,len(arr)):
			if (clusters[j]==i+1):
				small.append(arr[j])
		big.append(small)
		#print(big)
	return big
	
#test = split((gen))

#print(test)
#print(len(test))


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

ref = mn(mean=mean,cov=cov)

#print(np.shape(data))

#print(np.shape(gen))

gen = []
for i in range(0,1000):
	gen.append(np.random.multivariate_normal(-mean,cov))
for i in range(1000,2000):
	gen.append(np.random.multivariate_normal(-7*mean,cov))
for i in range(2000,3000):
	gen.append(np.random.multivariate_normal(11*mean,cov))

gen = np.asarray(gen)

data = np.transpose(gen)

estcov = np.corrcoef(np.transpose(gen))
estcov2 = np.cov(np.transpose(gen))

print(np.shape(estcov))

wts = np.zeros(k)

print(-estcov)

G = nx.from_numpy_matrix(-abs(estcov))
G = G.to_undirected()

T=nx.minimum_spanning_tree(G)
Order = np.asarray(T.edges(data='weight'))
Order = Order[Order[:,2].argsort()]
Dec = []
Dec.append(list(nx.connected_components(T)))

for i in range(0,k-1):
	print(np.shape(Order))
	sum = 0
	for j in range(0,len(Order)):
		sum = sum + Order[j,2]
	wts[i] = sum
	print(Order)
	idx = int(Order[len(Order)-i-1,0])
	idx2 = int(Order[len(Order)-i-1,1])
	print(idx,idx2)
	T.remove_edge(idx,idx2)
	Dec.append(list(nx.connected_components(T)))


PDF = []
print(Dec)

for i in range(0,len(Dec)):
	subpdf = []
	for j in (Dec[i]):
		m = submean(mean,j)
		#print(m)
		c = submat(estcov2,j)
		#print(c)
		subpdf.append(mn(mean=m,cov=c))
	PDF.append(subpdf)
	print(PDF)

wts[k-1] = 0.2

run = 0

#wts[0] = 0

for i in range(0,k-1):
	run = run+wts[i]

for i in range(0,k-1):
	wts[i] = (wts[i]/run)*0.8

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
			#print(Dec[i][j])
			pd = pd*(fix.pdf(submean(x,Dec[i][j])))
			#print(pd)
		pdf = pdf + pd*wt
		#print(pdf)
	return pdf

#samples = np.random.multivariate_normal(mean,estcov2,1000)

gen = []
for i in range(0,1000):
	gen.append(np.random.multivariate_normal(-mean,cov) + np.random.multivariate_normal(-0.00002*mean,cov))
for i in range(1000,2000):
	gen.append(np.random.multivariate_normal(-7*mean,cov)+ np.random.multivariate_normal(0.00001*mean,cov))
for i in range(2000,3000):
	gen.append(np.random.multivariate_normal(11*mean,cov)+ np.random.multivariate_normal(0.00001*mean,cov))

samples = np.asarray(gen)

pdf1 = np.zeros(3000)
pdf2 = np.zeros(3000)

ref = mn(mean=mean,cov=estcov2)

corr = 0
corr1 = 0

for i in range(0,3000):
	pdf1[i] = np.log(ref.pdf(samples[i]))
	corr = corr + pdf1[i]
	pdf2[i] = np.log(comppdf(samples[i]))
	corr1 = corr1 + pdf2[i]

plt.plot(pdf1)
plt.plot(pdf2)
plt.show()

print(corr,corr1)






