import numpy as np
from nodes import *
from modul import returnarr
from data import *
from sklearn.metrics import adjusted_mutual_info_score as ami

def infmat(mat,nvar):
	retmat = np.zeros(nvar*nvar)
	retmat = np.reshape(retmat,(nvar,nvar))
	for i in range(0,nvar):
		for j in range(0,nvar):
			if (i>j):
				retmat[i][j] = retmat[j][i]
			else:
				retmat[i][j] = ami(mat[:,i],mat[:,j])
	return retmat

def createpdf(mat,nsam,nvar):
	length = int(np.rint(np.power(2,nvar)))
	pdf = np.zeros(length)
	for i in range(0,nsam):
		print(mat[i,:])
		idx = bintodec(mat[i,:])
		print(idx)
		pdf[idx] = pdf[idx] + float((0.8)/nsam)
	for i in range(0,length):
		pdf[i] = pdf[i] + float(0.2/float(length))
	return pdf


def induce(tempdat,maxsize,scope,indsize,flag):
	full = len(tempdat)
	
	if (flag==0):
		if (full>=30*len(scope)):
			tempdat2 = split(tempdat,8)
			s = sumNode()
			arr = []
			for i in range(0,len(tempdat2)):
				if(len(tempdat2[i])>=(len(scope))):
					arr.append(len(tempdat2[i]))
					s.children.append(induce(np.asarray(tempdat2[i]),maxsize,scope,indsize,1))
			s.setwts(arr)
			print("wts are",arr)
			return s
	
	effdat = np.zeros(len(tempdat)*len(scope))
	effdat = np.reshape(effdat,(len(tempdat),len(scope)))
	for i in range(0,len(tempdat)):
		temp = submean(tempdat[i],scope)
		for j in range(0,len(scope)):
			effdat[i][j] = temp[j]

	fisher = infmat(effdat,len(scope))
	G = nx.from_numpy_matrix(-(fisher))
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
			sub = returnarr(j,scope)
			if (len(j)<=indsize):
				l = discNode()
				l.scope = sub
				pdf = createpdf(tempdat[:,sorted(list(sub))],len(tempdat),len(sub))
				l.create(pdf)
				p.children.append(l)
			else:
				p.children.append(induce(tempdat,maxsize-1,sub,indsize,0))
		

	return s

NLT = np.genfromtxt('../NLTCS.txt',delimiter="	")
nlt = NLT[:3000,:16]
print(nlt[0])

s = set(xrange(16))

Tst = induce(nlt[:2000,:],8,s,4,0)

for i in range(0,10000):
	idx = np.random.randint(0,2000)
	Tst.passon(nlt[idx])
	placeholder = Tst.retval()
	Tst.update()

sum = 0

plot1 = np.zeros(1000)

for i in range(2000,3000):
	Tst.passon(nlt[i])
	sum = sum + Tst.retval()
	plot1[i-2000] = Tst.retval()

print(Tst.wts)
print(sum/1000)


