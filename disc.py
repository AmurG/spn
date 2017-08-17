import numpy as np
import nodes as nd
import networkx as nx
from nodes import *
from modul import returnarr
from data import *
from sklearn.metrics import adjusted_mutual_info_score as ami
from time import time
from sklearn import metrics
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans


Leafcount = 0

def infmat(mat,nvar):
	retmat = np.zeros(nvar*nvar)
	retmat = np.reshape(retmat,(nvar,nvar))
	for i in range(0,nvar):
		for j in range(0,nvar):
			if (i>j):
				retmat[i][j] = retmat[j][i]
			else:
				retmat[i][j] = (float(np.dot(mat[:,i],mat[:,j])*np.dot(mat[:,i],mat[:,j]) + 1e-4))/(float(np.dot(mat[:,i],mat[:,i])*np.dot(mat[:,j],mat[:,j]) + 1e-2))
				#temp = np.corrcoef(retmat[:,i],retmat[:,j])
				#retmat[i][j] = abs(temp[0][1])
	return retmat

def createpdf(mat,nsam,nvar):
	length = int(np.rint(np.power(2,nvar)))
	pdf = np.zeros(length)
	for i in range(0,nsam):
		#print(mat[i,:])
		idx = bintodec(mat[i,:])
		#print(idx)
		pdf[idx] = pdf[idx] + float((0.95)/nsam)
	for i in range(0,length):
		pdf[i] = pdf[i] + float(0.05/float(length))
	return pdf


def induce(tempdat,maxsize,scope,indsize,flag,maxcount):
	full = len(tempdat)
	
	if (flag==0):
		if (full>=4000):
			tempdat2 = split(tempdat,3)
			s = sumNode()
			arr = []
			cnt = 0
			for i in range(0,len(tempdat2)):
				if(len(tempdat2[i])>=(2*len(scope))):
					arr.append(len(tempdat2[i]))
					s.children.append(induce(np.asarray(tempdat2[i]),maxsize,scope,indsize,1,maxcount))
					cnt = cnt + 1
			
			for i in range(0,cnt):
				chosen = s.children[i]
				w = 0
				for j in chosen.children:
					s.children.append(j)
					arr.append(chosen.wts[w]*arr[i])
					w = w+1
			arr = arr[cnt:]
			s.children = s.children[cnt:]			
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
	G = nx.from_numpy_matrix(-abs(fisher))
	G = G.to_undirected()

	Dec = []

	T=nx.minimum_spanning_tree(G)
	Order = np.asarray(T.edges(data='weight'))
	k = len(Order)
	#wts = np.zeros(k)
	Order = Order[Order[:,2].argsort()]
	Dec = []
	Gc = max(nx.connected_component_subgraphs(T), key=len)
	n = Gc.number_of_nodes()
	if(n<=maxsize):
		Dec.append(list(nx.connected_components(T)))

	count = 0
	
	for i in range(0,k):
		if(count>maxcount):
			break
		#sum = 0
		#for j in range(0,len(Order)-i):
		#	sum = sum - Order[j,2]
		#wts[i] = sum 
		idx = int(Order[len(Order)-i-1,0])
		idx2 = int(Order[len(Order)-i-1,1])
		T.remove_edge(idx,idx2)
		Gc = max(nx.connected_component_subgraphs(T), key=len)
		n = Gc.number_of_nodes()
		if((n<=maxsize)and(count<=maxcount)):
			Dec.append(list(nx.connected_components(T)))
			count = count + 1

	effwts = np.zeros(len(Dec))
	for i in range(0,len(Dec)):
		effwts[i] = 1./len(Dec)

	s = sumNode()
	s.setwts(effwts)
	print(effwts)

	print(Dec)

	for i in range(0,len(Dec)):
		if(len(Dec[i])>1):	
			p = prodNode()
			s.children.append(p)
			for j in (Dec[i]):
				sub = returnarr(j,scope)
				if (len(j)<=indsize):
					l = discNode()
					global Leafcount
					Leafcount = Leafcount+1
					l.scope = sub
					pdf = createpdf(tempdat[:,sorted(list(sub))],len(tempdat),len(sub))
					l.create(pdf)
					p.children.append(l)
				else:
					p.children.append(induce(tempdat,maxsize/2,sub,indsize,0,maxcount))
	
	if(len(scope)<=indsize):
		l = discNode()
		global Leafcount
		Leafcount = Leafcount+1
		l.scope = scope
		pdf = createpdf(tempdat[:,sorted(list(scope))],len(tempdat),len(scope))
		l.create(pdf)
		s.children.append(l)

	return s

NLT = np.genfromtxt('../retail.data',delimiter=",")

print(np.shape(NLT))
nlt = NLT[:22000,:135]
print(nlt[0])

'''
for i in range(2,10):
	clusterer = KMeans(n_clusters=i, random_state=10).fit(nlt)
	cluster_labels = clusterer.labels_
	print(metrics.calinski_harabaz_score(nlt, cluster_labels))
'''


s = set(xrange(135))

Tst = induce(nlt[:22000,:135],30,s,1,0,4)

Tst.normalize()

globalcallid = 0
nodearr = []
edgarr = []

nodearr.append(-1)
nodearr.append(Tst.kind)
nodearr.append(globalcallid)

Tst.createid = globalcallid

globalcallid = globalcallid + 1 

def onepush(givennode,arr1,arr2):
	global globalcallid
	w = 0
	for j in givennode.children:
		if(j.kind==2):
			arr1.append(globalcallid)
			arr1.append(j.scope)
			arr1.append(j.arr[0])
			j.createid = globalcallid
			globalcallid = globalcallid + 1	
		else:
			arr1.append(-1)
			arr1.append(j.kind)
			arr1.append(globalcallid)
			j.createid = globalcallid
			globalcallid = globalcallid + 1
		if(givennode.kind==0):
			arr2.append(-1)
			arr2.append(givennode.createid)
			arr2.append(j.createid)
		elif(givennode.kind==1):
			arr2.append(givennode.createid)
			arr2.append(j.createid)
			arr2.append(givennode.wts[w])
		w = w+1
	return

def wrapper(listofnodes,arr1,arr2):
	biglist = []
	count = 0
	for node in listofnodes:
		onepush(node,arr1,arr2)
		for j in node.children:
			if(j.kind!=2):
				biglist.append(j)
				count = count+1
	return biglist,count

def bigfun(starter,arr1,arr2):
	dummy = []
	dummy.append(starter)
	get,getcount = wrapper(dummy,arr1,arr2)
	while(getcount!=0):
		get,getcount = wrapper(get,arr1,arr2)
	return

bigfun(Tst,nodearr,edgarr)

print(nodearr[:200])
print(edgarr[:200])

print(len(nodearr))
print(Leafcount)


'''
nodes = []
leaves = []
sumedges = []
prodedges = []



def pushtoarr(givennode,arr1,arr2):
	global globalcallid
	if(givennode.kind==2):
		arr2.append(globalcallid)
		arr2.append(givennode.scope)
		arr2.append(givennode.arr[0])
		givennode.createid = globalcallid
		globalcallid = globalcallid + 1	
		return	
	else:
		arr1.append(globalcallid)
		arr1.append(givennode.kind)
		givennode.createid = globalcallid
		globalcallid = globalcallid + 1
		for j in givennode.children:
			pushtoarr(j,arr1,arr2)

def edgtoarr(givennode,arr1,arr2):
	if(givennode.kind==0):
		for j in givennode.children:
			arr2.append(givennode.createid)
			arr2.append(j.createid)
			edgtoarr(j,arr1,arr2)
	if(givennode.kind==1):
		w = 0
		for j in givennode.children:
			arr1.append(givennode.createid)
			arr1.append(j.createid)
			arr1.append(givennode.wts[w])
			w = w+1
			edgtoarr(j,arr1,arr2)
	else:
		return	

pushtoarr(Tst,nodes,leaves)

edgtoarr(Tst,sumedges,prodedges)

print(sumedges[:100])
print(prodedges[:100])
print(nodes[:100])
print(leaves[:100])

print(len(sumedges))
print(len(prodedges))
print(len(nodes))
print(len(leaves))
'''

file = open('./retcheck1.txt', 'w')
file.write('##NODES##\n')
for i in range(0,len(nodearr)/3):
	call = nodearr[3*i]
	if(call==(-1)):
		file.write(str(nodearr[3*i+2])+','+convert(nodearr[3*i+1])+'\n')
	else:
		aux = np.asarray(list(nodearr[3*i+1]))
		p = nodearr[3*i+2]
		file.write(str(nodearr[3*i])+','+'BINNODE'+','+str(aux[0])+','+str(p)+','+str(1-p)+'\n')
file.write('##EDGES##\n')
for i in range(0,len(edgarr)/3-1):
	call = edgarr[3*i]
	if(call==(-1)):
		file.write(str(edgarr[3*i+1])+','+str(edgarr[3*i+2])+'\n')
	else:
		file.write(str(edgarr[3*i])+','+str(edgarr[3*i+1])+','+str(edgarr[3*i+2])+'\n')

#change this bit

i = len(edgarr)/3-1
call = edgarr[3*i]
if(call==(-1)):
	file.write(str(edgarr[3*i+1])+','+str(edgarr[3*i+2])+'\n')
else:
	file.write(str(edgarr[3*i])+','+str(edgarr[3*i+1])+','+str(edgarr[3*i+2])+'\n')

file.close()

'''


for i in range(0,22000):
	t = time()
	idx = np.random.randint(0,22000)
	nd.globalarr = nlt[idx]
	Tst.passon()
	print("t1",time()-t)
	t = time()
	print(Tst.retval())
	print("t2",time()-t)
	t = time()
	Tst.update()
	print("t3",time()-t)

sum = 0

plot1 = np.zeros(2900)

nlt = np.genfromtxt('../retail2.data',delimiter=",")
nlt = nlt[:2900,:135]

for i in range(0,2900):
	nd.globalarr = nlt[i]
	Tst.passon()
	print(Tst.retval())
	sum = sum + Tst.retval()
	plot1[i] = Tst.retval()

print(Tst.wts)
print(sum/2900)
'''

