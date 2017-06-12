import networkx as nx
import numpy as np
from scipy.stats import multivariate_normal as mn
from nodes import *
from data import *
from disc import infmat, createpdf

NLT = np.genfromtxt('NLTCS.txt',delimiter="	")
nlt = NLT[:3000,:16]

print(infmat(nlt,16))

nlt = nlt[:,:4]

print(createpdf(nlt,3000,4))



