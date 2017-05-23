import networkx as nx
import numpy as np
from scipy.stats import multivariate_normal as mn
from nodes import *

G = nx.path_graph(4)
G.add_edge(5,6)
graphs = list(nx.connected_component_subgraphs(G))

print(graphs)

Gc = min(nx.connected_component_subgraphs(G), key=len)

print(Gc.number_of_nodes())

s = sumNode()
p = prodNode()
s.children.append(p)
p = prodNode()
s.children.append(p)
print(s.children)

print(set(xrange(10)))

test = Node()
test.scope = set(xrange(5))
print(test.scope)




