import networkx as nx

G = nx.path_graph(4)
G.add_edge(5,6)
graphs = list(nx.connected_component_subgraphs(G))

print(graphs)

Gc = min(nx.connected_component_subgraphs(G), key=len)

print(Gc.nodes())

