import numpy as np

import networkx as nx
import matplotlib.pyplot as plt

from pyvis.network import Network

#%% Inicialização
G = nx.path_graph(6)
center_nodes = {0, 3}
cells = nx.voronoi_cells(G, center_nodes)
partition = set(map(frozenset, cells.values()))
sorted(map(sorted, partition))


#%% Visualização

nx.draw(G, with_labels=True)
plt.show()

net = Network()
net.from_nx(G)

net.show('Erdos-Renyi.html')