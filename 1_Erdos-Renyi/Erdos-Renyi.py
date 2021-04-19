import numpy as np

import networkx as nx
import matplotlib.pyplot as plt

from pyvis.network import Network

#%% Inicialização

# n = int(input('Quantos nós? '))
# p = int(input('Qual a probabilidade? '))

p = 0.2
n = 10

matrix = np.zeros((n,n))

G = nx.Graph()
G.add_nodes_from(range(1,n+1))


for i in range(n):
    for j in range(i+1,n):
        if np.random.uniform(low=0, high=1) <= p:
            matrix[i,j] = 1
            G.add_edge(i+1,j+1)


#%% Visualização

nx.draw(G, with_labels=True)
plt.show()

net = Network()
net.from_nx(G)

net.show('Erdos-Renyi.html')