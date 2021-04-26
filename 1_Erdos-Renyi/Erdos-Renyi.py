import numpy as np
import scipy as sp
import scipy.stats as st

import networkx as nx
import matplotlib.pyplot as plt

from pyvis.network import Network

#%% Generating Graph

for i in range(10):
    n = int(input('Number of nodes: '))
    average_degreee = np.abs(float(input('Expected mean degree: ')))  
    p = average_degreee / n
    if p < 0 or p > 1:
        print('\nThe probability of an edge to be included is not in the interval [0,1].')
    else:
        break

print('\nThe probability of an edge to be included is '+str(p))

matrix = np.zeros((n,n))

G = nx.Graph()
G.add_nodes_from(range(1,n+1))


for i in range(n):
    for j in range(i+1,n):
        if np.random.uniform(low=0, high=1) <= p:
            matrix[i,j] = 1
            G.add_edge(i+1,j+1)


#%% Visualização

# nx.draw(G, with_labels=True)
# plt.show()

net = Network()
net.from_nx(G)

net.show('Erdos-Renyi_n='+str(n)+'_mu='+str(average_degreee)+'.html')

#%% Nodes histogram

degree = np.array(sorted([d for n, d in G.degree()], reverse=True))
bins = np.arange(0, degree.max()+2)-0.5

mu = degree.mean()
sigma = degree.std()


pmf = st.binom.pmf(np.arange(0, degree.max()+1), n, p) 

text = '\n'.join((
        r'$\mu=%.2f$' % (mu, ),
        r'$\sigma=%.2f$' % (sigma, )))


fig = plt.figure(figsize=(8,4.5))
ax = fig.add_axes([0, 0, 1, 1])

plt.scatter(np.arange(0, degree.max()+1), st.binom.pmf(np.arange(0, degree.max()+1), n, p),
            marker='_', s=800, linewidth=2.5, color='#f5ac00', label='Binomial', zorder=2)
plt.hist(degree, bins, label='Graph', edgecolor='black', linewidth=2, density=True, zorder=1)

plt.legend(loc='upper left', fontsize=18)
plt.xticks(range(degree.max()+1), fontsize = 14)
plt.yticks(fontsize = 14)
plt.xlim(bins[0], bins[-1])
plt.xlabel('Degree', fontsize=18)
plt.ylabel('Relative Frequency', fontsize=18)
plt.grid(axis='y')
plt.text(0.97,0.95, text,
         size = 20, verticalalignment='top', horizontalalignment='right',
         color='black', bbox={'facecolor': 'wheat', 'alpha': 1, 'pad': 0.5, 'boxstyle': 'round'},
         transform = ax.transAxes)
# plt.show()
plt.savefig('Degree_RelativeFrequency_n='+str(n)+'_mu='+str(average_degreee)+'.png',
            dpi=200, bbox_inches='tight')
