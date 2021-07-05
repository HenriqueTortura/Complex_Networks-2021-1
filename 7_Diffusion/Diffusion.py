import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

from datetime import datetime

import networkx as nx
from pyvis.network import Network
import Networks

#%%
def Activation(Graph, walk_lenght, filename='Activation', verbose=False):
    
    signal = np.zeros(walk_lenght, dtype=int)
    activation = np.zeros((n, walk_lenght))
    
    # Caminhada aleatória
    for i in range(walk_lenght-1):
        if verbose:
            print(str(i+1)+'/'+str(walk_lenght-1))
    
        # Escolhe um nó vizinho a partir de uma distribuição uniforme
        signal[i+1] = np.random.choice(np.array(list(Graph.adj[signal[i]])), 1, replace=False)
        
        # Salva a ativação de cada nó em cada passo da caminhada
        activation[:,i+1] = activation[:,i]
        activation[signal[i+1],i+1] += 1   

    # Plotando
    x = np.array(list(Graph.degree()))[:,1]
    pearson = pearsonr(x, activation[:,-1])
    label = 'Pearson Correlation Coefficient: '+'{:.3}'.format(pearson[0])+'\n'+r'$p=$'+'{:.3e}'.format(pearson[1])
    
    plt.figure(figsize=(8,4.5))
    
    plt.scatter(x, activation[:,-1], label = label)
    
    plt.legend(loc='best')        
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.xlabel('Degree', fontsize=18)
    plt.ylabel('Activation', fontsize=18)
    plt.grid()
    
    plt.savefig('img/Activation_'+filename+'.png',
                dpi=200, bbox_inches='tight')
    
    return signal, activation, pearson

#%%
def Plot_Transient(activation, Graph, filename='Activation'):
    x = np.array(list(Graph.degree()))[:,1]
    
    b = (np.max(x)*np.min(activation) - np.min(x)*np.max(activation))/(np.max(x) - np.min(x))
    expected = (np.max(activation) - np.min(activation))/(np.max(x) - np.min(x)) * x + b
    
    fig, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]})
    
    a0.imshow(activation, aspect='auto', interpolation='nearest',
              origin='lower')
    a1.imshow(expected[:,np.newaxis], aspect='auto', interpolation='nearest',
              origin='lower')
    
    a0.axis('off')
    a1.axis('off')
    plt.savefig('img/Transient_'+filename+'.png',
                dpi=200, bbox_inches='tight')
    
#%%
def Reconnect(Graph, reconnection):
    all_edges = np.array(list(Graph.edges), dtype=int)
    
    number_of_edges_to_delete = int(reconnection*np.size(all_edges, axis=0))
    edges_to_delete = np.random.choice(np.size(all_edges, axis=0), number_of_edges_to_delete,
                                       replace=False)
    for i in edges_to_delete:
        edge = all_edges[i,:] 
        forbidden = np.unique(np.concatenate((all_edges[np.where(all_edges[:,0]==all_edges[i,0]),1],
                                              all_edges[np.where(all_edges[:,1]==all_edges[i,0]),0]), axis = 1))
        allowed = np.delete(np.array(list(Graph.nodes)), forbidden)
        
        Graph.remove_edge(*edge)
        Graph.add_edge(edge[0], *np.random.choice(allowed, 1, replace=False))
    
    return Graph
        
#%%
# Definindo parâmetros das diversas redes
n = 500
average_degree = 6

r = 0.032

beta = 0.7
alpha = np.sqrt(2*np.pi*(n-1)*beta/average_degree)-1.5

p = 0.5
p1 = 0.001
p2 = 0.01
p3 = 0.1

n_0 = 10
average_degree_0 = 1.5
m = 3

dt_string = datetime.now().strftime("_%d-%m-%Y_%H-%M-%S")

tags = {    
        'ER'    : [n, average_degree],
        'Vor'   : [n],
        'Geo'   : [n, r],
        'Wax'   : [n, alpha, beta],
        'WS-1'  : [n, average_degree, p1],
        'WS-2'  : [n, average_degree, p2],
        'WS-3'  : [n, average_degree, p3],
        'BA'    : [n_0, average_degree_0, n, m]
       }

# for tag in tags.keys(): # Percorre cada modelo
# for tag in ['Geo']:
tag = 'BA'
   
if tag == 'WS-1' or tag == 'WS-2' or tag == 'WS-3': # Caso especial da Watts-Strogatz, que é gerada para diferentes parâmetros
    method_to_call = getattr(Networks, 'WS_network')
else: # Outros modelos
    method_to_call = getattr(Networks, tag+'_network')

filename = tag+'_'+dt_string

print(tag)
    
Graph = method_to_call(*tags[tag]).G

# Caminhada aleatória de grafo não-dirigido
walk_lenght = 100000
signal, activation, pearson = Activation(Graph, walk_lenght, filename=filename, verbose=False)
Plot_Transient(activation, Graph, filename= filename)

Graph = Graph.to_directed()

#%%
# Caminhada aleatória de grafo reconectado (0.1)
G_r = Reconnect(Graph, 0.01)
filename_r = filename + '_reconnection1'
signal, activation_r, pearson = Activation(G_r, walk_lenght, filename=filename_r, verbose=False)
Plot_Transient(activation_r, G_r, filename= filename_r)

#%%
# Caminhada aleatória de grafo reconectado (0.5)
G_r = Reconnect(Graph, 0.05)
filename_r = filename + '_reconnection2'
signal, activation_r, pearson = Activation(G_r, walk_lenght, filename=filename_r, verbose=False)
Plot_Transient(activation_r, G_r, filename= filename_r)

#%%
# Caminhada aleatória de grafo reconectado (0.9)
G_r = Reconnect(Graph, 0.1)
filename_r = filename + '_reconnection3'
signal, activation_r, pearson = Activation(G_r, walk_lenght, filename=filename_r, verbose=False)
Plot_Transient(activation_r, G_r, filename= filename_r)
