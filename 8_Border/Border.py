import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

from datetime import datetime

import osmnx as ox
import networkx as nx
from pyvis.network import Network
import Networks
import os

#%%
def Calculate_Accessibility(nodes, connections, h, filename, outputfilename):
    
    # Escrevendo arquivo .xnet
    with open(filename, 'w+') as datafile:
        datafile.write('#vertices '+str(len(nodes))+' nonweighted\n')
    datafile.closed
    
    for i in nodes:
        with open(filename, 'a+') as datafile:
            datafile.write('"'+str(i)+'"\n')
        datafile.closed
        
    with open(filename, 'a+') as datafile:
            datafile.write('#edges nonweighted undirected\n')
    datafile.closed
    
    for connection in connections:
        with open(filename, 'a+') as datafile:
            datafile.write('{:} {:}\n'.format(*connection))
        datafile.closed
        
    # Rodando script (https://github.com/filipinascimento/CVAccessibility)
    os.system("./CVAccessibility -l " + str(h) + " " + filename + " " + outputfilename)
    
    return np.genfromtxt(outputfilename)

#%%
def Plot_Accessibility(accessibility, h, tag, dt_string):
    bins = np.arange(0, accessibility.max()+2)-0.5
            
    fig = plt.figure(figsize=(8,4.5))
    
    plt.hist(accessibility, bins, color='#3971cc', edgecolor='#303030',
             linewidth=1)
    
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.xlim(bins[0], bins[-1])
    plt.xlabel('Accessibility', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.grid(axis='y')
    
    plt.savefig('img/' + tag + dt_string + '_Accessibility_Histogram_h='+str(h)+'.png',
                dpi=200, bbox_inches='tight')
    
#%%
def Plot_Spatial(network, border, center, tag, dt_string, h):
    
    plt.figure(figsize=(16,16))
    for i in range(np.size(network.connection,0)):
                plt.plot(network.coord[network.connection[i]][:,0],
                         network.coord[network.connection[i]][:,1],
                         '-', color='black', zorder=1)
    plt.scatter(network.coord[border,0], network.coord[border,1], s=40, color='crimson', zorder=2)
    plt.scatter(network.coord[center,0], network.coord[center,1], s=40, color='navy', zorder=2)
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
        
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.savefig('img/Border_Plot_'+ tag + '_' + dt_string + '_h=' + str(h) + '.png',
                dpi=200, bbox_inches='tight')
    
#%%
def Plot_Non_Spatial(Graph, border, center, tag, dt_string, h):
    net = Network()
    for n in Graph.nodes():
        if n in border[0]:
            color = "#FF0000"
        else:
            color = "#A0CBE2"
        net.add_node(n, label=n, color=color)
        print(n)
    for e in Graph.edges():
        net.add_edge(int(e[0]), int(e[1]))

    net.show_buttons()
    net.show('img/Border_Plot_'+ tag + '_' + dt_string + '_h=' + str(h) + '.html')

#%%
# Definindo parâmetros das diversas redes
n = 100
average_degree = 4

r = 0.032

beta = 0.7
alpha = np.sqrt(2*np.pi*(n-1)*beta/average_degree)-1.5

p1 = 0.001
p2 = 0.01
p3 = 0.1

n_0 = 10
average_degree_0 = 3
m = 3

# Definindo parâmetros da simulação
dt_string = datetime.now().strftime("_%d-%m-%Y_%H-%M-%S")

tags = {
        'ER'    : [n, average_degree],
        'Vor'   : [n],
        'Geo'   : [n, r],
        'Wax'   : [n, alpha, beta],
        'WS-1'  : [n, average_degree, p1],
        'WS-2'  : [n, average_degree, p2],
        'WS-3'  : [n, average_degree, p3],
        'BA' : [n_0, average_degree_0, n, m]
       }

#%%
#Gerando a rede
# for tag in ['Vor']:
for tag in tags.keys():

    if tag == 'WS-1' or tag == 'WS-2' or tag == 'WS-3': # Caso especial da Watts-Strogatz, que é gerada para diferentes parâmetros
        method_to_call = getattr(Networks, 'WS_network')
    else: # Outros modelos
        method_to_call = getattr(Networks, tag+'_network')
    
    print(tag)
        
    network = method_to_call(*tags[tag])
    Graph = network.G
    filename = 'files/' + tag + dt_string + '.xnet'
    
    for h in [2,3,4]:
        outputfilename = 'files/' + 'output_' + tag + '_' + dt_string + '_h=' + str(h) + '.txt'
        accessibility = Calculate_Accessibility(np.array(list(Graph.nodes())), np.array(list(Graph.edges())), h, filename, outputfilename)
        Plot_Accessibility(accessibility, h, tag, dt_string)
    
        p = np.percentile(accessibility, 25)
        
        border = np.where(accessibility <= p)
        center = np.where(accessibility > p)
            
        if tag in ['Vor', 'Geo', 'Wax']:
            Plot_Spatial(network, border, center, tag, dt_string, h)
        else:
            Plot_Non_Spatial(Graph, border, center, tag, dt_string, h)
