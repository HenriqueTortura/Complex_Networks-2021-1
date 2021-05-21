import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from datetime import datetime
import Networks

import networkx as nx
from pyvis.network import Network


#%% Classe da rede de Barabasi-Albert

class BarabasiAlbert_network(object):
    
    def __init__(self, n_0, average_degreee_0, n, m, degree_only=False):
        
        # Define os principais parâmetros
        self.n_0 = n_0
        self.average_degreee_0 = average_degreee_0
        self.n = n
        self.m = m
        
        # Cria grafo inicial como uma rede de Erdos-Renyi
        self.G = Networks.ErdosRenyi_network(self.n_0, self.average_degreee_0).G
        new_edge = np.zeros(self.m)
        
        # Adiciona novo nó
        for j in range(1, self.n-self.n_0+1):
            
            # print(str(j)+'/'+str(self.n-self.n_0))
            self.G.add_node(self.n_0+j)
            
            # Obtem nós com seus respectivos graus
            possible_edges = np.array(self.G.degree, dtype=float)[:,]
            # Normaliza
            possible_edges[:,1] = possible_edges[:,1]/np.sum(possible_edges[:,1])
            
            # Realiza nova conexão
            for i in range(self.m):
                # Escolhe conexão
                new_edge[i] = np.random.choice(possible_edges[:,0], p=possible_edges[:,1])
                # print(new_edge[i])
                
                # Elimina nó escolhido
                possible_edges = np.delete(possible_edges, np.where(possible_edges[:,0] == new_edge[i]), axis=0)
                
                # Renormaliza
                possible_edges[:,1] = possible_edges[:,1]/np.sum(possible_edges[:,1])
                
                self.G.add_edge(self.n_0+j, new_edge[i])
       
        # Obtem a média e o desvio padrão do grau
        self.degree = np.array(sorted([d for n, d in self.G.degree()], reverse=True))
        self.degree_mu = self.degree.mean()
        self.degree_sigma = self.degree.std()
        
        if not degree_only:
            # Obtem a média e o desvio padrão do coeficiente de aglomeração
            self.clustering_coefficient = np.array(sorted([nx.clustering(self.G,n) for n in nx.clustering(self.G)],
                                                          reverse=True))
            self.clustering_coefficient_mu = self.clustering_coefficient.mean()
            self.clustering_coefficient_sigma = self.clustering_coefficient.std()
            
            # Obtem a média e o desvio padrão do caminho mínimo para todos os pares de pontos
            # através do método de Floyd-Warshall
            fw_aux = np.asarray(nx.floyd_warshall_numpy(self.G)).reshape(-1)
            self.floyd_warshall = np.array(np.delete(fw_aux, np.where(np.logical_or(fw_aux == 0, fw_aux == float('inf')))), dtype=int)   
            self.shortest_path_length_mu = self.floyd_warshall.mean()
            self.shortest_path_length_sigma = self.floyd_warshall.std()
        
        #Identificador único do grafo gerado
        self.dt_string = datetime.now().strftime("_%d-%m-%Y_%H-%M-%S")
        self.filename = 'img/Barabasi-Albert'+'_n='+str(self.n)+'_m='+str(self.m)+self.dt_string
        
    # Gera uma visualização do grafo em png
    def Visualize(self):
        
        net = Network()
        net.from_nx(self.G)
        net.show_buttons()
        net.show(self.filename+'.html')
        
        
    # Confecciona o histograma do grau
    def Plot_Degree_Histogram(self, xticks = True):
        bins = np.arange(0, self.degree.max()+2)-0.5        
        
        text = '\n'.join((
                r'$\mu=%.2f$' % (self.degree_mu, ),
                r'$\sigma=%.2f$' % (self.degree_sigma, )))
        
        
        fig = plt.figure(figsize=(8,4.5))
        ax = fig.add_axes([0, 0, 1, 1])
        
        plt.hist(self.degree, bins, label='Graph', color='#3971cc', edgecolor='#303030',
                 linewidth=1.5, density=True, zorder=1)
        
        
        plt.legend(loc='best', fontsize=18)
        if xticks:
            plt.xticks(range(self.degree.max()+1), fontsize = 14)
        else:
            plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.xlim(bins[0], bins[-1])
        plt.xlabel('Degree', fontsize=18)
        plt.ylabel('Relative Frequency', fontsize=18)
        plt.grid(axis='y')
        plt.text(0.86,0.7, text,
                 size = 20, verticalalignment='top', horizontalalignment='center',
                 color='#3971cc', bbox={'facecolor': 'white', 'alpha': 0.8,
                                      'pad': 0.5, 'boxstyle': 'round'},
                 transform = ax.transAxes)
        
        plt.savefig(self.filename+'_Degree_Histogram.png',
                    dpi=200, bbox_inches='tight')


    # Confecciona o histograma do coeficiente de aglomeração
    def Plot_Clustering_Coefficient_Histogram(self, number_of_bins = 11):
        
        bins = np.linspace(0, 1, num = number_of_bins)
            
        text = '\n'.join((
                r'$\mu=%.2f$' % (self.clustering_coefficient_mu, ),
                r'$\sigma=%.2f$' % (self.clustering_coefficient_sigma, )))
        
        
        fig = plt.figure(figsize=(8,4.5))
        ax = fig.add_axes([0, 0, 1, 1])
        
        weights = np.ones_like(self.clustering_coefficient)/float(len(self.clustering_coefficient))
        
        plt.hist(self.clustering_coefficient, bins, label='Graph', color='#3971cc', edgecolor='#303030',
                 linewidth=1.5, weights=weights, zorder=1)
        
        # plt.legend(loc='best', fontsize=18)
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        if 1/(number_of_bins-1) != 0.1:
            ax.xaxis.set_minor_locator(MultipleLocator(1/(number_of_bins-1)))
            
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.xlim(bins[0], bins[-1])
        plt.xlabel('Clustering Coefficient', fontsize=18)
        plt.ylabel('Relative Frequency', fontsize=18)
        plt.grid(axis='y')
        plt.text(0.88,0.92, text,
                 size = 20, verticalalignment='top', horizontalalignment='center',
                 color='#3971cc', bbox={'facecolor': 'white', 'alpha': 0.8,
                                      'pad': 0.5, 'boxstyle': 'round'},
                 transform = ax.transAxes)
        
        plt.savefig(self.filename+'_Clustering_Coefficient_Histogram.png',
                    dpi=200, bbox_inches='tight')


    # Confecciona o histograma do caminho mínimo
    def Plot_Shortest_Path_Length_Histogram(self, xticks=True):
    
        bins = np.arange(1, self.floyd_warshall.max()+2)-0.5
        
        text = '\n'.join((
                r'$\mu=%.2f$' % (self.shortest_path_length_mu, ),
                r'$\sigma=%.2f$' % (self.shortest_path_length_sigma, )))
        
        fig = plt.figure(figsize=(8,4.5))
        ax = fig.add_axes([0, 0, 1, 1])
        
        plt.hist(self.floyd_warshall, bins, label='Graph', color='#3971cc', edgecolor='#303030',
                 linewidth=1.5, density=True, zorder=1)
        
        # plt.legend(loc='best', fontsize=18)
        if xticks:
           plt.xticks(range(self.floyd_warshall.max()+1), fontsize = 14)
        else:
            plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.xlim(bins[0], bins[-1])
        plt.xlabel('Shortest Path Length', fontsize=18)
        plt.ylabel('Relative Frequency', fontsize=18)
        plt.grid(axis='y')
        plt.text(0.88,0.92, text,
                 size = 20, verticalalignment='top', horizontalalignment='center',
                 color='#3971cc', bbox={'facecolor': 'white', 'alpha': 0.8,
                                      'pad': 0.5, 'boxstyle': 'round'},
                 transform = ax.transAxes)
        
        plt.savefig(self.filename+'_Shortest_Path_Length_Histogram.png',
                    dpi=200, bbox_inches='tight')        


#%%

n_0 = 10
n = 500
average_degreee_0 = 1.5
m = 3

BA = BarabasiAlbert_network(n_0, average_degreee_0, n, m)

print(BA.degree_mu)

BA.Visualize()

BA.Plot_Degree_Histogram(xticks=False)
BA.Plot_Clustering_Coefficient_Histogram()
BA.Plot_Shortest_Path_Length_Histogram()

#%%
if 0:
    n_0 = 10
    n = 10000
    average_degreee_0 = 1.5
    m = 3

    bin_max = 500
    bins = np.arange(0, bin_max)-0.5  
    number_of_graphs = 10
    
    data = np.zeros(bin_max-1)
    #%%
    for i in range(number_of_graphs):
        BA = BarabasiAlbert_network(n_0, average_degreee_0, n, m, degree_only = True)
        
        b = plt.hist(BA.degree, bins)[:2]
        data = data + plt.hist(BA.degree, bins)[:2][0]
    #%%
    fig = plt.figure(figsize=(8,4.5))
    ax = fig.add_axes([0, 0, 1, 1])
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    def fit(x, a, gamma):
        return a - gamma*x
        
    data = data / np.sum(data)
    
    plt.scatter(bins[:-1]+0.5, data, marker='o', label='Average from graphs')
    
    # Fitting 1
    b = np.log(np.delete(data[3:], np.where(data[3:]==0)))
    c = np.log(np.delete(bins[:-1][3:]+0.5, np.where(data[3:]==0)))
    [a, gamma], pcov  = curve_fit(fit, c, b)

    x = np.logspace(np.log10(bins[:-1][3]+0.5), np.log10(bin_max), num=100)
    plt.plot(x, np.exp(a)*x**(-gamma), color=colors[2], linestyle = 'dotted',  lw=2,
         label='Fit 1: '+r'${:.2f}\cdot k^{{{:.2f}}}$'.format(np.exp(a), -1*gamma))

    # Fitting 2
    b = np.log(np.delete(data[3:70], np.where(data[3:70]==0)))
    c = np.log(np.delete(bins[:-1][3:70]+0.5, np.where(data[3:70]==0)))
    [a, gamma], pcov  = curve_fit(fit, c, b)

    x = np.logspace(np.log10(bins[:-1][3]+0.5), np.log10(bins[:-1][70]+0.5), num=100)
    plt.plot(x, np.exp(a)*x**(-gamma), color=colors[1], lw=2,
             label='Fit 2: '+r'${:.2f}\cdot k^{{{:.2f}}}$'.format(np.exp(a), -1*gamma))
    

    plt.legend(loc='best', fontsize = 18)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.ylim(np.min(np.delete(data, np.where(data == 0)))/2, 1)
    plt.xlim(0.9, bin_max)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.xlabel('Degree', fontsize=18)
    plt.ylabel('Relative Frequency', fontsize=18)
    
    plt.grid()
    
    dt_string = datetime.now().strftime("_%d-%m-%Y_%H-%M-%S")
    filename = 'img/Power_Law_'+'_NumberGraphs'+str(number_of_graphs)+'_n='+str(n)+'_m='+str(m)+dt_string
        

    plt.savefig(filename+'.png',
                dpi=200, bbox_inches='tight')