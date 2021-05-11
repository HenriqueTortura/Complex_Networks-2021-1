import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from datetime import datetime

import networkx as nx
from pyvis.network import Network


#%%
class ErdosRenyi_network(object):
    
    def __init__(self, n, average_degreee, input_method = False):
        
        # Define os principais parâmetros sem pedir informação ao usuário
        if not input_method:
            self.n = n
            self.average_degreee = average_degreee
            self.p = self.average_degreee / (self.n-1)
        
        # Define os principais parâmetros por um input do usuário
        while input_method:
            self.n = int(input('Number of nodes: '))
            self.average_degreee = np.abs(float(input('Expected mean degree: ')))
            if self.n > 1:
                self.p = self.average_degreee / (self.n-1)
                if self.p < 0 or self.p > 1:
                    print('\nThe probability of an edge to be included is not in the interval [0,1].')
                else:
                    input_method = False
            else:
                print('\nMust have at least two nodes.')
        
        # Cria matriz de adjacência e grafo
        self.adjacency_matrix = np.zeros((n,n))
        
        self.G = nx.Graph()
        self.G.add_nodes_from(range(1,n+1))
        
        # Laço que percorre apenas os elementos acima da diagonal principal,
        # de maneira que a matriz adjacência se torna uma matriz triangular superior
        for i in range(self.n):
            for j in range(i+1,self.n):
                if np.random.uniform(low=0, high=1) <= self.p:
                    self.adjacency_matrix[i,j] = 1
                    self.G.add_edge(i+1,j+1)
                    
        # Obtem a média e o desvio padrão do grau
        self.degree = np.array(sorted([d for n, d in self.G.degree()], reverse=True))
        self.degree_mu = self.degree.mean()
        self.degree_sigma = self.degree.std()
        
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
        
    # Gera uma visualização do grafo em html
    def Visualize(self):
        net = Network()
        net.from_nx(self.G)
        
        net.show('Erdos-Renyi_n='+str(self.n)+'_mu='+str(self.average_degreee)+'.html')
        
        
    # Confecciona o histograma do grau
    def Plot_Degree_Histogram(self):
        bins = np.arange(0, self.degree.max()+2)-0.5        
        
        text = '\n'.join((
                r'$\mu=%.2f$' % (self.degree_mu, ),
                r'$\sigma=%.2f$' % (self.degree_sigma, )))
        
        pmf = st.binom.pmf(np.arange(0, self.degree.max()+1), self.n, self.p) 
        
        fig = plt.figure(figsize=(8,4.5))
        ax = fig.add_axes([0, 0, 1, 1])
        
        plt.hist(self.degree, bins, label='Graph', color='#3971cc', edgecolor='#303030',
                 linewidth=1.5, density=True, zorder=1)
        
        plt.scatter(np.arange(0, self.degree.max()+1), pmf,
                    marker='_', s=800, linewidth=2.5, color='#ff7f0e', label='Binomial', zorder=2)
        
        plt.legend(loc='best', fontsize=18)
        plt.xticks(range(self.degree.max()+1), fontsize = 14)
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
        
        filename = 'img/Degree_n='+str(self.n)+'_mu='+str(self.average_degreee)
        plt.savefig(filename+self.dt_string+'.png',
                    dpi=200, bbox_inches='tight')


    # Confecciona o histograma do coeficiente de aglomeração
    def Plot_Clustering_Coefficient_Histogram(self):
        
        bins = np.linspace(0, 1, num=11)
            
        text = '\n'.join((
                r'$\mu=%.2f$' % (self.clustering_coefficient_mu, ),
                r'$\sigma=%.2f$' % (self.clustering_coefficient_sigma, )))
        
        
        fig = plt.figure(figsize=(8,4.5))
        ax = fig.add_axes([0, 0, 1, 1])
        
        weights = np.ones_like(self.clustering_coefficient)/float(len(self.clustering_coefficient))
        
        plt.hist(self.clustering_coefficient, bins, label='Graph', color='#3971cc', edgecolor='#303030',
                 linewidth=1.5, weights=weights, zorder=1)
        
        # plt.legend(loc='best', fontsize=18)
        plt.xticks(bins, fontsize = 14)
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
        
        filename = 'img/Clustering_Coefficient_n='+str(self.n)+'_mu='+str(self.average_degreee)
        plt.savefig(filename+self.dt_string+'.png',
                    dpi=200, bbox_inches='tight')


    # Confecciona o histograma do caminho mínimo
    def Plot_Shortest_Path_Length_Histogram(self):
    
        bins = np.arange(1, self.floyd_warshall.max()+2)-0.5
        
        text = '\n'.join((
                r'$\mu=%.2f$' % (self.shortest_path_length_mu, ),
                r'$\sigma=%.2f$' % (self.shortest_path_length_sigma, )))
        
        fig = plt.figure(figsize=(8,4.5))
        ax = fig.add_axes([0, 0, 1, 1])
        
        plt.hist(self.floyd_warshall, bins, label='Graph', color='#3971cc', edgecolor='#303030',
                 linewidth=1.5, density=True, zorder=1)
        
        # plt.legend(loc='best', fontsize=18)
        plt.xticks(range(self.floyd_warshall.max()+1), fontsize = 14)
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
        
        filename = 'img/Shortest_Path_Length_n='+str(self.n)+'_mu='+str(self.average_degreee)
        plt.savefig(filename+self.dt_string+'.png',
                    dpi=200, bbox_inches='tight')

#%% Rodando 

n = 100
average_degreee = 2

ER = ErdosRenyi_network(n, average_degreee)

# print(ER.p)
ER.Visualize()
ER.Plot_Degree_Histogram()
ER.Plot_Clustering_Coefficient_Histogram()
ER.Plot_Shortest_Path_Length_Histogram()



