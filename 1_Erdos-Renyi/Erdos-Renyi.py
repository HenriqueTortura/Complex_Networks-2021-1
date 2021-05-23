import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from datetime import datetime

import networkx as nx
from pyvis.network import Network


#%% Classe da rede de Erdos-Renyi

class ErdosRenyi_network(object):
    
    def __init__(self, n, average_degreee):
        
        # Recebe os parâmetros principais
        self.n = n
        self.average_degreee = average_degreee
        
        # Calcula a probabilidade de conexão
        self.p = self.average_degreee / (self.n-1)
        
        # Inicializa o grafo
        self.G = nx.Graph()
        self.G.add_nodes_from(range(0,self.n))
        
        # Gera a lista de conexões a serem realizadas
        sorting = np.random.rand(self.n,self.n)
        self.connection = np.transpose(np.array(np.where(self.p>sorting)))
        # Elimina triangular inferior
        self.connection = np.delete(self.connection, np.where(self.connection[:,0]>=self.connection[:,1]), axis=0)
       
        # Realiza as conexões no grafo
        for i in range(np.size(self.connection, axis=0)):
            self.G.add_edge(int(self.connection[i,0]), int(self.connection[i,1]))
            
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
        self.filename = 'img/Erdos-Renyi'+'_n='+str(self.n)+'_mu='+str(self.average_degreee)+self.dt_string
        
    # Gera uma visualização do grafo em html
    def Visualize(self):
        
        net = Network()
        net.from_nx(self.G)
        net.show_buttons()
        net.show(self.filename+'.html')
        
        
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
        
        plt.savefig(self.filename+'_Shortest_Path_Length_Histogram.png',
                    dpi=200, bbox_inches='tight')


#%%

n = 500
average_degree = 5
tolerance = 0.02
limit = 100

i = 1

Graph = ErdosRenyi_network(n, average_degree)

while (np.abs(Graph.degree_mu-average_degree)>average_degree*tolerance) and i <= limit:
    Graph = ErdosRenyi_network(n, average_degree)
    i += 1

#%%
if (np.abs(Graph.degree_mu-average_degree)<average_degree*tolerance):

    Graph.Visualize()
    
    Graph.Plot_Degree_Histogram()
    Graph.Plot_Clustering_Coefficient_Histogram(number_of_bins=31)
    Graph.Plot_Shortest_Path_Length_Histogram()



