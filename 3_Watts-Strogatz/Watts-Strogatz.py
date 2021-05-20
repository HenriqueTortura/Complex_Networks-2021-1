import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.path as mpath
import matplotlib.patches as mpatches

from datetime import datetime

import networkx as nx
from pyvis.network import Network


#%% Classe da rede de Watts-Strogatz

class WattsStrogatz_network(object):
    
    def __init__(self, n, average_degreee, p, Lattice_2d = None):
        
        # Define os principais parâmetros
        self.Lattice_2d = Lattice_2d
        self.p = p
        
        # Caso da rede em forma de anel, o valor de n é usado e o grau médio inicial pode tomar qualquer valor (par)
        if self.Lattice_2d is None: 
            self.n = n
            self.average_degreee = average_degreee
            self.connection = np.zeros((int(self.n*self.average_degreee/2), 2), dtype = int)
            
            # Obtém os índices das conexões (vizinho da esquerda para vizinho da direita)
            for j in range(int(self.average_degreee/2)):
                for i in range(self.n):
                    self.connection[int(self.average_degreee*i/2+j), :] = [i, (i+j+1) % self.n]
        
        # Caso da rede como uma grade bidimensional, o valor n não é usado e o número de nós e dado pelo produto de Lattice_2d[0] com Lattice_2d[1], o grau médio é fixado em 4             
        else:
            self.n = self.Lattice_2d[0]*self.Lattice_2d[1]
            self.average_degreee = 4
            self.coord = np.zeros((self.n, 2))
            
            x = np.linspace(0.1, 0.9, self.Lattice_2d[0])
            y = np.linspace(0.9, 0.1, self.Lattice_2d[1])
            
            # Atribui posições no plano para os nós
            for j in range(self.Lattice_2d[1]):
                for i in range(self.Lattice_2d[0]):
                    self.coord[j*self.Lattice_2d[0]+i, :] = [x[i], y[j]]
                    
            self.connection = np.zeros((int(self.n*2), 2), dtype = int)
            
            # Obtém os índices das conexões
            for j in range(self.Lattice_2d[1]):
                for i in range(self.Lattice_2d[0]):
                    # Conexões horizontais
                    self.connection[j*self.Lattice_2d[0] + i, :] = j*self.Lattice_2d[0] + np.array([i, (i+1) % self.Lattice_2d[0]], dtype=int)
                    # Conexões verticais
                    self.connection[self.n + j*self.Lattice_2d[0] + i, :] = np.array([j*self.Lattice_2d[0] + i, (j*self.Lattice_2d[0] + i+self.Lattice_2d[0]) % self.n], dtype=int)

        # Realiza as reconexões
        if self.p > 0:
            for j in range(int(self.average_degreee/2)): # Percorre os vizinhos
                    for i in range(self.n): # Percorre os nós
                    
                        if np.random.uniform(low=0, high=1) <= self.p:
                            
                            # Apaga conexão
                            self.connection[int(self.average_degreee*i/2+j), :] = [i,i]
                            # Lista os nós que já estão conectados ao nó de interesse
                            forbidden = np.unique(np.concatenate((self.connection[np.where(self.connection[:,0]==i),1],
                                                                  self.connection[np.where(self.connection[:,1]==i),0]), axis = 1))
                            #Lista os nós que ainda não estão conectados ao nó de interesse
                            allowed = np.delete(np.arange(0,self.n), forbidden)
                            
                            # Reconecta
                            new_connection = np.random.randint(0,np.size(allowed))
                            self.connection[int(self.average_degreee*i/2+j), 1] = allowed[new_connection]
        
        # Cria grafo e conexões de acordo com a contiguidade das células
        self.G = nx.Graph()
        self.G.add_nodes_from(range(0,self.n))
        
        for i in range(np.size(self.connection, axis=0)):
            self.G.add_edge(int(self.connection[i-1,0]), int(self.connection[i-1,1]))
        
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
        self.filename = 'img/Watts-Strogatz_n='+str(self.n)+'_k='+str(self.average_degreee)+'_p='+str(self.p)+self.dt_string
        
    # Gera uma visualização do grafo em png
    def Visualize(self, PlaneView = False):
        
        if (self.Lattice_2d is None) or (not PlaneView): # Caso de rede circular
            net = Network()
            net.from_nx(self.G)
            net.show_buttons()
            net.show('img/Watts-Strogatz_n='+str(self.n)+'_mu='+str(self.average_degreee)+'_p='+str(self.p)+self.dt_string+'.html')
        
        elif PlaneView:
            fig, ax = plt.subplots(figsize=(16,16))
                    
            plt.scatter(self.coord[:,0], self.coord[:,1], s=20, color='black', zorder = 3)
            
            dx = np.minimum(self.coord[1,0] - self.coord[0,0], 0.05)
            dy = np.minimum(self.coord[0,1] - self.coord[self.Lattice_2d[0],1], 0.05)
            
            for i in range(np.size(self.connection,0)):
                if (i % self.Lattice_2d[0] == self.Lattice_2d[0]-1) and i < self.n:
                    Path = mpath.Path
                    pp1 = mpatches.PathPatch(
                          Path([self.coord[self.connection[i,0]], (0.5, self.coord[self.connection[i,1]][1]-dy), self.coord[self.connection[i,1]]],
                                [Path.MOVETO, Path.CURVE3, Path.CURVE3]),
                          fc="none", color='#2400b5', linewidth=1.5, transform=ax.transData)
                    
                    ax.add_patch(pp1)
                elif i>= np.size(self.connection, axis=0) - self.Lattice_2d[0]:
                    Path = mpath.Path
                    pp1 = mpatches.PathPatch(
                          Path([self.coord[self.connection[i,0]], (self.coord[self.connection[i,1]][0]-dx, 0.5),self.coord[self.connection[i,1]]],
                                [Path.MOVETO, Path.CURVE3, Path.CURVE3]),
                          fc="none", color='#2400b5', linewidth=1.5, transform=ax.transData)
                    
                    ax.add_patch(pp1)
                else:
                    plt.plot(self.coord[self.connection[i]][:,0],
                              self.coord[self.connection[i]][:,1],
                              '-', color='#d10000', zorder=1)
                        
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.xticks(fontsize = 14)
            plt.yticks(fontsize = 14)
            plt.grid()
            
            
            plt.savefig(self.filename+'.png',
                        dpi=200, bbox_inches='tight')
            
        
    # Confecciona o histograma do grau
    def Plot_Degree_Histogram(self):
        bins = np.arange(0, self.degree.max()+2)-0.5        
        
        text = '\n'.join((
                r'$\mu=%.2f$' % (self.degree_mu, ),
                r'$\sigma=%.2f$' % (self.degree_sigma, )))
        
        
        fig = plt.figure(figsize=(8,4.5))
        ax = fig.add_axes([0, 0, 1, 1])
        
        plt.hist(self.degree, bins, label='Graph', color='#3971cc', edgecolor='#303030',
                 linewidth=1.5, density=True, zorder=1)
        
        
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


#%% Watts-Strogatz

n = 500
average_degree = 10
p = 0.1
# Lattice_2d = [10,10]

limit = 1000
i = 1

WS = WattsStrogatz_network(n, average_degree, p)
# WS = WattsStrogatz_network(n, average_degreee, p, Lattice_2d = Lattice_2d)

while (WS.degree_sigma<=0.1) and i <= limit:
    print(i)
    WS = WattsStrogatz_network(n, average_degree, p)
    i += 1

#%%

WS.Visualize()
a = WS.p
b = WS.degree

# %%
WS.Plot_Degree_Histogram()
WS.Plot_Clustering_Coefficient_Histogram(number_of_bins=101)
WS.Plot_Shortest_Path_Length_Histogram()


#%% Varrendo valores de probabilidade
if 0:
    n = 500
    average_degree = 4
    
    number_of_probs = 10
    number_of_realizations = 100
    
    p = np.logspace(-4, -3, num = number_of_probs)
    # p = np.linspace(0.0001, 0.001, num = number_of_probs)
    
    clustering_coefficient = np.zeros((number_of_probs, 2))
    shortest_path_length = np.zeros((number_of_probs, 2))
    
    aux = np.zeros((number_of_realizations, 2))
    
    
    for i in range(number_of_probs):
        for j in range(number_of_realizations):
            
            print(str(i*number_of_realizations+j+1)+'/'+str(number_of_probs*number_of_realizations))
            WS = WattsStrogatz_network(n, average_degree, p[i])
            
            aux[j, 0] = WS.clustering_coefficient_mu
            aux[j, 1] = WS.shortest_path_length_mu
            
        clustering_coefficient[i,:] = [aux[:, 0].mean(), aux[:, 0].std()] 
        shortest_path_length[i,:]  = [aux[:, 1].mean(), aux[:, 1].std()] 
        
    #%% Ponto crítico
    
    dp = np.delete(np.roll(p, -1)-p, number_of_probs-1)
    
    d_cc = np.delete(np.roll(clustering_coefficient[:,0], -1) - clustering_coefficient[:,0], number_of_probs-1)
    derivative_cc = d_cc/dp
    crit_p_cc = np.where(np.abs(derivative_cc) == np.max(np.abs(derivative_cc)))
    
    d_spl = np.delete(np.roll(shortest_path_length[:,0], -1) - shortest_path_length[:,0], number_of_probs-1)
    derivative_spl = d_spl/dp
    crit_p_spl = np.where(np.abs(derivative_spl) == np.max(np.abs(derivative_spl)))
    
    #%% Plotando
    
    fig = plt.figure(figsize=(8,4.5))
    ax = fig.add_axes([0, 0, 1, 1])
            
    # Propagação de incerteza
    yerr_cc = ( clustering_coefficient[0,0]*clustering_coefficient[:,1] + clustering_coefficient[0,1]*clustering_coefficient[:,0] ) / (clustering_coefficient[0,0]**2)
    yerr_spl = ( shortest_path_length[0,0]*shortest_path_length[:,1] + shortest_path_length[0,1]*shortest_path_length[:,0] ) / (shortest_path_length[0,0]**2)
    
    plt.errorbar(p, clustering_coefficient[:,0]/clustering_coefficient[0,0],
                  yerr = yerr_cc, label=r'$C(p)/C(0)$, $p_C = {:.2e}$'.format(p[crit_p_cc][0]),
                  color='#1f77b4')
    
    plt.errorbar(p, shortest_path_length[:,0]/shortest_path_length[0,0],
                  yerr = yerr_spl, label=r'$L(p)/L(0)$, $p_C = {:.2e}$'.format(p[crit_p_spl][0]),
                  color='#ff7f0e')
    # plt.plot(p, shortest_path_length[:,0]/shortest_path_length[0,0],
    #          # yerr = yerr_spl,
    #          label=r'$L(p)/L(0)$, $p_C = {:.3e}$'.format(p[crit_p_spl][0]),
    #          color='#ff7f0e')
    
    ax.set_xscale('log')
    
    plt.legend(loc='best', fontsize=18)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.xlabel('p', fontsize=18)
    plt.grid()
    
    dt_string = datetime.now().strftime("_%d-%m-%Y_%H-%M-%S")
    filename = 'img/p_sweep_Watts-Strogatz_n='+str(n)+'_k='+str(average_degree)+'_probs='+str(number_of_probs)+dt_string
    plt.savefig(filename+'.png',
                        dpi=200, bbox_inches='tight')
