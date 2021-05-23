import numpy as np

import scipy.stats as st
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import matplotlib.path as mpath
import matplotlib.patches as mpatches

from datetime import datetime

import networkx as nx
from pyvis.network import Network


#%% Classe da rede de Erdos-Renyi

class ER_network(object):
    
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


#%% Classe da rede de Voronoi

class Vor_network(object):
    
    def __init__(self, n, bounded = True, coord = None):
        
        # Define os principais parâmetros
        self.n = n
        self.bounded = bounded
              
        # Gera as coordenadas a partir de uma distribuição uniforme
        if coord is None:
            self.coord = np.random.rand(self.n,2)
        # ou recebe coordenadas preestabelecidas
        else:
            self.coord = coord
            
        # Realiza a tesselação de Voronoi. A variável connection guarda os pares
        # de índices dos pontos que geram céluas contíguas.
        
        # Caso limitado: tesselação é realizada com quatro cópias dos pontos
        # refletidas acima, abaixo e aos lados, de maneira que as células
        # acabem nas bordas de [0,1]x[0,1]
        if self.bounded:
            aux = np.concatenate((self.coord, self.coord, self.coord, self.coord))
            
            aux[0:self.n,1] = 2-1*self.coord[:,1] # up
            aux[self.n:2*self.n,1] = -1*self.coord[:,1] # down
            aux[2*self.n:3*self.n,0] = -1*self.coord[:,0] # left
            aux[3*self.n:4*self.n,0] = 2-1*self.coord[:,0] # right
            
            self.vor = Voronoi(np.concatenate((self.coord, aux)))
                
            self.connection = self.vor.ridge_points[np.where(np.logical_not(np.logical_or(self.vor.ridge_points[:,1]>self.n, self.vor.ridge_points[:,0]>self.n) )) ]
            
        else:
            self.vor = Voronoi(self.coord)
            self.connection = self.vor.ridge_points
        
        # Cria grafo e conexões de acordo com a contiguidade das células
        self.G = nx.Graph()
        self.G.add_nodes_from(range(0,self.n))
        
        for i in range(np.size(self.connection,0)):
            self.G.add_edge(self.connection[i,0], self.connection[i,1])
        
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
        self.filename = 'img/Voronoi_n='+str(self.n)+self.dt_string
        
    # Gera uma visualização do grafo em png
    def Visualize(self):
        
        def voronoi_finite_polygons_2d(vor, radius=None):
                
            new_regions = []
            new_vertices = vor.vertices.tolist()
        
            center = vor.points.mean(axis=0)
            if radius is None:
                radius = vor.points.ptp().max()
        
            # Construct a map containing all ridges for a given point
            all_ridges = {}
            for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
                all_ridges.setdefault(p1, []).append((p2, v1, v2))
                all_ridges.setdefault(p2, []).append((p1, v1, v2))
        
            # Reconstruct infinite regions
            for p1, region in enumerate(vor.point_region):
                vertices = vor.regions[region]
        
                if all(v >= 0 for v in vertices):
                    # finite region
                    new_regions.append(vertices)
                    continue
        
                # reconstruct a non-finite region
                ridges = all_ridges[p1]
                new_region = [v for v in vertices if v >= 0]
        
                for p2, v1, v2 in ridges:
                    if v2 < 0:
                        v1, v2 = v2, v1
                    if v1 >= 0:
                        # finite ridge: already in the region
                        continue
        
                    # Compute the missing endpoint of an infinite ridge
        
                    t = vor.points[p2] - vor.points[p1] # tangent
                    t /= np.linalg.norm(t)
                    n = np.array([-t[1], t[0]])  # normal
        
                    midpoint = vor.points[[p1, p2]].mean(axis=0)
                    direction = np.sign(np.dot(midpoint - center, n)) * n
                    far_point = vor.vertices[v2] + direction * radius
        
                    new_region.append(len(new_vertices))
                    new_vertices.append(far_point.tolist())
        
                # sort region counterclockwise
                vs = np.asarray([new_vertices[v] for v in new_region])
                c = vs.mean(axis=0)
                angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
                new_region = np.array(new_region)[np.argsort(angles)]
        
                # finish
                new_regions.append(new_region.tolist())
        
            return new_regions, np.asarray(new_vertices)
        
        
        regions, vertices = voronoi_finite_polygons_2d(self.vor)
        
        plt.figure(figsize=(16,16))
        
        # colorize
        for region in regions:
            polygon = vertices[region]
            plt.fill(*zip(*polygon), alpha=0.4, edgecolor='black', linestyle='dotted')
            
        plt.scatter(self.coord[:,0], self.coord[:,1], s=20, color='black')
        for i in range(np.size(self.connection,0)):
            plt.plot(self.coord[self.connection[i]][:,0],
                     self.coord[self.connection[i]][:,1],
                     '-', color='black')
        
        if self.bounded:
            plt.xlim(0, 1)
            plt.ylim(0, 1)
        else:
            plt.xlim(-0.5, 1.5)
            plt.ylim(-0.5, 1.5)
            
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        
        plt.savefig(self.filename + '.png',
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
        

#%% Classe da rede geométrica

class Geo_network(object):
    
    def __init__(self, n, r, coord = None):
        
        # Define os principais parâmetros
        self.n = n
        self.r = r
              
        # Gera as coordenadas a partir de uma distribuição uniforme
        if coord is None:
            self.coord = np.random.rand(self.n,2)
        # ou recebe coordenadas preestabelecidas
        else:
            self.coord = coord
            
        
        # Calcula as distâncias entre cada par de pontos
        self.dist = cdist(self.coord, self.coord)
        # Obtem os índices dos nós que possuem distância não nula (mesmo nó) e menor que o critério
        self.connection = np.unique(np.sort(np.transpose(np.array(np.where(np.logical_and(self.dist > 0, self.dist < 2*r))))), axis=0)

        # Cria grafo e conexões
        self.G = nx.Graph()
        self.G.add_nodes_from(range(0,self.n))
        
        self.G.add_edges_from(self.connection)
                    
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
        self.filename = 'img/Geometric'+'_n='+str(self.n)+'_r='+str(self.r)+self.dt_string
        
        
    # Gera uma visualização do grafo em png
    def Visualize(self):
                
        fig, ax = plt.subplots(figsize=(16,16))
        
        for i in range(self.n):
            circle = plt.Circle((self.coord[i,0], self.coord[i,1]), radius=self.r,
                                 alpha=0.1, edgecolor='black', linestyle='dashed',
                                 linewidth = 1.5, zorder = 2)
            ax.add_patch(circle)
        
        plt.scatter(self.coord[:,0], self.coord[:,1], s=20, color='black', zorder = 3)
        for i in range(np.size(self.connection,0)):
            plt.plot(self.coord[self.connection[i]][:,0],
                     self.coord[self.connection[i]][:,1],
                     '-', color='#c94242', zorder=1)
        
        
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
        
        p = self.degree_mu / (self.n - 1)
        # pmf = st.poisson.pmf(np.arange(0, self.degree.max()+1), alpha)
        pmf = st.binom.pmf(np.arange(0, self.degree.max()+1), self.n, p) 
        
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
        

#%% Classe da rede de Waxman

class Wax_network(object):
    
    def __init__(self, n, alpha, beta, coord = None):
        
        # Define os principais parâmetros
        self.n = n
        self.alpha = alpha
        self.beta = beta
                     
        # Gera as coordenadas a partir de uma distribuição uniforme
        if coord is None:
            self.coord = np.random.rand(self.n,2)
        # ou recebe coordenadas preestabelecidas
        else:
            self.coord = coord
        
        # Calcula as distâncias entre cada par de pontos
        self.dist = cdist(self.coord, self.coord)
        
        # Calcula a probabilidade de decaimento exponencial
        self.prob = self.beta*np.exp(-self.alpha*self.dist)
        
        # Realiza conexões baseadas na probabilidade
        self.sorted = np.random.rand(self.n,self.n)
        self.connection = np.transpose(np.array(np.where(np.logical_and(self.prob>self.sorted, self.prob!=self.beta))))
        # Elimina autoconexões e duplicadas (``diagonal principal'' e ``triangular inferior'')
        self.connection = np.delete(self.connection, np.where(self.connection[:,0]>self.connection[:,1]), axis=0)
        
        # Cria grafo e conexões
        self.G = nx.Graph()
        self.G.add_nodes_from(range(0,self.n))
        
        self.G.add_edges_from(self.connection)
        
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
        self.filename = 'img/Waxman'+'_n='+str(self.n)+'_alpha='+str(self.alpha)+'_beta='+str(self.beta)+self.dt_string
        
        
    # Gera uma visualização do grafo em png
    def Visualize(self):

        fig, ax = plt.subplots(figsize=(16,16))
                
        plt.scatter(self.coord[:,0], self.coord[:,1], s=20, color='black', zorder = 3)
        for i in range(np.size(self.connection,0)):
            plt.plot(self.coord[self.connection[i]][:,0],
                     self.coord[self.connection[i]][:,1],
                     '-', color='#c94242', zorder=1)
        
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
        

#%% Classe da rede de Watts-Strogatz

class WS_network(object):
    
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


#%% Classe da rede de Barabasi-Albert

class BA_network(object):
    
    def __init__(self, n_0, average_degreee_0, n, m, degree_only=False):
        
        # Define os principais parâmetros
        self.n_0 = n_0
        self.average_degreee_0 = average_degreee_0
        self.n = n
        self.m = m
        
        # Cria grafo inicial como uma rede de Erdos-Renyi
        self.G = ER_network(self.n_0, self.average_degreee_0).G
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
