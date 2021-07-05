import numpy as np
from scipy.optimize import curve_fit

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import os

#%% Função para criar o sinal
def Create_Signal(m, n, sto_matrix, signs, random = 0):
    
    # Probabilidades aleatórias
    if random:
        sto_matrix = np.random.rand(n,n)
        sto_matrix = sto_matrix / sto_matrix.sum(axis=0) #normalização
    

    # Matriz para verficar resultado
    empirical_matrix = np.zeros((n,n))

    # Criando o sinal
    signal = np.zeros(m, dtype=int)
    for i in range(m-1):
        signal[i+1] = np.random.choice(signs, 1, p=sto_matrix[:,signal[i]])
        empirical_matrix[signal[i+1],signal[i]] += 1

    # Normalização 
    empirical_matrix = empirical_matrix / empirical_matrix.sum(axis=0)
    empirical_matrix = np.nan_to_num(empirical_matrix)
    
    return signal, empirical_matrix

#%% Função para plotar sinal 1D
def Plot_1D(n_signs, signal, label='_', colorbar = True):
    def colorbar_index(ncolors, cmap):
        mappable = cm.ScalarMappable(cmap=cmap)
        mappable.set_array([])
        mappable.set_clim(-0.5, ncolors+0.5)
        colorbar = plt.colorbar(mappable, orientation="horizontal")
        colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
        colorbar.set_ticklabels(range(ncolors))
    
    colors = ['navy', 'cornflowerblue', 'limegreen', 'gold', 'sandybrown', 'crimson']
    if n_signs == 2:
        colors = ['navy', 'gold']
    cmap = mpl.colors.ListedColormap(colors[:len(np.unique(signal))])
    
    fig, ax = plt.subplots(figsize=(10,2))
    plt.imshow(signal[np.newaxis,:], aspect='auto', interpolation='nearest',
                origin='lower', cmap=cmap)
    ax.set_xticks(np.arange(0, 200)-0.5) 
    ax.set_yticks([]) 
    ax.grid(axis='x', linestyle='-', color='k', linewidth=0.5)
    ax.tick_params(axis='x', colors=(0,0,0,0))
    ax.tick_params(axis='y', colors=(0,0,0,0))
    
    if colorbar:
        colorbar_index(ncolors=n_signs, cmap=mpl.colors.ListedColormap(colors[:n_signs]))    
    
    fname = 'Automaton_' + label + '_Cmap_'
    i = 0
    while os.path.exists(fname + str(i) + '.png'):
        i += 1
    plt.savefig(fname + str(i) + '.png',
                dpi=200, bbox_inches='tight')

#%%
def Plot_2D(n_signs, m, signal, signal2, label='_'):
    
    x = range(m)
    y = range(m)
    xx, yy = np.meshgrid(x, y, sparse=True)
    z = signal[xx] + signal2[yy]
    z[np.where(z==2)] = 1
    
    colors = ['navy', 'cornflowerblue', 'limegreen', 'gold', 'sandybrown', 'crimson']
    if n_signs == 2:
        colors = ['navy', 'gold']
    cmap = mpl.colors.ListedColormap(colors[:len(np.unique(signal))])
    
    fig, ax = plt.subplots(figsize=(10,10))
    plt.imshow(z, aspect='auto', interpolation='nearest',
               origin='lower', cmap=cmap)
    plt.axis('off')
    
    fname = 'Automaton_' + label + '_Cmap_'
    i = 0
    while os.path.exists(fname + str(i) + '.png'):
        i += 1
    plt.savefig(fname + str(i) + '_2D_signal.png',
                dpi=200, bbox_inches='tight')

#%%
def Plot_Circles(r, n_signs, m, signal, signal2, label='_'):
    
    x = range(m)
    y = range(m)
    xx, yy = np.meshgrid(x, y, sparse=True)
    z = signal[xx] + signal2[yy]
    
    circles = np.zeros( (np.size(z, axis=0), np.size(z, axis=0)) )
    points = np.zeros( (2*(2*r+1) * len(np.where(z==2)[0]), 2), dtype=int )
    
    j = 0
    
    for i in range(len(np.where(z==2)[0])):
        
        # Procura por intersecções
        center = np.where(z==2)[0][i], np.where(z==2)[1][i]

        # Lista os pontos da circunferência        
        for x_r in range(center[0]-r,center[0]+r+1):
            y_r1 = int(round(np.sqrt(r**2 - (x_r-center[0])**2) + center[1]))
            y_r2 = int(round(-1*np.sqrt(r**2 - (x_r-center[0])**2) + center[1]))
            points[j,] = [x_r, y_r1]
            points[j+1,] = [x_r, y_r2]
            j = j + 2
            
    # Restringe os pontos dentro da malha
    points = np.delete(points, np.where(np.logical_or(points[:,1]<0, points[:,0]<0)), axis=0)
    points = np.delete(points, np.where(np.logical_or(points[:,1]>=200, points[:,0]>=200)), axis=0)
    
    # Gera a malha
    for (i,j) in points:
        circles[i,j] = 1
        
    colors = ['navy', 'cornflowerblue', 'limegreen', 'gold', 'sandybrown', 'crimson']
    if n_signs == 2:
        colors = ['navy', 'gold']
    cmap = mpl.colors.ListedColormap(colors[:len(np.unique(signal))])
    
    fig, ax = plt.subplots(figsize=(10,10))
    plt.imshow(circles, aspect='auto', interpolation='nearest',
               origin='lower', cmap=cmap)
    plt.axis('off')
    
    fname = 'Automaton_' + label + '_Cmap_'
    i = 0
    while os.path.exists(fname + str(i) + '.png'):
        i += 1
    plt.savefig(fname + str(i) + '_2D_Circles.png',
                dpi=200, bbox_inches='tight')

#%%
def Plot_Density_of_1s(automata, n_m, runs, bins = 10, verbose=True):

    def gauss(x, sigma, mu):
        return (1/(np.sqrt(2*np.pi)*sigma))*np.exp(-(x-mu)**2/(2*sigma**2))
    
    f_a = np.zeros(runs)
    coeffs = np.zeros((len(n_m), 2, len(automata.keys())))
    
    
    for k in range(len(automata.keys())):
    # for automaton in ['a']:
        print(list(automata.keys())[k])
        for j in range(len(n_m)):
        # for m in [1000]
            
            for i in range(runs):
                if verbose:
                    print(str(i+1)+'/'+str(runs)+' '+str(j+1)+'/'+str(len(n_m))+' '+str(k+1)+'/'+str(len(automata.keys())))
                signal, empirical_matrix = Create_Signal(n_m[j], automata[list(automata.keys())[k]][0],
                                                         automata[list(automata.keys())[k]][1],
                                                         automata[list(automata.keys())[k]][2])
                f_a[i] = np.sum(signal)/n_m[j]
            
            y, x = np.histogram(f_a, bins = bins)
            x_data = np.delete((x + np.roll(x, 1))/2, 0)
            coeffs[j,:,k], var_matrix = curve_fit(gauss, x_data, y)
            print('Done')
        
    
    colors = ['navy', 'limegreen', 'crimson']
    linetypes = [ (0, (5, 10)), 'dashed', (0, (5, 1)), 'solid']
    
    fig = plt.figure(figsize=(8,4.5))
    ax = fig.add_axes([0, 0, 1, 1])
    
    x = np.linspace(0, 1, num=1000000)
    
    for k in range(len(automata.keys())):
    # for automaton in ['b']:
        for j in range(len(n_m)):
            if j == len(n_m)-1: 
                plt.plot(x, gauss(x, *coeffs[j,:,k]), linewidth=1.5,
                         label = list(automata.keys())[k], color = colors[k],
                         linestyle = linetypes[j])
            else:
                plt.plot(x, gauss(x, *coeffs[j,:,k]), linewidth=1.5,
                         color = colors[k], linestyle = linetypes[j])
        
    plt.legend(loc='best', fontsize=18)
    plt.xticks(np.linspace(0,1, 11), fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.xlabel('Frequency of 1\'s', fontsize=18)
    plt.ylabel('Density', fontsize=18)
    plt.grid()
    
    label = ''
    for i in list(automata.keys()):
        label = label+'_'+i
    plt.savefig('Density_of_1_'+label+'.png',
                dpi=200, bbox_inches='tight')
    
#%%
# Tamanho da sequência/sinal
m = 200

# Lista com autômatos do CDT-22
tags = {
        'a'   : [2, np.array([[0.9, 0.9], [0.1, 0.1]]), [0, 1]],
        'b'   : [2, np.array([[0.2, 0.2], [0.8, 0.8]]), [0, 1]],
        'c'   : [2, np.array([[0.5, 0.5], [0.5, 0.5]]), [0, 1]],
        'd'   : [6, np.array([[0.9, 0.882, 0, 0, 0, 0.01], [0.1, 0.098, 0, 0, 0, 0],
                              [0, 0.02, 0.2, 0.194, 0, 0], [0, 0, 0.8, 0.776, 0, 0],
                              [0, 0, 0, 0.03, 0.5, 0.495], [0, 0, 0, 0, 0.5, 0.495]]),
                [0, 1, 2, 3, 4, 5]],
        'e'   : [6, np.array([[0.9, 0.882, 0, 0, 0, 0.01], [0.1, 0.098, 0, 0, 0, 0],
                              [0, 0.02, 0.2, 0.194, 0, 0], [0, 0, 0.8, 0.776, 0, 0],
                              [0, 0, 0, 0.03, 0.5, 0.495], [0, 0, 0, 0, 0.5, 0.495]]),
                [0, 1, 0, 1, 0, 1]],
        '0.55/0.45'   : [2, np.array([[0.55, 0.55], [0.45, 0.45]]), [0, 1]],
        '0.4/0.6'     : [2, np.array([[0.4, 0.4], [0.6, 0.6]]), [0, 1]]
        }

# Selecionar o autômato (de acordo com a figura 6 do CDT-22)
automaton = 'a'

signal, empirical_matrix = Create_Signal(m, tags[automaton][0], tags[automaton][1],
                                         tags[automaton][2])


Plot_1D(len(np.unique(tags[automaton][2])), signal, label=automaton, colorbar = True)

Plot_2D(len(np.unique(tags[automaton][2])), m, signal, signal, label=automaton)

#%%
automaton = 'a'
signal2, empirical_matrix = Create_Signal(m, tags[automaton][0], tags[automaton][1],
                                         tags[automaton][2])

Plot_2D(len(np.unique(tags[automaton][2])), m, signal, signal2, label=automaton)

#%%
Plot_Circles(15, len(np.unique(tags[automaton][2])), m, signal, signal, label='a')


#%%

# n_m = [1000, 2000, 5000, 10000]
n_m = [200, 500, 750, 1000]
runs = 200
bins = 10

automata = {
            # 'a'   : [2, np.array([[0.9, 0.9], [0.1, 0.1]]), [0, 1]],
            # 'b'   : [2, np.array([[0.2, 0.2], [0.8, 0.8]]), [0, 1]],
            'c'   : [2, np.array([[0.5, 0.5], [0.5, 0.5]]), [0, 1]],
            '55% / 45%'   : [2, np.array([[0.55, 0.55], [0.45, 0.45]]), [0, 1]],
            '40% / 60%'     : [2, np.array([[0.4, 0.4], [0.6, 0.6]]), [0, 1]]
           }



Plot_Density_of_1s(automata, n_m, runs)
