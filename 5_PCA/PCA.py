import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn import preprocessing

import matplotlib.pyplot as plt

from datetime import datetime

import Networks


#%%

n = 500
average_degree = 6

r = 0.03

beta = 0.95
alpha = np.sqrt(2*np.pi*(n-1)*beta/average_degree)-1.5

p = 0.01

n_0 = 10
average_degree_0 = 1.5
m = 3

number_of_graphs = 50
tolerance = 0.05
limit = 100

dt_string = datetime.now().strftime("_%d-%m-%Y_%H-%M-%S")
filename = 'Data--n_of_graphs='+ str(number_of_graphs) + dt_string + '.dat'

head = ['Graph', 'Mean Degree', 'Degree STD', 'Mean CC',
           'CC STD', 'Mean SPL', 'SPL STD']

tags = {
        'ER'    : [n, average_degree],
        'Vor'   : [n],
        'Geo'   : [n, r],
        'Wax'   : [n, alpha, beta],
        'WS'    : [n, average_degree, p],
        'BA'    : [n_0, average_degree_0, n, m]
       }

with open(filename, 'w+') as datafile:
    datafile.write('#{:<5s}\t{:<10s}\t{:<10s}\t{:<10s}\t{:<10s}\t{:<10s}\t{:<10s}\n'.format(*head))
datafile.closed

#%%
for tag in tags.keys():
    
    method_to_call = getattr(Networks, tag+'_network')

    for i in range(number_of_graphs):
        
        print(tag+'_'+str(i+1))
        Graph = method_to_call(*tags[tag])

        j = 1
        
        while (np.abs(Graph.degree_mu-average_degree)>average_degree*tolerance) and j <= limit:
            Graph = method_to_call(*tags[tag])
            j += 1
            
        if j == limit+1:
            print('Warning: average degree of '+tag+'_'+str(i)+' does not meet required tolerance')
        
        values = ['{:<7s}'.format(tag+'_'+str(i+1)),
                  '{:.5f}'.format(Graph.degree_mu),
                  '{:.5f}'.format(Graph.degree_sigma),
                  '{:.5f}'.format(Graph.clustering_coefficient_mu),
                  '{:.5f}'.format(Graph.clustering_coefficient_sigma), 
                  '{:.5f}'.format(Graph.shortest_path_length_mu),
                  '{:.5f}'.format(Graph.shortest_path_length_sigma)]
        
        with open(filename, 'a+') as datafile:
            datafile.write('{:<5s}\t{:<10s}\t{:<10s}\t{:<10s}\t{:<10s}\t{:<10s}\t{:<10s}\n'.format(*values))
        datafile.closed

#%%
# index = [3,5]
index = range(2,7)

print(filename)
data = pd.read_csv(filename, delimiter='\t', index_col=0, usecols=(0,*index))

 
print(data.head())
print(data.shape)

#%%
scaled_data = preprocessing.scale(data)
 
pca = PCA() # create a PCA object
pca.fit(scaled_data) # do the math
pca_data = pca.transform(scaled_data) # get PCA coordinates for scaled_data

#The following code constructs the Scree plot
per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
 
plt.figure(figsize=(16,9))
plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)

plt.grid(axis='y')
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.ylabel('Percentage of Explained Variance', fontsize=18)
plt.xlabel('Principal Component', fontsize=18)
plt.savefig(dt_string + '_ScreePlot.png',
                    dpi=200, bbox_inches='tight')


#%%
#the following code makes a fancy looking plot using PC1 and PC2
plt.figure(figsize=(16,9))

pca_df = pd.DataFrame(pca_data, index = np.genfromtxt(filename, usecols = 0, dtype=str),
                      columns=labels)

for i in range(np.size(list(tags.keys()))):
    
    plt.annotate(list(tags.keys())[i],
                 (pca_df.PC1[i*number_of_graphs], pca_df.PC2[i*number_of_graphs]),
                 fontsize = 18, zorder = 3)
    plt.scatter(pca_df.PC1[i*number_of_graphs:(i+1)*number_of_graphs],
                pca_df.PC2[i*number_of_graphs:(i+1)*number_of_graphs],
                label = list(tags.keys())[i], zorder = 2)

plt.grid()
plt.legend(loc='best', fontsize=18)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.xlabel('PC1 - {0}%'.format(per_var[0]), fontsize=18)
plt.ylabel('PC2 - {0}%'.format(per_var[1]), fontsize=18)
 
# for sample in pca_df.index:
#     plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))
 
plt.savefig(dt_string + '_PCA.png',
                    dpi=200, bbox_inches='tight')

#%% loading scores

loading_scores = pd.Series(pca.components_[0], index=[head[i] for i in index])
## now sort the loading scores based on their magnitude
sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
 
# get the names of the top 10 genes
top_10_genes = sorted_loading_scores.index.values
 
## print the gene names and their scores (and +/- sign)
print(loading_scores[top_10_genes]**2)
print(np.sum(loading_scores[top_10_genes]**2))