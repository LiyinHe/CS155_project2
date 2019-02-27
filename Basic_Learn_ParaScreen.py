import numpy as np
import os
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from prob2utils_YM import train_model, get_err
from multiprocessing import Pool

ratings = pd.read_csv(os.path.join('data', 'data.txt'), sep='\t', header=None, names=['User ID', 'Movie ID', 'Rating'])

# Read in information on movies
names = ['Movie ID', 'Movie Title', 'Unknown', 'Action',
         'Adventure', 'Animation', 'Children\'s', 'Comedy',
         'Crime', 'Documentary', 'Drama', 'Fantasy',
         'Film-Noir', 'Horror', 'Musical', 'Mystery',
         'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies = pd.read_csv(os.path.join('data', 'movies.txt'), sep='\t', header=None, encoding='latin_1', names=names)
movies['Movie Title'] = movies['Movie Title'].str.strip()

savefig_kwargs = {'dpi': 200, 'bbox_inches': 'tight', 'transparent': True}
plt.rc('pdf', fonttype=42)

# Separate movie name and year if desired
split_year = False
if split_year:
    movies.loc[266, 'Movie Title'] = 'unknown (0000)'
    movies.loc[1411, 'Movie Title'] =         'Land Before Time III: The Time of the Great Giving (V) (1995)'
    movies['Year'] = [int(title[-5:-1]) for title in movies['Movie Title']]
    movies['Movie Title'] = [title[:-7] for title in movies['Movie Title']]

# Merge ratings data with movie metadata
pdData = ratings.merge(movies, how='left', on='Movie ID')
data = np.random.permutation(pdData)

totRate = np.zeros(len(movies))
numRate = np.zeros(len(movies))
for y in data:
    totRate[y[1] - 1] += y[2]
    numRate[y[1] - 1] += 1
    
avgRate = totRate/numRate

rMovies = movies
rMovies['Average Rating'] = avgRate

trainData = data[0: 90000]
testData = data[90000:]

M = max(max(trainData[:,0]), max(testData[:,0])) # users
N = max(max(trainData[:,1]), max(testData[:,1])) # movies
print("Factorizing with ", M, " users, ", N, " movies.")

regs = np.logspace(-4, 1, 11)
Ks = [1, 2, 5, 10, 20, 40, 80]

trainLosses = np.zeros((len(regs), len(Ks)))
testLosses = np.zeros((len(regs), len(Ks)))

def train_model_mt(para):
    return train_model(*para)

if __name__ == '__main__':
    for idx, reg in enumerate(regs):
        paramVector = [(M, N, K, 0.03, reg, trainData[:, 0:3], 1e-10, 300) for K in Ks]
        # train_model_mt(paramVector[0])
        # print('finished single thread')
        with Pool(12) as p:
            regResults = p.map(train_model_mt, paramVector)
        for jdx, regResult in enumerate(regResults):
            U, V, loss = regResult
            testLosses[idx, jdx] = get_err(U, V, testData[:, 0:3])
            trainLosses[idx, jdx] = loss
        

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    heatmap = ax.pcolor(testLosses)
    ax.set_title('Testing errors')
    ax.set_xlabel('Latent factor K')
    ax.set_ylabel('Regularization')
    ax.set_yticks(np.arange(testLosses.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(testLosses.shape[1]) + 0.5, minor=False)

    ax.set_xticklabels(Ks, minor=False)
    ax.set_yticklabels(['{0:.1e}'.format(a) for a in regs], minor=False)

    fig.colorbar(heatmap)

    plt.show()