#Main imports
import sklearn as skl
import pandas as pd
import numpy as np
import ROOT
import os

np.random.seed(42)

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', titlesize=20)
mpl.rc('axes', labelsize=16)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


from pandas.plotting import scatter_matrix


fname = "data/SR-PHOTONS-LUND-NOFMT-20bun-for-Andrey-NoAU.events.root"; oname = "NoAU"
# fname = "data/SR-PHOTONS-LUND-NOFMT-20bun-for-Andrey-AU.events.root"; oname = "AU"
ff = ROOT.TFile(fname)
rdf = ROOT.RDataFrame("hits",fname)
ff.Close()

data = rdf.AsNumpy()
vx, vy, vz = data['vx'], data['vy'], data['vz']
px, py, pz = data['px'], data['py'], data['pz']
ilay, idet, ee = data['ilay'].astype(int), data['idet'].astype(int), data['ee']
X, y = np.array([vx, vy, vz, px, py, pz]).T, ilay*100+idet#np.array([ilay, idet]).T

df = pd.DataFrame(np.c_[X, y], columns = ['vx', 'vy', 'vz', 'px', 'py', 'pz', 'det'])

dfs = []
l = 0
for h in [50, 150, 250, 350, 450, 550]:
    dfs.append(df[(l <= df['det']) & (df['det'] < h)].drop(columns="det"))


bins = 200
fig, axs = plt.subplots(2, 3, figsize=(30,15))
for i, f in enumerate(dfs):
    axs = f.hist(bins=bins, histtype='step', ax=axs.flatten()[:6], label="layer "+str(i))
    
for ax in axs.flatten():
    ax.set_yscale('log')
    ax.set_grid()

axs[0].legend()
fig.savefig("imgs/1Dhists_"+oname+".png")



axs = scatter_matrix(dfs[0], alpha=0.1, figsize=(30,30))
for f in dfs[1:]:
    print(i)
    scatter_matrix(f, alpha=0.1, ax=axs)

for ax in axs.flatten():
    ax.set_grid()
    
plt.savefig("imgs/2Dhists_"+oname+".png")
