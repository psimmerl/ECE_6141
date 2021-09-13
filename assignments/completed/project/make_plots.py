import pandas as pd
import numpy as np
import ROOT
ROOT.gStyle.SetOptStat(0)
ROOT.gROOT.SetBatch()
np.random.seed(42)

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', titlesize=20)
mpl.rc('axes', labelsize=16)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

fname = "data/SR-PHOTONS-LUND-NOFMT-20bun-for-Andrey-NoAU.events.root"; oname = "NoAU"
# fname = "data/SR-PHOTONS-LUND-NOFMT-20bun-for-Andrey-AU.events.root"; oname = "AU"
rdf_all = ROOT.RDataFrame("hits",fname)
rdf_all = rdf_all.Define("pp", "px*px + py*py + pz*pz")
rdf = rdf_all

cols = [c for c in rdf_all.GetColumnNames()]
# cols.remove('ee')
cols.remove('ilay')
cols.remove('idet')
hhs = []
c1 = ROOT.TCanvas("c1","c1",8000,8000)

for layer in range(-1, 6):
    c1.Clear(); c1.Divide(len(cols),len(cols))
    if layer != -1:
        rdf = rdf_all.Filter(f"ilay == {layer}")

    for i in range(len(cols)):
        for j in range(len(cols)):
            c1.cd(i*len(cols)+j+1).SetGrid()
            if i == j:
                hh = rdf.Histo1D((f"h{i*len(cols)+j}",f";{cols[i]};{cols[j]}",100,rdf_all.Min(cols[i]).GetValue(),rdf_all.Max(cols[i]).GetValue()),cols[i])
            else:
                hh = rdf.Histo2D((f"h{i*len(cols)+j}",f";{cols[i]};{cols[j]}",100,rdf_all.Min(cols[i]).GetValue(),rdf_all.Max(cols[i]).GetValue(),100,rdf_all.Min(cols[j]).GetValue(),rdf_all.Max(cols[j]).GetValue()),cols[i],cols[j])
            hh.Draw("COL")
            hhs.append(hh)
            c1.Update()
    c1.Print(f"imgs/layer{layer if layer >= 0 else 's'}_lin.png")
    if layer == -1:
        c1.Print("imgs/plot.pdf(")
    else:
        c1.Print("imgs/plot.pdf")

    c1.Clear(); c1.Divide(len(cols),len(cols))
    
    for i in range(len(cols)):
        for j in range(len(cols)):
            c1.cd(i*len(cols)+j+1).SetGrid()
            if i == j:
                c1.cd(i*len(cols)+j+1).SetLogy(1)
                hh = rdf.Histo1D((f"h{i*len(cols)+j}",f";{cols[i]};{cols[j]}",100,rdf_all.Min(cols[i]).GetValue(),rdf_all.Max(cols[i]).GetValue()),cols[i])
            else:
                c1.cd(i*len(cols)+j+1).SetLogz(1)
                hh = rdf.Histo2D((f"h{i*len(cols)+j}",f";{cols[i]};{cols[j]}",100,rdf_all.Min(cols[i]).GetValue(),rdf_all.Max(cols[i]).GetValue(),100,rdf_all.Min(cols[j]).GetValue(),rdf_all.Max(cols[j]).GetValue()),cols[i],cols[j])
            hh.Draw("col")
            hhs.append(hh)
            c1.Update()
    c1.Print(f"imgs/layer{layer if layer >= 0 else 's'}_log.png")
    if layer == 5:
        c1.Print("imgs/plot.pdf)")
    else:
        c1.Print("imgs/plot.pdf")


# data = rdf.AsNumpy()
# df = pd.DataFrame(data)
# df = df.drop(columns=['ee','ilay','idet'])
# cols = df.columns
# print(cols)
# bins = 50
# fig, axs = plt.subplots(len(cols), len(cols), figsize=(20,20))
# fig.tight_layout()

# for i in range(len(cols)):
#     for j in range(len(cols)):
#         if i == j:
#             axs[i,j].hist(df[cols[i]], bins=bins)
#         else:
#             axs[i,j].hist2d(df[cols[i]], df[cols[j]], bins=bins)
# plt.savefig("imgs/hexbin_vx_vy_"+oname+".png")
# plt.show()
















# vx, vy, vz = data['vx'], data['vy'], data['vz']
# px, py, pz = data['px'], data['py'], data['pz']
# ilay, idet, ee = data['ilay'].astype(int), data['idet'].astype(int), data['ee']







# X, y = np.array([vx, vy, vz, px, py, pz]).T, ilay*100+idet#np.array([ilay, idet]).T

# df = pd.DataFrame(np.c_[X, y], columns = ['vx', 'vy', 'vz', 'px', 'py', 'pz', 'det'])

# dfs = []
# l = 0
# for h in [50, 150, 250, 350, 450, 550]:
#     dfs.append(df[(l <= df['det']) & (df['det'] < h)].drop(columns="det"))


# bins = 200
# fig, axs = plt.subplots(2, 3, figsize=(30,15))
# for i, f in enumerate(dfs):
#     axs = f.hist(bins=bins, histtype='step', ax=axs.flatten()[:6], label="layer "+str(i))
    
# for ax in axs.flatten():
#     ax.set_yscale('log')
#     ax.set_grid()

# axs[0].legend()
# fig.savefig("imgs/1Dhists_"+oname+".png")



# axs = scatter_matrix(dfs[0], alpha=0.1, figsize=(30,30))
# for f in dfs[1:]:
#     print(i)
#     scatter_matrix(f, alpha=0.1, ax=axs)

# for ax in axs.flatten():
#     ax.set_grid()
    
