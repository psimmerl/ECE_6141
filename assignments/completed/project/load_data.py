from numpy.core.fromnumeric import mean
from numpy.lib.arraysetops import unique
from numpy.lib.function_base import median
import ROOT
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

def load_data(fname, train_size, add_cols=False, random_state=42, discard_data=False, upsample=False):
    # ff = ROOT.TFile(fname)
    rdf = ROOT.RDataFrame("hits",fname)
    # rdf = rdf.Filter('ilay > 0')
    # rdf = rdf.Filter('!(ilay == 0 && idet == 2)')
    # ff.Close()

    if add_cols:
        rdf = rdf.Define("p", "sqrt(px*px+py*py+pz*pz)")
        rdf = rdf.Define("pphi", "atan2(py, px)*TMath::RadToDeg()")
        rdf = rdf.Define("ptheta", "acos(pz/p)*TMath::RadToDeg()")

        rdf = rdf.Define("r", "sqrt(vx*vx+vy*vy+vz*vz)")
        rdf = rdf.Define("phi", "atan2(vy, vx)*TMath::RadToDeg()")
        rdf = rdf.Define("theta", "acos(vz/r)*TMath::RadToDeg()")

        rdf = rdf.Define("x0","vx+px*abs(vz/pz)")
        rdf = rdf.Define("y0","vy+py*abs(vz/pz)")
        rdf = rdf.Define("r0", "sqrt(x0*x0+y0*y0)")
        rdf = rdf.Define("phi0", "atan2(y0, x0)*TMath::RadToDeg()")


    data = rdf.AsNumpy()
    # ff.Close()

    vx, vy, vz = data['vx'], data['vy'], data['vz']
    px, py, pz = data['px'], data['py'], data['pz']
    ilay, idet, ee = data['ilay'].astype(int), data['idet'].astype(int), data['ee']

    X = np.vstack((vx, vy, vz, px, py, pz, ee))
    if add_cols:
        p, pphi, ptheta = data['p'], data['pphi'], data['ptheta']
        r, phi, theta = data['r'], data['phi'], data['theta']
        x0, y0, r0, phi0 = data['x0'], data['y0'], data['r0'], data['phi0']
        X = np.vstack((X, p, pphi, ptheta, r, phi, theta, x0, y0, r0, phi0))
    
    X, y = X.T, ilay#ilay*100+idet#np.array([ilay, idet]).T#

    if discard_data:
        # mlaydet = max([len(y[y == val]) if val else 0 for val in np.unique(y)])
        
        mlay = int(median([len(ilay[ilay == lay]) if lay != 0 else 0 for lay in sorted(np.unique(ilay))]))
        XO = X[ilay != 0]
        yO = y[ilay != 0]

        XBE1, XBE2 = X[ilay == 0], X[ilay == 0]
        yBE1, yBE2 = y[ilay == 0], y[ilay == 0]
        permutation = np.random.permutation(len(XBE1))
        for old_index, new_index in enumerate(permutation):
            XBE1[new_index] = XBE2[old_index]
            yBE1[new_index] = yBE2[old_index]

        X = np.r_[XO, XBE1[:mlay, :]]        
        y = np.append(yO, yBE1[:mlay])

    if upsample:
        mlay = int(max([len(ilay[ilay == lay]) for lay in np.unique(ilay)]))

        X1, y1 = resample(X[ilay == 1],
                          y[ilay == 1],
                        replace=True,
                        n_samples=mlay,
                        random_state=random_state)
        X2, y2 = resample(X[ilay == 2],
                          y[ilay == 2],
                        replace=True,
                        n_samples=mlay,
                        random_state=random_state)
        X3, y3 = resample(X[ilay == 3],
                          y[ilay == 3],
                        replace=True,
                        n_samples=mlay,
                        random_state=random_state)
        X4, y4 = resample(X[ilay == 4],
                          y[ilay == 4],
                        replace=True,
                        n_samples=mlay,
                        random_state=random_state)
        X5, y5 = resample(X[ilay == 5],
                          y[ilay == 5],
                        replace=True,
                        n_samples=mlay,
                        random_state=random_state)
        X = np.vstack((X[ilay == 0], X1, X2, X3, X4, X5))
        y = np.hstack((y[ilay == 0], y1, y2, y3, y4, y5))

    print(fname, rdf.GetColumnNames(), X.shape, y.shape)


    return train_test_split(X, y, train_size=train_size, random_state=random_state), rdf
