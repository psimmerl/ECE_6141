import ROOT
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

def load_data(fname, train_size, add_cols=False, random_state=42, discard_data=False, upsample=False):
    rdf = ROOT.RDataFrame("hits",fname)
    if discard_data:
        rdf = rdf.Filter('ilay > 0')
        # rdf = rdf.Filter('!(ilay == 0 && idet == 2)')

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

    vx, vy, vz = data['vx'], data['vy'], data['vz']
    px, py, pz = data['px'], data['py'], data['pz']
    ilay, idet, ee = data['ilay'].astype(int), data['idet'].astype(int), data['ee']

    X = np.vstack((vx, vy, vz, px, py, pz))
    if add_cols:
        p, pphi, ptheta = data['p'], data['pphi'], data['ptheta']
        r, phi, theta = data['r'], data['phi'], data['theta']
        x0, y0, r0, phi0 = data['x0'], data['y0'], data['r0'], data['phi0']
        X = np.vstack((X, p, pphi, ptheta, r, phi, theta, x0, y0, r0, phi0))
    
    X, y = X.T, ilay#ilay*100+idet#np.array([ilay, idet]).T#
    
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=float(train_size), random_state=random_state)
        

    if upsample:
        mlay = int(max([len(y_train[y_train == lay]) for lay in np.unique(y_train)]))

        X_train1, y_train1 = resample(X_train[y_train == 1],
                          y_train[y_train == 1],
                        replace=True,
                        n_samples=mlay,
                        random_state=random_state)
        X_train2, y_train2 = resample(X_train[y_train == 2],
                          y_train[y_train == 2],
                        replace=True,
                        n_samples=mlay,
                        random_state=random_state)
        X_train3, y_train3 = resample(X_train[y_train == 3],
                          y_train[y_train == 3],
                        replace=True,
                        n_samples=mlay,
                        random_state=random_state)
        X_train4, y_train4 = resample(X_train[y_train == 4],
                          y_train[y_train == 4],
                        replace=True,
                        n_samples=mlay,
                        random_state=random_state)
        X_train5, y_train5 = resample(X_train[y_train == 5],
                          y_train[y_train == 5],
                        replace=True,
                        n_samples=mlay,
                        random_state=random_state)
        X_train = np.vstack((X_train[y_train == 0], X_train1, X_train2, X_train3, X_train4, X_train5))
        y_train = np.hstack((y_train[y_train == 0], y_train1, y_train2, y_train3, y_train4, y_train5))


        Xold, yold = X_train, y_train
        for ind1, ind2 in enumerate(np.random.permutation(len(y_train))):
          X_train[ind1], y_train[ind1] = Xold[ind2], yold[ind2]


    print(fname, rdf.GetColumnNames(), X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    return X_train, X_test, y_train, y_test, rdf

###########################################################################################################

# mlaydet = max([len(y[y == val]) if val else 0 for val in np.unique(y)])
        


        # mlay = int(max([len(ilay[ilay == lay]) if lay != 0 else 0 for lay in sorted(np.unique(ilay))]))


        # X0 = X[0, :]
        # y0 = y[0]
        # for i in [0, 1, 2, 3, 4, 5]:
        #     X1, y1 = resample(X[ilay == i],
        #                   y[ilay == i],
        #                 replace=True,
        #                 n_samples=mlay,
        #                 random_state=random_state)

        #     X0 = np.vstack((X0, X1))
        #     y0 = np.hstack((y0, y1))
        
        # X = X0[1:, :]
        # y = y0[1:]