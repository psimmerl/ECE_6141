# Analysis for the Application of Machine Learning to the EIC Synchrotron Radiation
#Main imports
import sklearn as skl
import numpy as np
import os

from time import time, process_time, localtime
from load_data import load_data
from joblib import dump

from sklearn.multiclass import OneVsOneClassifier,  OneVsRestClassifier
from sklearn.preprocessing import StandardScaler,  LabelBinarizer
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,  GradientBoostingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score

np.random.seed(42)

fname = "data/SR-PHOTONS-LUND-NOFMT-20bun-for-Andrey-NoAU.events.root"; oname = "NoAU"
# fname = "data/SR-PHOTONS-LUND-NOFMT-20bun-for-Andrey-AU.events.root"; oname = "AU"
(X_train, X_test, y_train, y_test), rdf = load_data(fname, .8, add_cols=True, discard_data=True)

# lb = MultiLabelBinarizer()
# lb.fit(np.r_[y_train,y_test])
lb = LabelBinarizer() 
lb.fit(np.append(y_train,y_test))

scaler = StandardScaler() 
scaler.fit(np.r_[X_train,X_test])

X_train_transformed = scaler.transform(X_train)
y_train_transformed = lb.transform(y_train)
X_test_transformed = scaler.transform(X_test)
y_test_transformed = lb.transform(y_test)

def train(X, y, clf, param_grid, out):
    t1, t1p = time(), process_time()
    clf = Pipeline([('scaler', StandardScaler()),
                    (out, clf)])
    grid_search = GridSearchCV(clf, param_grid, scoring='balanced_accuracy', cv=3, n_jobs=16, return_train_score=True, verbose=100)
    grid_search.fit(X, y)

    tt, ttp = time()-t1, process_time()-t1p
    t = localtime()
    print(f"Finished fitting {out}:\tElapsed time {ttp//60**2:02.0f}:{ttp//60:02.0f}:{ttp%60:02.0f}"+
            f" process, {tt//60**2:02.0f}:{tt//60:02.0f}:{tt%60:02.0f} clock")
    
    dump(grid_search, f"models/{out}_{t.tm_mon:02d}{t.tm_mday:02d}{t.tm_year-2000:02d}_{t.tm_hour:02d}{t.tm_min:02d}.model")

    return grid_search

clfs = np.array([ RandomForestClassifier(), 
                  AdaBoostClassifier(), 
                  GradientBoostingClassifier(), 
                  LinearSVC(), 
                  GaussianNB(), 
                #   KNeighborsClassifier(), 
                  QuadraticDiscriminantAnalysis() ] )

param_grids = [ { 'random_forest__n_estimators' : [3, 10, 30, 100, 200, 300], 'random_forest__max_features' : [1, 3, 6], 'random_forest__class_weight' : ['balanced'] },
                { 'adaboost__base_estimator' : [DecisionTreeClassifier(max_depth=1, class_weight='balanced'), DecisionTreeClassifier(max_depth=10, class_weight='balanced')], 
                    'adaboost__learning_rate' : [0.01, 0.1, 1], 'adaboost__n_estimators' : [10, 50, 100, 200] },
                { 'gradient_boost__learning_rate' : [0.01, 0.1, 1], 'gradient_boost__n_estimators' : [10, 50, 100] },
                { 'linearSVC__C' : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100], 'linearSVC__class_weight' : ['balanced'] },
                { 'gaussianNB__var_smoothing' : [1e-9] },
                # { 'knn__n_neighbors' : [1, 2, 3, 10] },
                { 'quad_discrim__tol' : [1e-4] } ]
clfs_names = np.array([ "random_forest", "adaboost", "gradient_boost", "linearSVC", "gaussianNB", "quad_discrim"])#"knn",

for i in range(len(clfs)):
    clfs[i] = train(X_train_transformed, y_train, clfs[i], param_grids[i], clfs_names[i])



for i, clf in enumerate(clfs):
    try:
        clf = clf.best_estimator_
        y_predp = clf.predict_proba(X_test_transformed)
        y_pred = clf.predict(X_test_transformed)
        auc_score = roc_auc_score(y_test_transformed, y_predp, average='weighted')
        acc_score = balanced_accuracy_score(y_test, y_pred)
        print(f"{clfs_names[i]} / accuracy = {acc_score} / AUC = {auc_score}")
    except:
        print(f"failed {clfs_names[i]}")


#######################################################
# without ilay == 0
# random_forest / accuracy = 0.3461339859106035 / AUC = 0.6521373390754522
# adaboost / accuracy = 0.3414255560752998 / AUC = 0.6516690203875132
# gradient_boost / accuracy = 0.33404828712956336 / AUC = 0.669387794797012
# failed linearSVC
# gaussianNB / accuracy = 0.23524013492170726 / AUC = 0.5444328316164044
# knn / accuracy = 0.3131625879530021 / AUC = 0.6243241938000604
# quad_discrim / accuracy = 0.24215584422483155 / AUC = 0.5645352614243804
#######################################################3


#######################################################3
# with ilay == 0 and new custom resample
# Finished fitting quad_discrim:	Elapsed time 00:00:00 process, 00:00:00 clock
# random_forest / accuracy = 0.4766628817026763 / AUC = 0.7607250390351693
# adaboost / accuracy = 0.47370574396800585 / AUC = 0.773472526869468
# gradient_boost / accuracy = 0.4173107811061085 / AUC = 0.7661738875008145
# failed linearSVC
# gaussianNB / accuracy = 0.2880187328063251 / AUC = 0.6468537084311059
# knn / accuracy = 0.4520205667876944 / AUC = 0.6702585061281358
# quad_discrim / accuracy = 0.29710736449341196 / AUC = 0.6624272056884883
#######################################################3


