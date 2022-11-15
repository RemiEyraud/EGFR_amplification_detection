#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 2022

@author: Rémi Eyraud

EGFR_amplification_detection is free software: you can redistribute it and/or 
modify it under the terms of the GNU General Public License as published by the 
Free Software Foundation, either version 3 of the License, or (at your option) 
any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY 
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with 
this program. If not, see <https://www.gnu.org/licenses/>.
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import LeaveOneOut

def learn(clf, X, y):
    """
    Leave one out.
    Get number and type of badly classified data in test 
    """
    correct = 0
    fake_positive_test = 0
    fake_negative_test = 0

    loo = LeaveOneOut()
    loo_index = loo.split(X)

    for train, test in loo_index:
        clf.fit(X[train, :], y[train])
        predit = clf.predict(X[test, :])[0]
        """ Treating error on the sole test data """
        if predit == y[test]:
            correct += 1
        elif int(predit) == 0:
            """ Fake negative"""
            fake_negative_test += 1
        else:
            """ Fake positive """
            fake_positive_test += 1
    accuracy = correct / X.shape[0]
    print('accuracy on test set:', accuracy*100,
          '\n\t with %s fake positives and %s fake negatives' 
          % (fake_positive_test, fake_negative_test))

    
""""""
""" Data preparation"""
""""""

"""Load data for predicting EGFR status"""
csv="./data_egfr.csv"
df = pd.read_csv(csv, header=None)

"""Get the class vector, that is, EGFR information"""
y = np.array(df[df.columns[-1]])

"""Get the output of the nanoDSF machine"""
X = np.array(df.drop(df.columns[-1], axis = 1))

print("Nb Data", X.shape[0], "with", X[y==0].shape[0], "negatives and", 
      X.shape[0]-X[y==0].shape[0], "positives.")

"""Standardize"""
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

""""""
""" Learning & Testing via Leave One Out"""
""""""

""" Fix randomness for reproductivity """
seed = 1
np.random.seed(seed)
print("Seed value:",seed)

""" Adaptative Boosting """

print("* AdaBoost")
from sklearn.ensemble import AdaBoostClassifier
clf_boost = AdaBoostClassifier(n_estimators = 100)
learn(clf_boost, X, y)

""" Random Forest """
from  sklearn.ensemble import RandomForestClassifier
print("* Random Forest")
clf_rf = RandomForestClassifier(n_estimators = 500)
learn(clf_rf, X, y)

""" SVM """
from sklearn.svm import SVC
print("* SVM ")
clf_poly = SVC(kernel='poly', degree=3, gamma='auto', C=1)
learn(clf_poly, X, y)

""" Logistic regression """
from sklearn.linear_model import LogisticRegression
print("* Logistic Reg")
clf_lr = LogisticRegression(solver='liblinear', max_iter=1000)
learn(clf_lr, X, y)
