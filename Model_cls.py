#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 20:27:52 2020

@author: faigagaumand
"""

import pandas as pd 
import numpy as np 

#Data Viz
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#Scale 
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

#Model
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import tree
import json
from imblearn.ensemble import EasyEnsembleClassifier, BalancedBaggingClassifier, EasyEnsemble
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score,recall_score, precision_score, classification_report

#Features engineering
import category_encoders as ce

from json_function import data_clean_missing, dicho_nominales, extract_date, dichotomisation, jsonFileTree, bbc_train_opti, results_models, eval_model

class Model:
    def __init__(self, datafile = "/Users/faigagaumand/Documents/D3JS_Project/data_énondé.csv"):
        self.df = pd.read_csv(datafile)
        self.clf = tree.DecisionTreeClassifier(max_depth=10)
        
    def preprocess (self, df):
        self.df_clean = data_clean_missing(self.df)
        self.df_dicho = dicho_nominales(self.df_clean, cols=['specialite','cheveux'])
        self.df_date = extract_date(self.df_dicho)
        self.df_dichotomisation = dichotomisation(self.df_dicho)
        self.df_scale = scale_data(self.df_dichotomisation)
        
    def split (self, test_size):
        X = self.df_dichotomisation.drop(["Unnamed: 0","index",'embauche'], axis=1)
        y = self.df_dichotomisation['embauche']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = test_size, random_state = 42)
        
    def fit(self):
        self.model = self.clf.fit(self.X_train, self.y_train)
    
    def getJsonFile(self): 
        self.features = self.X_test.columns
        self.label = 'embauche'
        self.labels_name = np.unique(self.df[self.label])
        original_features = self.X_test.columns
        self.jsonFile = jsonFileTree(self.clf, self.features, self.labels_name, original_features=self.features, node_index=0,side=0)
        
    def balancebaggingclassifier(self): 
        self.X_train_scaled = scale_data(self.X_train)
        self.grille, self.result = bbc_train_opti(self.X_train_scaled, self.y_train)
        self.results = results_models(self.grille, self.X_test, self.y_test)
        self.evals = eval_model(self.grille, self.X_test, self.y_test)
        
        
if __name__ == '__main__':
    model_instance= Model()
    model_instance.preprocess(df)
    model_instance.split(0.2)
    model_instance.fit()
    model_instance.getJsonFile()
    print(model_instance.X_train)
    print(model_instance.balancebaggingclassifier())
    print(model_instance.grille)
    print(model_instance.evals)
    print(model_instance.results)
    with open('projet_class.json', 'w') as outfile:
        json.dump(model_instance.jsonFile, outfile, indent=4)
