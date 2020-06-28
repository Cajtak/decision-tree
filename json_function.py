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



#NaN Values
class nanValues: 
    """"Displays a table with the percentage of missing values sorted in descending order
        n : number of lines displayed """
    
    def count(self, data, n=20):
        total = data.isnull().sum().sort_values(ascending=False)
        percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending=False)
        return pd.concat([total, percent], axis=1, keys=['Total', 'Percent']).head(n)


class correlation():
    """
    Displays a correlation matrix of all variables"""
    
    def plotMatrix(data):
        corr_data=data.corr(method='pearson')
        sns.set(style="white")
    
        mask = np.zeros_like(corr_data, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
    
        f, ax = plt.subplots(figsize=(40, 40))
    
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
        sns.heatmap(corr_data, mask=mask, cmap=cmap, annot=True, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        
    def listMatrix (data, var='variable'):
        '''
        # Displays a descending list of correlations with the target variable

        data = dataframe 
        var = variable to calculate the correlation
        '''
        corr_matrix = data.corr(method='pearson')
        print(corr_matrix[var].sort_values(ascending=False))

def data_clean_missing (data):
    '''
    # Nettoie les données manquantes en remplaçant par les modes ou médiandes les plus fréquentes
    '''
    #Complete les données manquantes par le mode le plus fréquent des variables
    data.fillna(data.mode().iloc[0], inplace=True)
    
    #Complete les données manquantes par la note mediane
    data['note'].fillna(data['note'].median(), inplace = True)

    #Complete les données manquantes par l'âge median
    data['age'].fillna(data['age'].median(), inplace = True)

    #Complete les données manquantes par le salaire median
    data['salaire'].fillna(data['salaire'].median(), inplace = True)
    
    #Complete les données manquantes par l'expérience moyenne
    data['exp'].fillna(data['exp'].median(), inplace = True)
        
    return(data)

def dicho_nominales (data, cols=['var1','var2']):
    '''
    # Dichotomise les variables catégorielles nominales 
    '''
    enc = ce.BinaryEncoder(cols=cols)
    data_drop = enc.fit_transform(data)
        
    return(data_drop)

def dichotomisation (data):
    '''
    # Dichotomise les variables catégorielles binaires et non binaires
    '''
    #Les variables sexe et dispo sont binaires, on peut utiliser cat.codes pour les encoder
    data['dispo'] = data['dispo'].astype('category').cat.codes
    data['sexe'] = data['sexe'].astype('category').cat.codes
    
    #Pout les variables non binaires nous devons encoder différemment
    data['diplome'] = data['diplome'].map({'bac':0, 'licence':1, 'master':2, 'doctorat':3} ).astype(int) #Variable ordinale
    
    return(data)

def scale_data (data):
    '''
        # Standardise les données
        '''
    scaler = preprocessing.StandardScaler().fit(data)
    data_scaled = scaler.transform(data)
    columns = data.columns
    data_scaled = pd.DataFrame(data=data_scaled, columns=columns)
    
    return (data_scaled)

def extract_date (dataset, var='date'):
    '''
    # Transformation de la variable date en datetime et extraction des années, mois, jours
    '''
    for i in dataset: 
        dataset.loc[:,var] = pd.to_datetime(dataset.loc[:,var])
    for i in dataset:
        dataset['month'] = pd.DatetimeIndex(dataset[var]).month
    
        dataset['year']= pd.DatetimeIndex(dataset[var]).year
    
        dataset['day']=pd.DatetimeIndex(dataset[var]).weekday
    dataset.drop(var, axis=1, inplace=True)
    return dataset

def bbc_train_opti (X_train, y_train):
        
    bbc = BalancedBaggingClassifier(bootstrap=False, warm_start=True, n_jobs=-1)

    params = {
            'n_estimators': [100, 1000, 5000],
            'sampling_strategy':['majority', 0.7, 0.9],
            'max_samples':[0.1,0.2, 0.4]
            }
    
    scorers = {
            'precision_score': make_scorer(precision_score),
            'recall_score': make_scorer(recall_score),
            'f1_score': make_scorer(f1_score)
            }

    grid = GridSearchCV(estimator=bbc, 
                        param_grid=params,
                        scoring=scorers,
                        refit='f1_score',
                        cv=5, 
                        verbose=3)
    
    grille_bbc = grid.fit(X_train, y_train)
    results_bbc = grille_bbc.cv_results_
    
    return (grille_bbc, results_bbc)

def jsonFileTree (clf, features, labels,original_features, node_index=0,side=0):
      
    node = {}
    if clf.tree_.children_left[node_index] == -1:  # indicates leaf
        count_labels = zip(clf.tree_.value[node_index, 0], labels)
        node['name'] = ', '.join(('{} of {}'.format(int(count), label)
                                  for count, label in count_labels))
        node['size'] = sum( clf.tree_.value[node_index, 0]  )   
        node['side'] = 'left' if side == 'l' else 'right'        
        node['side_bool'] = 'True' if side == 'l' else 'False'   
    else:

        count_labels = zip(clf.tree_.value[node_index, 0], labels)
        node['pred'] = ', '.join(('{} of {}'.format(int(count), label)
                                  for count, label in count_labels))
                                      
        node['side'] = 'left' if side == 'l' else 'right'
        node['side_bool'] = 'True' if side == 'l' else 'False' 
        feature = features[clf.tree_.feature[node_index]]
        threshold = clf.tree_.threshold[node_index]
        
        if ('_-_' in feature) and (feature not in original_features):
            node['name'] =  '{} = {}'.format(feature.split('_-_')[0], feature.split('_-_')[1] )
            node['type'] = 'categorical'
        else:
            node['name'] = '{} > {}'.format(feature, round(threshold,2) )
            node['type'] = 'numerical'
        
        left_index = clf.tree_.children_left[node_index]
        right_index = clf.tree_.children_right[node_index]
        
        node['size'] = sum (clf.tree_.value[node_index, 0])
        node['children'] = [jsonFileTree(clf, features, labels, original_features, right_index,'r'),
                            jsonFileTree(clf, features, labels, original_features, left_index,'l')]
                            
        
    return node

def results_models (grille, X_test, y_test):
    '''
    # Affiche les paramètres pour les meilleurs résultats pour GridsearchCV
    '''
    print(pd.DataFrame.from_dict(grille.cv_results_)[["params","mean_test_f1_score"]])
    print("--------------------------------------------------------")
    print("Meilleurs paramètres pour le modèle:",grille.best_params_)
    print("--------------------------------------------------------")

def eval_model (clf, X_test, y_test):
    '''
    # Affiche une matrice de confusion et les métriques du modèle
    '''
    y_pred=clf.predict(X_test)
    y_test_int=y_test.astype(int)

    confu_mat = pd.crosstab(y_test_int, y_pred, rownames=["Classe réelle"],colnames=["Classe prédite"])
    
    classif_report = classification_report(y_test, y_pred)
    
    return (print("Confusion matrice:"),
            print(confu_mat),
            print("Classification report"), 
            print(classif_report))