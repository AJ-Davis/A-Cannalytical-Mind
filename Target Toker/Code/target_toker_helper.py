#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 15:45:47 2018

@author: ajdavis
"""
## Load Libraries
import numpy as np
import pandas as pd
import pandas.io.sql as pd_sql
from sklearn.pipeline import Pipeline
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.plotting import plot_decision_regions
from sklearn import ensemble, tree, model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import patsy
from sklearn.linear_model  import SGDClassifier
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.utils import check_X_y
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn import naive_bayes
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.neighbors  import KNeighborsClassifier as KNN
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.metrics import classification_report
import xgboost as xgb


## General Data Processing and Feature Engineering

def munge_df(df):
    # Create 'target toker'
    df['target'] = df['mrdaypyr'].astype(str).str.extract('(\d+)').astype(int)
    df['target'] = df['target'].apply(lambda x: 1 if (x == 993) or (x < 240) else 0 )
    df['pnrnmlif'] = df['pnrnmlif'].astype(str).str.extract('(\d+)').astype(int)
    df['pnrnmlif'] = df['pnrnmlif'].apply(lambda x: 1 if (x == 1) or (x == 5) else 0 )
    df['target'] = df['target'] + df['pnrnmlif']
    df['target'] = df['target'].apply(lambda x: 1 if (x == 2) else 0 )
    df['target'] = df['target'].replace({1: 'Yes', 0: 'No'})
    

    # Replace 99-coded questions ('skips' and 'don't knows') with nan
    df = df.apply(lambda x: x.str.replace(r'(^.*99.*$)', 'None'), axis=1)
    df = df.apply(lambda x: x.str.replace(r'(^.*94 - .*$)', 'None'), axis=1)
    df = df.apply(lambda x: x.str.replace(r'(^.*97 -.*$)', 'None'), axis=1)
    df = df.apply(lambda x: x.str.replace(r'(^.*999 -.*$)', 'None'), axis=1)
    df = df.apply(lambda x: x.str.replace(r'(^.*998 -.*$)', 'None'), axis=1)
    df = df.apply(lambda x: x.str.replace(r'(^.*994 -.*$)', 'None'), axis=1)
    df = df.apply(lambda x: x.str.replace(r'(^.*997 -.*$)', 'None'), axis=1)
    df = df.apply(lambda x: x.str.replace(r'(^.*85 -.*$)', 'None'), axis=1)
    df = df.apply(lambda x: x.str.replace(r'(^.*89 -.*$)', 'None'), axis=1)
    df = df.apply(lambda x: x.str.replace(r'(^.*98 -.*$)', 'None'), axis=1)
    df = df.apply(lambda x: x.str.replace(r'(^.*989 -.*$)', 'None'), axis=1)
    df = df.replace("None", np.nan)
    
    
    # Remove response numbers from categorical features
    pattern = '|'.join(['0 - ', '1 - ', '2 - ', '3 - ', '4 - ', '5 - ', '6 - ', '7 - ', '8 - ', '9 - ',
            '10 - ', '11 - ', '12 - ', '13 - ', '14 - ', '15 - ', '16 - ', '17 - ', 
            '18 - ', '19 - ', '20 - '])
    
    df = df.apply(lambda x: x.str.replace(pattern, ''), axis=1)
    
    # Turn yes/no questions to dummy
    df = df.replace('Yes', 1)
    df = df.replace('No', 0)
    
    # Feature Engineering
    
    return df
    

    

def analysis_prep(df, columns, d_index):
    df = df[columns] # Select features identified in EDA
    df = df.dropna() # Remove na's
    dummies = columns[d_index:] # Select columns to one hot encoded
    df = pd.get_dummies(df, columns=dummies, drop_first = True) # Create dummies
    
    df.columns = df.columns.str.replace(" - ", "_")
    df.columns = df.columns.str.replace("-", "_")
    df.columns = df.columns.str.replace(" ", "_")
    df.columns = df.columns.str.replace("'", "")
    df.columns = df.columns.str.replace(".", "")
    df.columns = df.columns.str.replace("|", "_")
    df.columns = df.columns.str.replace("&", "_")
    df.columns = df.columns.str.replace(")", "")
    df.columns = df.columns.str.replace("(", "")
    df.columns = df.columns.str.replace("/", "")
    df.columns = df.columns.str.replace("$", "")
    df.columns = df.columns.str.replace(",", "")
    df.columns = df.columns.str.strip()
    
    # Create formula for patsy
    my_formula = df.columns[0] + " ~ " + \
    " + ".join(list(df.columns[1:len(df.columns)])) + ' - 1'

    # Create your feature matrix (X) and target vector (y)
    y, X = patsy.dmatrices(my_formula, data=df, return_type="dataframe")        
    
    return X, y

def feature_selection(X, y, count):
    
    # Chi-squared
    chi2_selector = SelectKBest(chi2, k=count)
    X_kbest = chi2_selector.fit_transform(X, y)
    Chi2_features = list(X.columns[chi2_selector.get_support()])

    # Feature Importance from Random Forest
    ET = ExtraTreesClassifier()
    ET.fit(X, y)    
    importances = list(zip(ET.feature_importances_, X.columns))
    importances.sort(reverse=True)
    FI_features = [x[1] for x in importances][0:count]

    # Recursive Feature Elimination
    lm1 = LogisticRegression()
    rfe = RFE(lm1, count)
    fit = rfe.fit(X, y)
    RFE_features = list(X.columns[fit.support_])

    # PCA
    pca = PCA(n_components=5)
    fit = pca.fit(X)    
    df = pd.DataFrame(pca.components_,columns=X.columns).abs().mean().sort_values(ascending = False)
    PCA_features = list(df.index.values)[0:count]
    
    feats = pd.DataFrame(
        {'Chi2': Chi2_features,
         'Feature_Importance': FI_features,
         'RFE': RFE_features,
         'PCA': PCA_features
        })
    
    return feats


def score_ht_models(X, y, scoring = 'accuracy'):

    models = [
              ('knn', KNN), 
              ('logistic', LogisticRegression),
              ('tree', DecisionTreeClassifier),
              ('forest', RandomForestClassifier),
              ('svc', svm.SVC),
              ('xgb', xgb.XGBClassifier),
              ('SGD-Log', SGDClassifier),
              ('Naive Bayes', naive_bayes.BernoulliNB)
             ]
    
    param_choices = [
        {
            'n_neighbors': np.arange(1, 12, 1)
        },
        {
            'C': np.logspace(-3,6, 12),
            'penalty': ['l1', 'l2']
        },
        {
            'max_depth': [1,2,3,4,5],
            'min_samples_leaf': [3,6]
        },
        {
            'n_estimators': [50, 100, 200],
            'max_depth': [1, 5, 10, 20],
            'min_samples_leaf': [1,2,4,6,10]
        },
        {
            'C': [0.001, 0.01, 0.1, 1, 10],
            'gamma': [1, 2, 4, 10, 20, 50]
        },
        {
             "objective": ["binary:logistic"],
             'min_child_weight': [1, 5],
             'subsample': [0.8, 1],
             'gamma': [0.5, 2],
             'colsample_bytree': [0.8, 1],
             'learning_rate': [0.3],
             'max_depth': [4, 5], 
             'reg_alpha': [5],
             'n_jobs': [-1]
        },
        {
            'loss': ["log"]
        },
        {
        }
    ]
    
    grids = {}
    for model_info, params in zip(models, param_choices):
        print(model_info)
        name, model = model_info
        grid = GridSearchCV(model(), params, scoring = scoring)
        grid.fit(X, y)
        s = f"{name}: best score: {grid.best_score_}"
        print(s)
        grids[name] = grid
    return grids       


def make_prediction(features):
    X = np.array([int(features['Age_18-25'] == 'Not Age_18-25'), \
                 int(features['White'] == 'White'), \
                 int(features['Enrolled'] == 'Enrolled'), \
                 int(features['Less_$10k'] == 'More_$10k'), \
                 int(features['Food_Stamps'] == 'Not Food_Stamps')]).reshape(1,-1)
    
    prob_target_toker = pipeline.predict_proba(X)[0, 1]

    result = {
        'prediction': int(prob_target_toker > 0.5),
        'prob_target_toker': prob_target_toker
    }
    return result


# Helper function for printing confusion matrices (see: https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823)# Helpe 
def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=18):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    return fig


        
def plot_rocs(grids, y, X):
    roc_plotting_stuff = []
    for key, value in grids.items():
        try:
            y_predict_proba = grids[key].best_estimator_.predict_proba(X).tolist()
        except Exception as e:
            print(e)
        probabilities = np.array(y_predict_proba)[:, 1]
        fpr, tpr, _ = roc_curve(y, probabilities)
        roc_auc = auc(fpr, tpr)
        roc_plotting_stuff.append((key, tpr, fpr, roc_auc))
            
    fig = plt.figure(dpi=250)
    for key, tpr, fpr, auc_score in roc_plotting_stuff:
        plt.plot(fpr, tpr, label=key+' (auc: %.2f)'%auc_score)
    plt.legend(loc='lower right', fontsize=9)
    plt.plot([0, 1], [0, 1], color='k', linestyle='--');
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Comparing ROC Curves")
    return fig