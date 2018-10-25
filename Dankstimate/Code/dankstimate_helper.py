#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 15:07:50 2018

@author: ajdavis
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.metrics import mean_squared_error
from math import sqrt
import re
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import ProbPlot
import patsy
import itertools
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn import preprocessing
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
import random
from fuzzywuzzy import process, fuzz
from sklearn import linear_model,ensemble, tree, model_selection, datasets
import numpy as np
import pickle
import warnings
import json
import urllib.request
import geocoder
from bs4 import BeautifulSoup
from selenium import webdriver
import time
import os
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV
from scipy.stats import uniform as sp_rand
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import xgboost as xgb
import Levenshtein
import geocoder
warnings.filterwarnings('ignore')

## Helper functions

def get_gcs(list_names):
    gcs = []
    for name in list_names:
        g = geocoder.yandex(name)
        if g.status == 'OK':
            gc = g.latlng
            gcs.append(gc)
            gcs = list(map(str, gcs))
            gcs = [gc.replace('[', '') for gc in gcs]
            gcs = [gc.replace(']', '') for gc in gcs]
            gcs = [gc.replace(', ', ',') for gc in gcs]
            gcs = [gc.replace("'", "") for gc in gcs]
            gcs = [i for i in gcs if i != 'None']
        else:
            print('status: {}'.format(g.status))
    return gcs

def get_disp_names(gc):
    url = 'https://api-g.weedmaps.com/wm/v2/location?include%5B%5D=regions.listings&latlng={}'.format(gc)
    req = urllib.request.urlopen(url)
    data = json.loads(req.read().decode('utf-8'))
    try:
        lst = data['data']['regions']['dispensary']['listings']
        disp_df = pd.DataFrame(lst)
        disps = list(disp_df['slug'].unique())
    except:
        print(gc)
        print('No Dispensaries')
        disps = []
    return disps


def menu_to_df(disp_names):
    try:
        time.sleep(1)
        url = 'https://api-g.weedmaps.com/wm/web/v1/listings/{}/menu?type=dispensary'.format(disp_names)
        req = urllib.request.urlopen(url)
        data = json.loads(req.read().decode('utf-8'))
        disp_df = pd.DataFrame(
                {'dname':[data['listing']['name']],
                 'rating':[data['listing']['rating']],
                 'region':[data['listing']['region']],
                 'zip':[data['listing']['zip_code']],
                 
                })

        m_dict = { k:[d[k] for d in data['categories']] for k in data['categories'][0] }
        flowers = ['Indica', 'Sativa', 'Hybrid']
        weed_df = pd.DataFrame()

        for d in m_dict['items']:
            df = pd.DataFrame.from_dict(d)
            weed_df = weed_df.append(df)
            weed_df = weed_df[['prices', 'body', 'license_type', 
                               'category_name', 'name', 'listing_name']]
            weed_df = weed_df[weed_df['category_name'].isin(flowers)]
            weed_df['THC'] = weed_df['body'].astype(str).str.extract("(\d+.\d+)%").astype(float) 

        disp_df = pd.concat([disp_df]*len(weed_df), ignore_index=True)
        disp_df.reset_index(inplace=True, drop=True)
        weed_df.reset_index(inplace=True, drop=True)
        weed_df = pd.concat([weed_df, disp_df], axis=1)
        return weed_df
    except:
        print("Error")
    


def munge_df(df):

    df['prices'] = df['prices'].astype(str)
    df["prices"] = df["prices"].apply(lambda x : dict(eval(x)) )
    tmp = df["prices"].apply(pd.Series )
    df = pd.concat([df, tmp], axis=1)
    df['two_grams'] = (df['two_grams']/2).replace(0, np.nan)
    df['eighth'] = (df['eighth']/3.5).replace(0, np.nan)
    df['quarter'] = (df['quarter']/7).replace(0, np.nan)
    df['half_ounce'] = (df['half_ounce']/14).replace(0, np.nan)
    df['ounce'] = (df['ounce']/28).replace(0, np.nan)
    df['price'] = df[['gram', 'two_grams', 'eighth', 'quarter', 'half_ounce', 'ounce']].mean(axis=1)
    
    df['gram'] = df['gram'].apply(lambda x: 0 if pd.isnull(x) else 1)
    df['two_grams'] = df['two_grams'].apply(lambda x: 0 if pd.isnull(x) else 1)
    df['eighth'] = df['eighth'].apply(lambda x: 0 if pd.isnull(x) else 1)
    df['quarter'] = df['quarter'].apply(lambda x: 0 if pd.isnull(x) else 1)
    df['half_ounce'] = df['half_ounce'].apply(lambda x: 0 if pd.isnull(x) else 1)
    df['ounce'] = df['ounce'].apply(lambda x: 0 if pd.isnull(x) else 1)
    df = df[df['price'] != 0] # remove 0 obs
    
    # Fix strain name text
    df['Strain Name'] = df['name'].str.split(' by |:|-|\(|\|').str[0]
    df['MatchedName'] = df['Strain Name'].str.upper()
    df['MatchedName'] = df['MatchedName'].str.replace('[^A-Za-z0-9]+', '')
    
    # Remove outliers
    df = (df.loc[df['price'] < 1000]) # Prices above $1000/gram very unlikely
    df[np.abs(df['price']-df['price'].mean())<=(3*df['price'].std())] #keep only the ones that are within +3 to -3 standard deviations
    df[np.abs(df['THC']-df['THC'].mean())<=(3*df['THC'].std())]
    return df


def get_top_strains():
    url = 'http://cannabis.net/blog/opinion/15-most-popular-cannabis-strains-of-all-time'
    response = requests.get(url)
    print(response.status_code)
    page = response.text
    soup = BeautifulSoup(page,"html5lib")   
    text = [x.get_text() for x in soup.find_all('a', {'href': re.compile('https://cannabis.net/strains/')})] 
    top_strains = list(filter(None, text))
    top_strains = [x.upper() for x in top_strains]
    top_strains = [x.replace(' ', '') for x in top_strains]
    
    return top_strains

def ts_feature(df):
    top_strains = get_top_strains() # Scrape top strains
    text = df['MatchedName']
    match_list = []
    
    for i in text:
        result = process.extract(i, pd.Series(top_strains), scorer=fuzz.ratio)[0]
        match_list.append(result)
    
    match = pd.Series(match_list)
    df['Match'] = match.values
    
    df[['Matched Strain', 'Match Score', '_Index']] =  df['Match'].apply(pd.Series)
    
    return df


def content_feature(df):
    content = pd.read_csv('/Users/ajdavis/Desktop/GitHub/A-Cannalytical-Mind/Dankstimate/Data/I_502.csv', encoding='iso-8859-1')
    mask = ['MatchedName', 'THCmax', 'CBDmax']
    content = content[mask]
    content = content.groupby(['MatchedName']).mean().reset_index()
    text = df['MatchedName']

    match_list = [process.extractOne(i, content['MatchedName'], scorer=fuzz.ratio) for i in text]    
    
    match = pd.Series(match_list)
    df['MatchNew'] = match.values
    df[['Matched Strain', 'Match Score', '_Index']] =  df['MatchNew'].apply(pd.Series)
    df.set_index('_Index', inplace=True)
    content = content.drop('MatchedName', 1)
    df = df.join(content, how='left')
            
    return df

def get_strain_info():
    chromedriver = "/Applications/chromedriver" # path to the chromedriver executable
    os.environ["webdriver.chrome.driver"] = chromedriver
    driver = webdriver.Chrome(chromedriver)
    driver.get("https://weedmaps.com/strains")
    more_button=driver.find_element_by_xpath('.//a[@class="btn btn-more-strains"]')

    while True:
        try:
            more_button=driver.find_element_by_xpath('.//a[@class="btn btn-more-strains"]')
            time.sleep(2)
            more_button.click()
            time.sleep(10)
        except Exception as e:
            print(e)
            break
    print('Complete')
    time.sleep(10)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    text = [x.get_text().strip() for x in soup.find_all('div', class_='strain-cell Hybrid')] 
    text = [i.replace('\n', '') for i in text]
    text = [re.sub('  +', ',', i) for i in text]
    d = {}
    for b in text:
        i = b.split(',')
        x = i[2].split('THC')
        try:
            x[1] = x[1].replace('CBD', '')
        except:
            print('Error')
        try:
            d[i[0]] = float(x[0].strip().replace('%', ''))

        except:
            print('Error')
    driver.quit()
    return d


def plot_overfit(X,y,model_obj,param_ranges,param_static=None): 
    for parameter,parameter_range in param_ranges.items():
        avg_train_score, avg_test_score = [],[]
        std_train_score, std_test_score = [],[]
        
        for param_val in parameter_range:
            param = {parameter:param_val}
            if param_static:
                param.update(param_static)
            
                
            model = model_obj(**param)
            
            train_scores,test_scores = [],[]
            for i in range(5):
                X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size = .3)
                model.fit(X_train,y_train)
                
                train_scores.append(model.score(X_train,y_train))
                test_scores.append(model.score(X_test,y_test))
            
            avg_train_score.append(np.mean(train_scores))
            avg_test_score.append(np.mean(test_scores))
            
            std_train_score.append(np.std(train_scores))
            std_test_score.append(np.std(test_scores))
            
        fig,ax = plt.subplots()
        ax.errorbar(parameter_range,avg_train_score,yerr=std_train_score,label='training score')
        ax.errorbar(parameter_range,avg_test_score,yerr=std_test_score,label='testing score')
        
        ax.set_xlabel(parameter)
        ax.set_ylabel('score')
        ax.legend(loc=0)  


    
def score_ht_models(X, y, scoring = 'neg_mean_absolute_error'):

    models = [
              ('linear_model', linear_model.LinearRegression), 
              ('ridge_model', linear_model.Ridge),
              ('lasso_model', linear_model.Lasso),
              ('robust_regression', linear_model.SGDRegressor),
              ('eps_insensitive', linear_model.SGDRegressor),
              ('xgb', xgb.XGBRegressor),
              ('randomForest', ensemble.RandomForestRegressor),
              ('cart', tree.DecisionTreeRegressor), 
              ('extratrees', tree.ExtraTreeRegressor), 
              ('adaboostedTrees', ensemble.AdaBoostRegressor),
              ('gradboostedTrees', ensemble.GradientBoostingRegressor)

             ]
    
    param_choices = [
         {

         },
         {
             'alpha': [0.1, 0.2, 0.5, 0.7]
         },
         {
             'alpha': [0.1, 0.2, 0.5, 0.7]
         },
         {
             'loss': ['huber'],
             'max_iter': [1000, 2000]
         },
         {
             'loss': ['epsilon_insensitive'],
             'max_iter': [1000, 2000]
         },
         {
             "objective": ["reg:linear"],
             'min_child_weight': [3],
             'subsample': [0.3],
             'gamma': [0.3],
             'colsample_bytree': [0.3],
             'learning_rate': [0.3],
             'max_depth': [5], 
             'reg_alpha': [5],
             'n_jobs': [-1]
         },
                  {
             'bootstrap': [True],
             'max_depth': [80],
             'max_features': [2, 3],
             'min_samples_leaf': [3],
             'min_samples_split': [5, 8],
             'n_estimators': [300],
             'n_jobs':[-1]
         }
            ,
         {
             'max_depth': [2, 5, 7]
         },
         {
             'max_depth': [2, 5, 7]
         },
         {
             'bootstrap': [True],
             'max_depth': [80, 90, 100, 110],
             'max_features': [2, 3],
             'min_samples_leaf': [3, 4, 5],
             'min_samples_split': [8, 10, 12],
             'n_estimators': [100, 200, 300, 1000] 
         },
         {
             'bootstrap': [True],
             'max_depth': [80, 90, 100, 110],
             'max_features': [2, 3],
             'min_samples_leaf': [3, 4, 5],
             'min_samples_split': [8, 10, 12],
             'n_estimators': [100, 200, 300, 1000]   
         }

    ]
    
    grids = {}
    for model_info, params in zip(models, param_choices):
        print(model_info)
        name, model = model_info
        grid = GridSearchCV(model(), params, scoring = scoring, n_jobs = -1,)
        grid.fit(X, y)
        s = f"{name}: best score: {grid.best_score_}"
        print(s)
        grids[name] = grid
    return grids


def get_features(features_to_exclude, df):
    return [x for x in df.columns if x not in features_to_exclude]
    
def get_target(target_to_include, df):
    return [x for x in df.columns if x in target_to_include]

def graph_predVSact(X, y, mod, xlim, ylim, xlab, ylab):
    X_scaled = scaler.fit_transform(X)
    predicted = grids[mod].best_estimator_.predict(X_scaled)
    fig, ax = plt.subplots()
    ax.scatter(y, predicted, edgecolors=(0, 0, 0))
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlim([0, xlim])
    ax.set_ylim([0, ylim])
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    fig = ax.get_figure()
    fig.savefig('/Users/ajdavis/Desktop/GitHub/A-Cannalytical-Mind/Dankstimate/Images/predict_actual.jpg', bbox_inches = 'tight')
    return plt.show()
    
