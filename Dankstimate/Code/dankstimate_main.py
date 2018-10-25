#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 15:24:05 2018

@author: ajdavis
"""

# =============================================================================
# Import relevant libraries helper functions
# =============================================================================

import sys
sys.path.insert(0, '/Users/ajdavis/Desktop/GitHub/A-Cannalytical-Mind/Dankstimate/Code')

from dankstimate_helper import *

# =============================================================================
# Data Processing
# =============================================================================
    
## Get list of dispensary names
url = 'https://raw.githubusercontent.com/grammakov/USA-cities-and-states/master/us_cities_states_counties.csv'
cities = pd.read_csv(url, sep = '|', error_bad_lines=False)
legal_states = ['California', 'Colorado', 'Nevada', 'Oregon', 'Washington']

legal_cities = cities[cities['State full'].isin(legal_states)]
legal_cities = list(legal_cities['City alias'] + ',' + ' ' + legal_cities['State short'])

# Get city geocodes
gcs = get_gcs(legal_cities) 

# Create a list of dispensary names to pull menu data from   
disp_names = list(itertools.chain.from_iterable(list(map(get_disp_names, gcs))))

## Loop through dispensary menus and create aggregated data set
weed_df = pd.concat(map(menu_to_df, disp_names))

## Munge data
weed_df_munge = munge_df(weed_df)

## Average prices by location

# prices by zip
price_zip = (weed_df_munge
.groupby(['zip'])
.agg({
     'price': ['mean', 'count']
 }))
price_zip.columns = price_zip.columns.get_level_values(0)
price_zip.columns = ['price', 'count']
price_zip = (price_zip.loc[price_zip['count'] > 1000])
price_zip.to_csv('/Users/ajdavis/Desktop/GitHub/A-Cannalytical-Mind/Dankstimate/Images/price_zip.csv')

# prices by region
price_region = (weed_df_munge
.groupby(['region'])
.agg({
     'price': ['mean', 'count']
 }))

price_region.columns = price_region.columns.get_level_values(0)
price_region.columns = ['price', 'count']
price_region = (price_region.loc[price_region['count'] > 1000])
price_region.to_csv('/Users/ajdavis/Desktop/GitHub/A-Cannalytical-Mind/Dankstimate/Images/price_region.csv')

price_region.index.name = ''

ax = price_region['price'].sort_values().tail(15).plot.barh(title='$/Gram by Region')
fig = ax.get_figure()
fig.savefig('/Users/ajdavis/Desktop/GitHub/A-Cannalytical-Mind/Dankstimate/Images/high_price.jpg', bbox_inches = 'tight')

ax = price_region['price'].sort_values(ascending=False).tail(15).plot.barh()
fig = ax.get_figure()
fig.savefig('/Users/ajdavis/Desktop/GitHub/A-Cannalytical-Mind/Dankstimate/Images/low_price.jpg', bbox_inches = 'tight')

ax = weed_df_munge['category_name'].value_counts().sort_values().plot.pie(title='Frequency of Sub Types', autopct='%1.0f%%')
ax.set_ylabel('')
fig = ax.get_figure()
fig.savefig('/Users/ajdavis/Desktop/GitHub/A-Cannalytical-Mind/Dankstimate/Images/subtype.jpg', bbox_inches = 'tight')

ax = weed_df_munge['dname'].value_counts().sort_values().tail(15).plot.barh(title='Dispensaries with the Most Data')
fig = ax.get_figure()
fig.savefig('/Users/ajdavis/Desktop/GitHub/A-Cannalytical-Mind/Dankstimate/Images/dispensary.jpg', bbox_inches = 'tight')

# =============================================================================
# Feature Engineering
# =============================================================================

# Create top strains feature
weed_df_munge_feat = ts_feature(weed_df_munge)

# average price per top strain
price_ts = (weed_df_munge_feat.loc[weed_df_munge_feat['Match Score'] > 90])
price_ts = (price_ts
.groupby(['Matched Strain'])
.agg({
     'price': ['mean', 'count']
 }))

## Get ancillary Strain Data
weed_df_munge_feat_con = content_feature(weed_df_munge_feat)

# Only keep observations where there was 100% match
weed_df_munge_feat_con = weed_df_munge_feat_con.loc[weed_df_munge_feat_con['Match Score'] == 100]

## logs, polynomials, interaction terms

# Check to see if target is normally distributed
weed_df_munge_feat_con['price'].hist()

# Log Transformations
weed_df_munge_feat_con['ln_price'] = np.log(weed_df_munge_feat_con['price']) 
weed_df_munge_feat_con['ln_price'].hist()


# Polnomial Transformations
weed_df_munge_feat_con['THC2'] = weed_df_munge_feat_con['THC']**2
weed_df_munge_feat_con['THC3'] = weed_df_munge_feat_con['THC']**3
weed_df_munge_feat_con['THC4'] = weed_df_munge_feat_con['THC']**4


# Interactions
weed_df_munge_feat_con['THC:rating'] = weed_df_munge_feat_con['THC']*weed_df_munge_feat_con['rating']

# Create categorical dummies

dum1 = pd.get_dummies(weed_df_munge_feat_con['category_name'], drop_first=True)
dum2 = pd.get_dummies(weed_df_munge_feat_con['license_type'], drop_first=True)
dum3 = pd.get_dummies(weed_df_munge_feat_con['region'], drop_first=True)

a_df = pd.concat([weed_df_munge_feat_con, dum1], axis=1)
a_df = pd.concat([a_df, dum2], axis=1)
a_df = pd.concat([a_df, dum3], axis=1)


# =============================================================================
# Modeling 
# =============================================================================

## Specifiy Models
# Features (add those that should be excluded)
mf1 = [
       'price', 'ln_price', 'THC', 'THC2', 'THC3', 'THC4', 'THC:rating', 
       'body', 'category_name', 'dname', 'license_type', 'listing_name',
       'name', 'prices', 'region', 'zip', 'gram', 'Strain Name', 'MatchedName',
       'Match', 'Matched Strain', '_Index', 'Match Score', 'MatchNew'
       ]
mf2 = ['price', 'ln_price', 'THC', 'THC2', 'THC3', 'THC4', 'THC:rating', 
       'body', 'category_name', 'dname', 'license_type', 'listing_name',
       'name', 'prices', 'region', 'zip', 'gram', 'Strain Name', 'MatchedName',
       'Match', 'Matched Strain', '_Index', 'Match Score', 'MatchNew'
       ]
mf3 = ['price', 'ln_price', 'THC', 'THC3', 'THC4', 'THC:rating', 
       'body', 'category_name', 'dname', 'license_type', 'listing_name',
       'name', 'prices', 'region', 'zip', 'gram', 'Strain Name', 'MatchedName',
       'Match', 'Matched Strain', '_Index', 'Match Score', 'MatchNew'
       ]

# Targets
mt1 = ['price']
mt2 = ['ln_price']
mt3 = ['price']

# Set X and y 
features = get_features(mf1, a_df)
target = get_target(mt1, a_df)

design = target + features
analysis_df = a_df[design]

analysis_df = analysis_df.dropna()

# Clean categorical dummy column names for patsy (need to write helper function)
analysis_df.columns = analysis_df.columns.str.replace(" - ", "_")
analysis_df.columns = analysis_df.columns.str.replace("-", "_")
analysis_df.columns = analysis_df.columns.str.replace(" ", "_")
analysis_df.columns = analysis_df.columns.str.replace("'", "")
analysis_df.columns = analysis_df.columns.str.replace(".", "")
analysis_df.columns = analysis_df.columns.str.replace("|", "_")
analysis_df.columns = analysis_df.columns.str.replace("&", "_")
analysis_df.columns = analysis_df.columns.str.replace(")", "")
analysis_df.columns = analysis_df.columns.str.replace("(", "")
analysis_df.columns = analysis_df.columns.str.replace("/", "")
analysis_df.columns = analysis_df.columns.str.replace(",", "")
analysis_df.columns = analysis_df.columns.str.strip()

# Create formula for patsy
my_formula = analysis_df.columns[0] + " ~ " + \
" + ".join(list(analysis_df.columns[1:len(a_df.columns)])) + ' - 1'

# Create your feature matrix (X) and target vector (y)
y, X = patsy.dmatrices(my_formula, data=analysis_df, return_type="dataframe")

# Look at OLS before testing all models
sm.OLS(y, sm.add_constant(X)).fit().summary()

## Test/Train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

## Normalization
scaler = preprocessing.StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

## Check all models on training data

grids = score_ht_models(X_train_scale, y_train, scoring = 'r2')

# It's clear that randomForest is the best performing model
# Now let's check for overfitting
grids['randomForest'].best_estimator_.score(X_train_scale, y_train)
grids['randomForest'].best_estimator_.score(X_test_scale, y_test)

X_scale = scaler.fit_transform(X)
grids['randomForest'].best_estimator_.score(X_scale, y)



# Score RMSE
predicted = grids['randomForest'].best_estimator_.predict(X_train_scale)
sqrt(mean_squared_error(y_train, predicted))

predicted = grids['randomForest'].best_estimator_.predict(X_test_scale)
sqrt(mean_squared_error(y_test, predicted))


## Predicted vs. Actual
graph_predVSact(X, y, 'randomForest', 20, 20, 'Actual $/gram', 'Predicted $/gram')



    




