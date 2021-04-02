# -*- coding: utf-8 -*-
"""
Model building for the NYC Airbnb dataset
@author: Alexander Ngo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

df = pd.read_csv('airbnb_cleaned.csv')

# choose relevant columns
df_model = df[['neighbourhood_group', 'neighbourhood', 'latitude', 'longitude',
               'room_type', 'minimum_nights', 'number_of_reviews',
               'reviews_per_month', 'calculated_host_listings_count',
               'availability_365', 'days_since_last_review', 'reviewed_yn',
               'name_len', 'price_log']]

numerical_cols = ['latitude', 'longitude', 'minimum_nights', 'number_of_reviews',
             'reviews_per_month', 'calculated_host_listings_count',
             'availability_365', 'days_since_last_review', 'name_len']
categorical_cols = ['neighbourhood_group', 'neighbourhood', 'room_type',
             'reviewed_yn']

# create pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

numerical_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
    #('labelenc', OrdinalEncoder(handle_unknown='ignore'))
    ])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
        ])

# train test split
from sklearn.model_selection import train_test_split

X = df_model.drop('price_log', axis = 1)
y = df_model.price_log.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=1)

# model training
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

# =============================================================================
# #first with only default parameters
# #random forest
# rf = RandomForestRegressor()
# rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
#                               ('model', rf)
#                               ])
# rf_pipeline.fit(X_train, y_train)
# y_pred=rf_pipeline.predict(X_test)
# print('MAE: %f'% mean_absolute_error(y_test, y_pred))
# print('Rsquared: %f'% r2_score(y_test, y_pred))
# #MAE: 0.310711
# #Rsquared: 0.589794
# 
# # multiple linear regression
# lr = LinearRegression()
# lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
#                               ('model', lr)
#                               ])
# lr_pipeline.fit(X_train, y_train)
# y_pred=lr_pipeline.predict(X_test)
# 
# print('MAE: %f'% mean_absolute_error(y_test, y_pred))
# print('Rsquared: %f'% r2_score(y_test, y_pred))
# #MAE: 0.337080
# #Rsquared: 0.539544
# 
# # lasso regression
# las = Lasso()
# las_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
#                               ('model', las)
#                               ])
# las_pipeline.fit(X_train, y_train)
# y_pred=las_pipeline.predict(X_test)
# 
# print('MAE: %f'% mean_absolute_error(y_test, y_pred))
# print('Rsquared: %f'% r2_score(y_test, y_pred))
# #MAE: 0.538743
# #Rsquared: 0.020389
# 
# # ridge regression
# rr = Ridge()
# rr_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
#                               ('model', rr)
#                               ])
# rr_pipeline.fit(X_train, y_train)
# y_pred=rr_pipeline.predict(X_test)
# print('MAE: %f'% mean_absolute_error(y_test, y_pred))
# print('Rsquared: %f'% r2_score(y_test, y_pred))
# #MAE: 0.343149
# #Rsquared: 0.526198
# 
# =============================================================================

# tune models via gridsearch
from sklearn.model_selection import GridSearchCV

def random_forest_tuning(x,y, cv=5):
    
    start_process_time = time.process_time()
    start_time = time.time()
    
    rf_model = RandomForestRegressor(random_state=1)
    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', rf_model)
                              ])
    
    parameters = {'model__n_estimators':[1500,2500], 
                  'model__max_features':['auto', 'sqrt', 'log2'],
                  'model__max_depth':[10,12]}
    
    #Use k-fold cross validation and mean squared error as a scoring metric
    rf_grid_search = GridSearchCV(estimator=rf_pipeline,
                                  param_grid=parameters,
                                  scoring='neg_mean_squared_error',
                                  cv=cv,n_jobs=-1)
    
    rf_grid_search.fit(x,y)
    best_params = rf_grid_search.best_params_
    best_score = rf_grid_search.best_score_
    print('best params: ', end = ' ')
    print(best_params)
    print('best score: %f'% best_score)
    
    print(time.process_time() - start_process_time, end = ' '), print('cpu seconds')
    print(time.time() - start_time, end = ' '), print('real seconds')

#random_forest_tuning(X_train, y_train)

#best params:  {'model__max_depth': 12, 'model__max_features': 'auto', 'model__n_estimators': 2500}
#best score: -0.190108
#1700.140625 cpu seconds
#5677.976025104523 real seconds

def linear_reg_tuning(x,y, cv =5):
    
    start_process_time = time.process_time()
    start_time = time.time()
    
    lr_model = LinearRegression()
    lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', lr_model)
                              ])
    
    parameters = {'model__fit_intercept':[True,False],
                  'model__normalize':[True,False],
                  'model__copy_X':[True, False]}
    
    #Use k-fold cross validation and mean squared error as a scoring metric
    lr_grid_search = GridSearchCV(estimator=lr_pipeline,
                                  param_grid=parameters,
                                  scoring='neg_mean_squared_error',
                                  cv=cv,n_jobs=-1)
    
    lr_grid_search.fit(x,y)
    best_params = lr_grid_search.best_params_
    best_score = lr_grid_search.best_score_
    print('best params: ', end = ' ')
    print(best_params)
    print('best score: %f'% best_score)
    
    print(time.process_time() - start_process_time, end = ' '), print('cpu seconds')
    print(time.time() - start_time, end = ' '), print('real seconds')
    
#linear_reg_tuning(X_train, y_train)

#best params:  {'model__copy_X': True, 'model__fit_intercept': True, 'model__normalize': False}
#best score: -0.218543
#4.34375 cpu seconds
#18.422568798065186 real seconds

def lasso_tuning(x,y, cv=5):
    
    start_process_time = time.process_time()
    start_time = time.time()
    
    las_model = Lasso()
    las_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', las_model)
                              ])
    
    parameters = {'model__alpha':[1,0.1,0.01,0.001,0.0001], 
                  'model__normalize':['True', 'False'],}
    
    #Use k-fold cross validation and mean squared error as a scoring metric
    las_grid_search = GridSearchCV(estimator=las_pipeline,
                                  param_grid=parameters,
                                  scoring='neg_mean_squared_error',
                                  cv=cv,n_jobs=-1)
    
    las_grid_search.fit(x,y)
    best_params = las_grid_search.best_params_
    best_score = las_grid_search.best_score_
    print('best params: ', end = ' ')
    print(best_params)
    print('best score: %f'% best_score)
    
    print(time.process_time() - start_process_time, end = ' '), print('cpu seconds')
    print(time.time() - start_time, end = ' '), print('real seconds')

#lasso_tuning(X_train, y_train)
#best params:  {'model__alpha': 0.0001, 'model__normalize': 'True'}
#best score: -0.231683
#1.828125 cpu seconds
#3.5050370693206787 real seconds

def ridge_tuning(x,y, cv=5):
    
    start_process_time = time.process_time()
    start_time = time.time()
    
    rr_model = Lasso()
    rr_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', rr_model)
                              ])
    
    parameters = {'model__alpha':[1,0.1,0.01,0.001,0.0001], 
                  'model__normalize':['True', 'False'],}
    
    #Use k-fold cross validation and mean squared error as a scoring metric
    rr_grid_search = GridSearchCV(estimator=rr_pipeline,
                                  param_grid=parameters,
                                  scoring='neg_mean_squared_error',
                                  cv=cv,n_jobs=-1)
    
    rr_grid_search.fit(x,y)
    best_params = rr_grid_search.best_params_
    best_score = rr_grid_search.best_score_
    print('best params: ', end = ' ')
    print(best_params)
    print('best score: %f'% best_score)
    
    print(time.process_time() - start_process_time, end = ' '), print('cpu seconds')
    print(time.time() - start_time, end = ' '), print('real seconds')

#ridge_tuning(X_train, y_train)
#best params:  {'model__alpha': 0.0001, 'model__normalize': 'True'}
#best score: -0.231683
#1.9375 cpu seconds
#4.816668272018433 real seconds

# plot model RMeanSquaredError
rmse = {"Model":["RF","LR","Lasso", "RR"],"RMSE":[0.190108,0.218543,0.231683,0.231683]}
rmse = pd.DataFrame(rmse)

sns.catplot(x="Model", y="RMSE", linestyles=["-"], kind="point", data=rmse)