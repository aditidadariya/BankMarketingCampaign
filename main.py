#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:55:45 2023

@author: aditidadariya
"""

import utility as util
import yaml

import os
import sys
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot

import warnings
warnings.filterwarnings("ignore")

# ========================== CREATE YAML FILE ========================================

# Created a new empty yaml file to store the global parameters. 
#util.CreateYAMLFle()           # Commeted this line as the yaml file has been created initially and do not need to be created again

# Load config.yaml file and store it as a dictionary object
param = util.LoadYAMLFile("config.yaml")

# ========================== CLEAR LOG FILE ==========================================

util.ClearTextFile(param['logfilename'])

# ======================= READ CSV FILE USING PANDAS read_csv ========================

# Read the file using PANDAS read_csv method from local location and store it in a dataframe
df_data = util.PD_read_csv(param['filename'], param['delimiter'], param['columns'])

if len(df_data) > 0:
    # Write data in text file
    util.WriteInTextFile("The dataset has {} Rows and {} Columns.".format(df_data.shape[0], df_data.shape[1]))
    print("The dataset has {} Rows and {} Columns.".format(df_data.shape[0], df_data.shape[1]))
else:
    print("Error while reading the dataset")
    sys.exit(1)
    
# Add a new line here
util.Newline()

# Print the dataframe
print(df_data.head())

# Add a new line here
util.Newline()

# Print the info
print(df_data.info())

# Add a new line here
util.Newline()

# Print the statistics
print(df_data.describe())

# Add a new line here
util.Newline()

# ============================ FIND THE MISSING VALUES ================================

# Print the missing values
print('Missing values in the dataset:')
print(df_data.isnull().sum())

# Add a new line here
util.Newline()

# Variables consisting "unknown" values
columns_with_unknown = df_data.columns[df_data.isin(['unknown']).any()]
print('Variables having "unknown" value {} ' .format(columns_with_unknown.values))

# Add a new line here
util.Newline()

# Find the number the "unknown" in each variable
unknown_counts = df_data[df_data == 'unknown'].count()
print('Variables with "unknown" counts:')
print(unknown_counts)

# Add a new line here
util.Newline()

# Replace "unknown" to NaN values
df_data.replace('unknown', np.nan, inplace=True)

# Verify if "unknown" values still exists
columns_unknown = df_data.columns[df_data.isin(['unknown']).any()]

if len(columns_unknown) == 0:
    print('There are no "unknown" values for any variable now')
else:
    print('The variables having "unknown" values are {}'.format(columns_unknown))

# Add a new line here
util.Newline()

# Print the missing values
print('Missing values in the dataset: \r\n{}'.format(df_data.isnull().sum()))

# Add a new line here
util.Newline()

# Find the columns having null values
columns_with_null = df_data.columns[df_data.isnull().any()].tolist()

# ReplaceNullValues function is called to replace null values with its most frequently used value by SimpleImputer method
df_data = util.ReplaceNullValues(df_data, columns_with_null)

# Convert the columns names to upper case
df_data.columns = df_data.columns.str.upper()

# Verify the data types of each variables and correct it
df_data = util.CorrectDTypes(df_data)

# Find the columns having null values
columns_with_null = df_data.columns[df_data.isnull().any()].tolist()

if len(columns_with_null) > 0:
    print('There are still few null values available in the data')
else:
    print('All the null values are substituted by the most frequently used value')

# Add a new line here
util.Newline()

# ========================= EXPLORATORY DATA ANALYSIS ===============================

print('Finding the correlation between numeric variables by heat map')
# Calling PlotHeatMap function to understand the correlation between variables
util.PlotHeatMap(df_data)

# Add a new line here
util.Newline()

print("Finding a relation between highly correlated variables by pair plot")
# Calling PairPlot to plot a relation between highly correlation variables
util.Pairplot(df_data[['EMP.VAR.RATE','EURIBOR3M','NR.EMPLOYED']])

# Add a new line here
util.Newline()

print('Finding the unique values of each variables')
# Calling FindUniqueValues function to find the unique values in dataset
util.FindUniqueValues(df_data)

# Add a new line here
util.Newline()

print('Finding the relation of categorical variables with target variable')
# Navigate to all columns
for i, col in enumerate(df_data.columns):
    # Get the datatype of column
    datatype = df_data[col].dtype
    if datatype == "object" and col != "Y" :
        # Calling BarPlot function to understand the relation of categorical variable with Target Y variable
        util.BarPlot(df_data, col, 'Y')

# ========================= TRANSFORM CATEGORICAL VALUES ============================

# Calling EncodingVariable to encode the categorical variable to numeric
# Binary Categorical variables are converted using Label Encoder
# Multi Categorical variables are converted using One-Hot Encoder
df_data = util.EncodingVariable(df_data)

# Add a new line here
util.Newline()

# ====================== FIND THE OUTLIERS USING BOX PLOT ===========================

# Plot a boxplot to find the outliers in numeric variables
util.BoxPlotNumericColumns(df_data)

# Add a new line here
util.Newline()

# ========================= FIND THE OUTLIERS USING IQR =============================

print("Outlier detection by Interquantile Range method:")

# Find the outliers in dataset by Interquantile Range technique
util.OutliersIQR(df_data)

# Add a new line here
util.Newline()

# ========================= FIND THE OUTLIERS USING ZScore ===========================

print("Outlier detection by ZScore method:")

# Find the outliers in dataset by ZScore technique and impute the outliers using QuantileTransformer technique
util.OutliersZScore(df_data)

# Add a new line here
util.Newline()

# ========================= FIND IF DATASET IS IMBALANCE ============================

# Find the count of each target category
print('The count of each category in the target variable is:')
print(df_data['Y'].value_counts())

# view the percentage distribution of target column
print('The percentage distribution of target class is: ')
print(df_data['Y'].value_counts()/float(len(df_data)))

# Add a new line here
util.Newline()
    
# Observation: By the output it is clear that the percentage of observations of the 
# class label 0 and 1 is 88.7% and 11.3%. So, this is a class imbalanced problem. 
# This will be addresses in subsequent steps

# ============================ DIVIDE THE DATASET ===================================

# Divide the dataset into Features and Target
X = df_data.drop('Y', axis=1)
Y = df_data['Y']
Y = Y.to_frame()

# ====================== HOLD OUT MOETHOD OF DATA SPLIT =============================

# Clear lists
param["models"].clear()
param["names"].clear()
param["results"].clear()
param["basicscore"].clear()

# The training set and test set has been splited below using the feature and target dataframes
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=None)

# ============================= BASELINE MODEL =======================================

# Baseline model
param["models"].append(('SVC', SVC()))
baseacc = util.BasicModel(param["models"], X_train, Y_train, X_test, Y_test)

# ============================== CREATE MODEL ========================================

# Clear lists
param["models"].clear()
param["names"].clear()
param["results"].clear()
param["basicscore"].clear()

# Calling CreateModels function
modelobjects = util.CreateModels(param["models"]) #, param["n_estimators"])

# ============================== BASIC MODEL ========================================

# Calling BasicModel function
basicscores = util.BasicModel(models, X_train, Y_train, X_test, Y_test)
# Create a dataframe to store accuracy
dfbasicscore = pd.DataFrame(basicscores) 
print("Accuracy of each model:")   
print(dfbasicscore)

# Add a new line here
util.Newline()

# ===================== OPTIMIZING THE MODEL WITH RANDOM STATE ======================

# Calling BuildModelRS to evaluate the models with random states and return the accuracy
scores = util.BuildModelRS(modelobjects,param["randstate"], X, Y)
# Create a dataframe to store accuracy
dfrsscore = pd.DataFrame(scores)
print("Accuracy of each model after optimizing model with random states:") 
print(dfrsscore)

# Add a new line here
util.Newline()

# Calling FindBestRandomState function to find the random state best performed for each model
dfrsmodels = util.FindBestRandomState(dfrsscore)
print("The best performed random state for each model:")
print(dfrsmodels)

# Add a new line here
util.Newline()

# ============================ BALANCE THE DATASET ==================================

# Calling BalanceData function to balance the dataset using SMOTE and RandomUnderSampler
X_Balanced, Y_Balanced = util.BalanceData(X,Y)
# Find the count of each target category
print('The count of each category in the target variable after balancing the data is:')
print(Y_Balanced.value_counts())

# Add a new line here
util.Newline()

print('The shape of Features is now {} and Target is {}'.format(X_Balanced.shape, Y_Balanced.shape))

# Add a new line here
util.Newline()

# ========== OPTIMIZING THE MODEL ON BALANCED DATA AND CROSS VALIDATION =============

# Calling BuildModelBalCV function to get the accuracy, cross validation results and names of models used
score, results, names = util.BuildModelBalCV(modelobjects, X_Balanced, Y_Balanced)
# Create a dataframe to store accuracy
dfcrossval = pd.DataFrame(score)    
print("Mean and Standard deviation of each model on balanced data with cross validation:") 
print(dfcrossval)

# Add a new line here
util.Newline()

# Compare Algorithms and plot them in boxplot
pyplot.clf()
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

# Add a new line here
util.Newline()

# ================ IDENTIFY THE BEST HYPERPARAMETERS OF EACH MODEL ==================

# LR hyperparameter tuning
lr_para = util.BestHyperparameters('LR', modelobjects, dfrsmodels, param['lr_grid'], X_Balanced, Y_Balanced)
#print(lr_para.best_params_['solver'], lr_para.best_params_['penalty'])

# LDA hyperparameters tuning
lda_para = util.BestHyperparameters('LDA', modelobjects, dfrsmodels, param['lda_grid'], X_Balanced, Y_Balanced)
#print(lda_para.best_params_['solver'])

# RF hyperparameters tuning
rf_para = util.BestHyperparameters('RFC', modelobjects, dfrsmodels, param['rf_grid'], X_Balanced, Y_Balanced)
#print(rf_para.best_params_['max_depth'], rf_para.best_params_['n_estimators'],rf_para.best_params_['max_features'])

# ADABoost hyperparameter tuning
adab_para = util.BestHyperparameters('ADAB', modelobjects, dfrsmodels, param['adab_grid'], X_Balanced, Y_Balanced)
#print(adab_para.best_params_['n_estimators'], adab_para.best_params_['learning_rate'])

# XGBoost hyperparameter tuning
xgb_para = util.BestHyperparameters('XGB', modelobjects, dfrsmodels, param['xgb_grid'], X_Balanced, Y_Balanced)
#print(xgb_para.best_params_['max_depth'], xgb_para.best_params_['min_child_weight']) #, xgb_para.best_params_['lambda'])

# HGBC hyperparameter tuning
hgbc_para = util.BestHyperparameters('HGBC', modelobjects, dfrsmodels, param['hgbc_grid'], X_Balanced, Y_Balanced)
#print(hgbc_para.best_params_['max_bins'], hgbc_para.best_params_['max_iter'])

# Store all the best parameters into a list
parameters = [lr_para.best_params_, lda_para.best_params_, 
              rf_para.best_params_, adab_para.best_params_, 
              xgb_para.best_params_, hgbc_para.best_params_]

# == BUILD FINAL MODEL ON BALANCED DATA WITH HYPERPARAMETERS, KFOLD AND CROSS VALIDATION ==

final_models = util.CreateFinalModels(param["final_models"], parameters)
# Calling BuildFinalModel to get the accuracy after tuning
score, names, final_results, model = util.BuildFinalModel(final_models, dfrsscore, X_Balanced, Y_Balanced)
# Create a dataframe to store accuracy
dffinalscore = pd.DataFrame(score) 
print(dffinalscore)

# Find the best model performed
max_mean = dffinalscore['Accuracy - Mean'].max()
max_mean_indx = dffinalscore['Accuracy - Mean'].idxmax()
model_name = dffinalscore.loc[max_mean_indx, 'Model Name']

# Give a new line to clearly format the output
util.Newline()

print('%s has the max accuracy as %f' % (model_name, max_mean))

# Give a new line to clearly format the output
util.Newline()

# Compare Algorithms and plot them in boxplot
pyplot.clf()
pyplot.boxplot(final_results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

############################# BUILD BEST PERFORMED MODEL ############################


# Clear lists
param["models"].clear()
param["names"].clear()
param["results"].clear()
param["basicscore"].clear()
param["final_models"].clear()

# Define models with the best hyperparameters found earlier
if model_name == "LR":
    param['models'].append(('LR', LogisticRegression(penalty = parameters[0]['penalty'], solver = parameters[0]['solver'])))
elif model_name == "LDA":
    param['models'].append(('LDA', LinearDiscriminantAnalysis(solver = parameters[1]['solver'])))
elif model_name == "RPC":
    param['models'].append(('RFC', RandomForestClassifier(max_depth = parameters[2]['max_depth'], max_features = parameters[2]['max_features'], n_estimators = parameters[2]['n_estimators'])))
elif model_name == "ADAB":
    base_estimator = DecisionTreeClassifier(max_depth=1)
    param['models'].append(('ADAB', AdaBoostClassifier(base_estimator=base_estimator, learning_rate = parameters[3]['learning_rate'], n_estimators = parameters[3]['n_estimators'])))
elif model_name == "XGB":
    param['models'].append(('XGB', XGBClassifier(max_depth = parameters[4]['max_depth'], min_child_weight = parameters[4]['min_child_weight'])))
elif model_name == "HGBC":
    param['models'].append(('HGBC', HistGradientBoostingClassifier(max_bins = parameters[5]['max_bins'], max_iter = parameters[5]['max_iter'])))

# Calling BuildFinalModel to get the prediction after tuning
score, name, final_results, model = util.BuildFinalModel(param['models'], dfrsscore, X_Balanced, Y_Balanced)

################################## MODEL.PKL FILE ##################################

# Dump the model.pkl file
import pickle
pickle.dump(model, open('model.pkl', 'wb'))
