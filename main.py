#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:55:45 2023

@author: aditidadariya
"""

import utility as util
import os
import sys
import numpy as np
from imblearn.over_sampling import SMOTE

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
    util.WriteInTextFile("The dataset has Rows {} and Columns {} ".format(df_data.shape[0], df_data.shape[1]))
    print("The dataset has Rows {} and Columns {} ".format(df_data.shape[0], df_data.shape[1]))
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
    
# ========================= EXPLORATORY DATA ANALYSIS ===============================

# Calling FindUniqueValues function to find the unique values in dataset
util.FindUniqueValues(df_data)

# Calling RelationshipAgeWise function to find a relationship between Age with Job and Education
util.RelationshipAgeWise(df_data,'AGE','JOB','EDUCATION')

# ========================= TRANSFORM CATEGORICAL VALUES ============================

# Transform the categorical values into numerical values to 
df_data = util.LabelEncoding(df_data)

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

# Add a new line here
util.Newline()
    
# Observation: By the output it is clear that the data is imbalance. 
# SMOTE technique is used to balance the data in further steps

# ============================ DIVIDE THE DATASET ===================================

# Divide the dataset into Features and Target
X = df_data.drop('Y', axis=1)
y = df_data['Y']
y = y.to_frame()

# ============================ BALANCE THE DATASET ==================================

# Balance the dataset using SMOTE technique
smote = SMOTE()
X_features, y_target = smote.fit_resample(X, y)

# Find the count of each target category
print('The count of each category in the target variable is:')
print(y_target.value_counts())

# Add a new line here
util.Newline()

print('The shape of Features is now {} and Target is {}'.format(X_features.shape, y_target.shape))


