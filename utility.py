#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:56:03 2023

@author: aditidadariya
"""

import yaml
import os
import pandas as pd
import numpy as np
from datetime import datetime

from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer
from scipy.stats import mode

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# Import RandomUnderSampler, SMOTE, Pipeline to balance the dataset
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics 
from sklearn.model_selection import train_test_split
# Import confusion_matrix function
from sklearn.metrics import confusion_matrix
# Import classification_report function
from sklearn.metrics import classification_report
# Import ConfusionMatrixDisplay function to plot confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.model_selection import RepeatedStratifiedKFold
# Import StratifiedKFold function
from sklearn.model_selection import StratifiedKFold
# Import GridSearchCV function
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

#==============================================================================

# Newline gives a new line
def Newline():
    print("\r\n")

# CreateYAMLFle function is defined to create a new yaml file
def CreateYAMLFle():
    # Define the data to hold empty value
    data = {}
    # Specify the file path and name for the YAML file
    file_path = 'config.yaml'
    
    # Write the data to the YAML file
    with open(file_path, 'w') as file:
        yaml.dump(data, file)
    
    print(f"YAML file '{file_path}' has been created.")

# LoadYAMLFile function is defined to read the parameters defined in config.yaml file
def LoadYAMLFile(yamlfilename):
    # Specify the file path and name of the YAML file
    file_path = yamlfilename
    
    # Load the contents of the YAML file
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    
    return data

# RemoveWhiteSpace finction is defined to remove the white spaces from column name
def RemoveWhiteSpace(df):
    # Remove white spaces from column names
    df.rename(columns=lambda x: x.replace(" ", ""), inplace=True)
    return df

# RemoveSpecialChar function is defined to remove the special character
def RemoveSpecialChar(df):
    pattern = r'[^\w\s]'

    # Iterate over each column in the DataFrame
    for column in df.columns:
        # Apply pattern matching to identify rows with special characters
        df[column].replace(pattern, '', regex=True)
        
    return df

# ClearTextFile function is defined to clear the text file in the begining
def ClearTextFile(filename):
    # Open the file in write mode and truncate it
    with open(filename, 'w') as file:
        file.truncate(0)
    
# WriteInTextFile function is defined to add content in text file
def WriteInTextFile(stmt):
    # Open the file in write mode
    with open('textfile.txt', 'a') as file:
        # Write the data to the file
        file.write(str(datetime.now()) + " ")
        file.write(stmt)
        file.write('\n')

# PD_read_csv function reads the dataset csv file using Pandas and store it in a dataframe
def PD_read_csv(filename, delimiter, columnnames):
    # absolute_path and location_of_file defined to locate the file
    absolute_path = os.path.dirname(__file__)
    location_of_file = os.path.join(absolute_path, filename)
    
    # Records the Start time
    Start_datetime = datetime.now()
        
    # Read the data file from specified location using Pandas
    df = pd.read_csv(location_of_file, delimiter=delimiter)
    
    # Record the End time
    End_datetime = datetime.now()
    
    # Calculate the time Pandas took to read the file
    dt = End_datetime - Start_datetime
    WriteInTextFile("Total time taken for Pandas to read the file is: {}".format(dt))
    # Compare DataFrame column names with the columns fetched from config.yaml file
    matching_columns = [col for col in df.columns if col in columnnames]
    
    if len(matching_columns) == len(df.columns) and list(matching_columns) == list(df.columns):
        return df
    else:
        return print("Columns are not same as specified in configurations")
   
# Replace Null values with the SimpleImputer method
def ReplaceNullValues(df, columns):
    # Initialize the SimpleImputer
    imputer = SimpleImputer(strategy='most_frequent', missing_values=np.nan)

    # Apply the imputer to the columns with null values
    df[columns] = imputer.fit_transform(df[columns])
    
    return df

# Find if any column has multiple data types
def CorrectDTypes(df):
    # Iterate over each column and check for multiple data types
    for column in df.columns:
        unique_dtypes = df[column].apply(lambda x: type(x)).nunique()
        if unique_dtypes > 1:
            print(f"Column '{column}' has multiple data types.")
            column_dtypes = df[column].apply(lambda x: type(x)).unique()
            print(f"Data types in '{column}': {column_dtypes}")
            # Convert job to string (str)
            df[column] = df[column].astype(str)
    return df

# ImputeOutliers function is defind to impute outliers with mode using QuantileTransformer
def ImputeOutliers(z_scores, threshold, df, column):
    # Calculate the mode of the column
    mode_value = mode(df[column])[0][0]

    # Create a copy of the column for imputation
    impute_column = df[column].copy()

    # Replace outliers with mode value
    impute_column[abs(z_scores) > threshold] = mode_value

    # Fit and transform the impute_column using QuantileTransformer
    quantile_transformer = QuantileTransformer(output_distribution='normal')
    imputed_values = quantile_transformer.fit_transform(impute_column.values.reshape(-1, 1))

    # Assign the imputed values back to the original column
    df[column] = imputed_values

    # Verify the imputed values
    return df[column]

# OutliersIQR function is defined to find the outliers using InterQuantile Range method
def OutliersIQR(df):
    # Define the numerical columns in the dataframe
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

    # Loop through each numerical column
    for column in numerical_columns:
        # Calculate the IQR for the column
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        # Define the lower and upper bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Find the outliers in the column
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

        # Print the outliers for the column
        if not outliers.empty:
            print(f"{outliers.shape[0]} number of row are Outliers in column {column}")
    return df

# Find the outliers using ZScore technique
def OutliersZScore(df):
    # Iterate over each column in the DataFrame
    for column in df.columns:
        # Exclude the AGE column
        if column != 'AGE':
            # Calculate Z-scores for the column
            z_scores = zscore(df[column])

            # Find the indices of outliers using a threshold (e.g., Z-score > 3 or < -3)
            outliers_indices = (z_scores > 3) | (z_scores < -3)
            
            outlier_values = df.loc[outliers_indices]
            # Print the outliers for the column
            if not outlier_values.empty and outlier_values.shape[0] > 100 :
                print(f"{outlier_values.shape[0]} number of row are Outliers in column {column}")
                
            # Impute the outliers with QuantileTransformer
            quantile_transformer = QuantileTransformer()
            imputed_values = quantile_transformer.fit_transform(df[column].values.reshape(-1, 1))

            # Update the original DataFrame with imputed values for outliers
            #df_outliers_imputed.loc[outliers_indices, column] = imputed_values[outliers_indices].flatten()
            df.loc[outliers_indices, column] = imputed_values[outliers_indices].flatten()
    return df

# EncodingVariable function is defined to encode the categorical variables to numeric
def EncodingVariable(df):
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False)

    # Navigate to each column in dataframe
    for i, col in enumerate(df.columns):
        # Find the datatype of columns
        datatype = df[col].dtype

        if datatype == 'object':
            # Get the unique values in the column
            unique_values = df[col].unique()
            if len(unique_values) == 2:
                df[col] = label_encoder.fit_transform(df[col])
            else:
                df[col] = onehot_encoder.fit_transform(df[[col]])
    return df

# STDScaling function is defined to scale the data between mean 0 and std 1
def STDScaling(df):
    # Initialize the StandardScaler object
    scaler = StandardScaler()
    # Fit the scaler to the dataframe
    scaler.fit(df)
    # Transform the dataframe using the scaler
    df_standardized = scaler.transform(df)
    # Convert the transformed array back to a dataframe
    df_standardized = pd.DataFrame(df_standardized, columns=df.columns)
    return df_standardized

# BoxPlotNumericColumns function is defined to plot a Box plot
def BoxPlotNumericColumns(df):
    # Assuming df is your DataFrame
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

    # Plot box plots for numeric columns
    plt.figure(figsize=(12, 6))
    df[numeric_columns].boxplot()
    plt.title('Box Plot for int64 and float64 Columns')
    plt.xlabel('Variables')
    plt.ylabel('Values')
    plt.xticks(rotation=45)
    plt.show()
    
# FindUniqueValues function is defined to retrieve and plot the unique values in data
def FindUniqueValues(df):

    # Find unique values in each column
    for i, col in enumerate(df.columns):
        # Get the unique values in the column
        unique_values = df[col].unique()
        print("In {}, there are {} number of unique values".format(col.upper(), len(unique_values)))
        # Get the datatype of column
        datatype = df[col].dtype
        # Set the figure size
        #plt.figure(figsize=(10, 6))

        if datatype == "object":
            # Create the countplot
            sns.countplot(data=df, y=col)

            # Set labels and title
            plt.xlabel('COUNT')
            plt.ylabel(col.upper())
            plt.title('Countplot for {}'.format(col))

        elif datatype == 'int64' or datatype == 'float64':
            # Evaluate the mean and standard deviation of the column
            mean = df[col].mean()
            std = df[col].std()
            # Print the mean and standard deviation
            print("Mean: ", mean)
            print("Standard deviation:", std)
            
            # Count the occurrences of each unique value in the column
            value_counts = df[col].value_counts()
            # Create the bar plot
            plt.bar(value_counts.index, value_counts.values)

            # Add labels and title
            plt.xlabel(col.upper())
            plt.ylabel('COUNT')
            plt.title('Bar Plot for {}'.format(col))

        # Show the plot
        plt.show() 
        # Add a new line here
        Newline()

# PlotHeatMap function is defined to create Heat Map to understand the correlation
def PlotHeatMap(df):
    # Customizing the heatmap
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    subset_df = df[numeric_columns]
    correlation_matrix = subset_df.corr()
    # Print the column names
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, cmap='coolwarm', linewidths=0.5, annot=True)
    plt.title('Correlation Matrix')
    # Show the plot
    plt.show() 
    
    # Add a new line here
    Newline()

    # Convert the correlation matrix to dataframe
    corr_df = pd.DataFrame(correlation_matrix)
    # Iterate in correlation matrix and display the relation where correlation is greater than 0.50
    for i in range(len(corr_df.columns)):
        for j in range(i+1, len(corr_df.columns)):
            value = corr_df.iloc[i, j]
            if value > 0.50 and value != 1:
                col1 = corr_df.columns[i]
                col2 = corr_df.columns[j]
                print(f"Correlation between {col1} and {col2}: {value:.4f}")

# Pairplot function is defined to create a pair plot between variables
def Pairplot(df):
    sns.pairplot(df)
    plt.show()

# BarPlot function is defined to create a bar plot with 2 variables
def BarPlot(df, x_axis, hue_name):
    # Group by x_axis and hue_name and calculate the value counts
    grouped = df.groupby([x_axis, hue_name]).size()
    # Calculate the total counts for each level in hue_name
    total_counts = grouped.groupby(level=hue_name).sum()
    # Plot the count plot with annotations for each bar
    plt.figure(figsize=(12, 6))
    ax = sns.countplot(x=x_axis, hue=hue_name, data=df)
    ax.set_ylabel(hue_name)
    ax.legend(title=hue_name, bbox_to_anchor=(1, 1))
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45)   
    # Add annotations for each bar
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        ax.annotate(height, ((x) + width / 2, (y) + height / 2), ha='center', va='bottom')
    # Display the plot
    plt.show()

# Defined CreateModels function to create the model object
def CreateModels(models, n_estimators):
    # Define Linear models
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    # Define Ensemble Models
    models.append(('RFC', RandomForestClassifier(n_estimators=n_estimators, random_state=42)))
    base_estimator = DecisionTreeClassifier(max_depth=1)
    models.append(('ADAB', AdaBoostClassifier(base_estimator=base_estimator, n_estimators=n_estimators, random_state=42)))
    # Define Boosting Models
    models.append(('XGB', XGBClassifier()))
    models.append(('HGBC', HistGradientBoostingClassifier(max_bins=10, max_iter=100)))
    return models

# Define BuildModel function 
# to evaluate the models, display the confusion matrix and plot it
# to display the classification_report
def BasicModel(models, X_train, Y_train, X_test, Y_test):
    results = []
    names = []
    basicscore = []
    # evaluate each model in turn
    for name, model in models:
        # Train Decision Tree Classifer
        modelfit = model.fit(X_train,Y_train)
        #Predict the response for test dataset
        Y_predict = modelfit.predict(X_test)
        results.append(metrics.accuracy_score(Y_test, Y_predict))
        names.append(name)
        # Print the prediction of test set
        print('On %s Accuracy is: %f ' % (name, metrics.accuracy_score(Y_test, Y_predict)*100))
        basicscore.append({"Model Name": name, "Accuracy": metrics.accuracy_score(Y_test, Y_predict)*100})
        # Print Confusion Matrix and Classification Report
        print("Confusion matrix:")
        print(confusion_matrix(Y_test, Y_predict))
        print("Classification report:")
        print(classification_report(Y_test, Y_predict))
        # Plot Confusion Matrix
        print("Confusion matrix plot:")
        cm = confusion_matrix(Y_test, Y_predict, labels=modelfit.classes_)
        cmdisp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=modelfit.classes_)
        cmdisp.plot()
        plt.show()
        
        # Add a new line here
        Newline()
        
    return basicscore

# Define BuildModelRS function to evaluate the models based on the random states
def BuildModelRS(models,randstate, X, Y):
    results = []
    names = []
    score = []
    # evaluate each model in turn
    for name, model in models:
        # for loop will run the decision tree model on different random states to find the accuracy
        for n in randstate:
            # The training set and test set has been splited below using the feature and target dataframes
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=n)
            # Train Decision Tree Classifer
            modelfit = model.fit(X_train,Y_train)
            #Predict the response for test dataset
            Y_predict = modelfit.predict(X_test)
            results.append(metrics.accuracy_score(Y_test, Y_predict))
            names.append(name)
            # Store the prediction of test set
            score.append({"Model Name": name, "Random State": n, "Accuracy": metrics.accuracy_score(Y_test, Y_predict)*100})
    return score

# FindBestRandomState function is defined to retrieve the best performed random state of all models
def FindBestRandomState(df):
    # Group the DataFrame by column 1
    groupmodels = df.groupby('Model Name')
    randstatelist = []
    randstatelist.clear()
    # Iterate over each group
    for name, data in groupmodels:
        # Find the maximum value in column 3
        max_value = data['Accuracy'].max()
        # Find the corresponding value in column 2
        bestrandstate = data.loc[data['Accuracy'] == max_value, 'Random State'].values[0]
        randstatelist.append({"Model Name": name, "Random State": bestrandstate, 'Accuracy': max_value})
        # Print the value in column 2
    #print(randstatelist)
    dfrsmodels = pd.DataFrame(randstatelist)
    return dfrsmodels

# Define BalanceData function to balance the dataset
def BalanceData(X,Y):
    # Define oversmaple with SMOTE function
    oversample = SMOTE()
    # Define undersample with RandomUnderSampler function
    undersample = RandomUnderSampler()
    # Define Steps for oversample and undersample
    steps = [('o', oversample), ('u', undersample)]
    # Define the pipeline with the steps
    pipeline = Pipeline(steps = steps)
    # Fit the features and target and resample them to get X and Y
    X_Balanced, Y_Balanced = pipeline.fit_resample(X, Y)
    return X_Balanced, Y_Balanced

# Define function BuildModelBalCV on the balanced data and utilizing cross validation
def BuildModelBalCV(models, X, Y):
    results = []
    names = []
    score = []
    # evaluate each model in turn  
    for name, model in models:
        # define StratifiedKFold
        skfold = StratifiedKFold(n_splits=10, random_state= None, shuffle=True)
        # get the X and Y using StratifiedKFold
        skfold.get_n_splits(X,Y)
        # evaluate each model with cross validation
        if name == "ADAB":
            # evaluate the model
            rskfold = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=None)
            cv_results = cross_val_score(model, X, Y.values.ravel(), scoring='accuracy', cv=rskfold, n_jobs=-1, error_score='raise')
        elif name == 'XGB':
            rskfold = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=None)
            cv_results = cross_val_score(model, X, Y.values.ravel(), cv=rskfold, scoring='accuracy', n_jobs=-1)#, early_stopping_rounds=10)
        else:
            cv_results = cross_val_score(model, X, Y, cv=skfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        # print the results
        #print('On %s: Mean is %f and STD is %f' % (name, cv_results.mean()*100, cv_results.std()))
        score.append({"Model Name": name, "Mean": cv_results.mean()*100, "STD": cv_results.std()})
    return score, results, names

# BestHyperparameters function is defined to find the best hyperparameter from all moels
def BestHyperparameters(modelname, models, dfrs, grid, X_Balanced, Y_Balanced):
    for name, model in models:
        if name == 'ADAB':
            escore = 'raise'
        else:
            escore = 0
            
        if name == modelname:
            # Retrieve the best random state values derived above
            rs = dfrs.loc[dfrs["Model Name"] == name,['Random State']]
            # define StratifiedKFold
            skfold = StratifiedKFold(n_splits=10, random_state=rs.iloc[0][0], shuffle=True)
            # define search
            search = GridSearchCV(estimator=model, param_grid=grid, scoring='accuracy', cv=skfold, n_jobs=-1, error_score=escore)
            # perform the search
            grid_results = search.fit(X_Balanced, Y_Balanced.values.ravel())
            # summarize
            print('%s Mean Accuracy: %f' % (name, grid_results.best_score_))
            print('Config: %s' % grid_results.best_params_)
            # summarize results
            #print("Best Accuracy: %f using %s" % (results.best_score_, results.best_params_))
            #means = results.cv_results_['mean_test_score']
            #stds = results.cv_results_['std_test_score']
            #params = results.cv_results_['params']
    return grid_results

# Defined CreateFinalModels function to create the model object with hyperparameters
def CreateFinalModels(models, parameters):
    # Define Linear models
    models.append(('LR', LogisticRegression(#C = parameters[0]['C'], 
                                            #max_iter = parameters[0]['max_iter'],
                                            penalty = parameters[0]['penalty'], 
                                            solver = parameters[0]['solver'])))
    
    models.append(('LDA', LinearDiscriminantAnalysis(solver = parameters[1]['solver'])))
    
    # Define Ensemble Models
    #models.append(('RFC', RandomForestClassifier(n_estimators=n_estimators, random_state=42)))
    models.append(('RFC', RandomForestClassifier(max_depth = parameters[2]['max_depth'],
                                                max_features = parameters[2]['max_features'], 
                                                n_estimators = parameters[2]['n_estimators'])))
    
    base_estimator = DecisionTreeClassifier(max_depth=1)
    #base_estimator = DecisionTreeClassifier()
        #models.append(('ADAB', AdaBoostClassifier(base_estimator=base_estimator, n_estimators=n_estimators, random_state=42)))
    models.append(('ADAB', AdaBoostClassifier(base_estimator=base_estimator,
                                             learning_rate = parameters[3]['learning_rate'],
                                             n_estimators = parameters[3]['n_estimators'])))
    
    # Define Boosting Models
    models.append(('XGB', XGBClassifier(max_depth = parameters[4]['max_depth'],
                                       min_child_weight = parameters[4]['min_child_weight'])))
    
    #models.append(('HGBC', HistGradientBoostingClassifier(max_bins=10, max_iter=100)))
    models.append(('HGBC', HistGradientBoostingClassifier(max_bins = parameters[5]['max_bins'],
                                                         max_iter = parameters[5]['max_iter'])))
    
    return models

# Define BuildFinalModel function to evaluate models with their hyper-parameters
def BuildFinalModel(final_models, dfrs, X, Y):
    #final_models = []
    final_results = []
    names = []
    score = []
    #final_models.clear()
    #final_results.clear()
    names.clear()
    score.clear()
    # Evaluate each model in turn 
    for name, model in final_models:
        rs = dfrs.loc[dfrs["Model Name"] == name,['Random State']]
        # Define StratifiedKFold
        skfold = StratifiedKFold(n_splits = 10, random_state = rs.iloc[0][0], shuffle = True)
        # Get the X and Y using StratifiedKFold
        skfold.get_n_splits(X,Y)
        # Evaluate each model with cross validation
        cv_results = cross_val_score(model, X, Y, cv=skfold, scoring='accuracy')
        # Store the cross validationscore into results
        final_results.append(cv_results)
        #print(final_results)
        # Store model name into names
        names.append(name)
        # Print the results
        #print('On %s: Mean is %f and STD is %f' % (name, cv_results.mean(), cv_results.std()))
        # Store the Model Name, Mean and STD into score list
        score.append({"Model Name": name, "Accuracy - Mean": cv_results.mean(), "Accuracy - STD": cv_results.std()})
    return score, names, final_results

