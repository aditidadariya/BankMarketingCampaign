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
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import QuantileTransformer
from scipy.stats import mode

import matplotlib.pyplot as plt
import seaborn as sns

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
    # Define the numerical columns in the dataframe
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Define the threshold
    threshold = 3

    # Loop through each numerical column
    for column in numerical_columns:
        # Keeping 'age' variable intact
        if column != 'AGE':
            # Compute the Z-scores
            z_scores = np.abs(stats.zscore(df[column]))
        
            # Find the outliers
            outliers = np.where(z_scores > threshold)
            outlier_indices = outliers[0]
            outlier_values = df.iloc[outlier_indices]
            # Print the outliers for the column
            if not outlier_values.empty and outlier_values.shape[0] > 100 :
                print(f"{outlier_values.shape[0]} number of row are Outliers in column {column}")
                # Correct the outliers in dataset
                df[column] = ImputeOutliers(z_scores, threshold, df, column)
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

