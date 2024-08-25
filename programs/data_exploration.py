# Data Exploration Module

# This Python module contains functions for exploratory data analysis.
# It provides methods to efficiently carry out data exploration analysis on the input variables and target variable

# Writen By: F315284
# Date: August, 2024

from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import data_loading as data
import streamlit as st
import time


# Function 0: Exploratory Analysis - Dataset Shape
##############################################################################

def dataset_shape():
    """
    Analyze and print the shape of each DataFrame stored in the global `files_dict`.

    Globals:
        files_dict (dict): A dictionary containing DataFrames loaded from CSV files.

    Returns:
        None
    """
    
    print("Exploratory Data Analysis: Dataset Shape Analysis")
    print("-------------------------------------------------\n") 
    
    try:
        for key, value in tqdm(data.files_dict.items(), desc="Analyzing Dataset Shape", total=len(data.files_dict.items())):
            shape = value.shape
            # Formatted shape
            formatted_shape = f"{shape[0]:,} rows, and {shape[1]:,} columns"
            # Print dataset shape
            st.write(f'The shape of the {key} table is: {formatted_shape}')

    except Exception as e:
        print(f"An error occurred during dataset shape analysis: {e}")

    print(f"\nStep 2 completed!!!\n")
    
    

# Function 1: Exploratory Data Analysis(EDA) - Target Variable Analysis
##############################################################################

def target_variable_EDA():
    """
    Perform EDA on the target variable from the application dataset.

    Extracts the 'TARGET' variable, calculates its distribution, visualizes it with a bar plot, 
    and prints unique values and missing counts.

    Globals:
        Y (DataFrame): 'TARGET' variable from 'application_train'.
        files_dict (dict): Loaded DataFrames from CSV files.

    Returns:
        None
    """
    
    try:
        # Split the target variable from the application_train dataset
        globals()['Y'] = data.files_dict["Y"][['TARGET']].copy()

        # Distribution of Target variable
        value_counts = Y['TARGET'].value_counts()
        value_counts_pct = (value_counts / len(Y)) * 100

        # Set the theme for the plot
        sns.set_theme(style="whitegrid")

        # Create the bar plot for the Target variable distribution
        plt.figure(figsize=(6, 4))
        ax = value_counts.plot(kind='bar', rot=0, color=['blue', 'red'])

        # Annotate each bar with the value and the percentage rounded to 1 decimal place
        for index, (value, pct) in enumerate(zip(value_counts, value_counts_pct)):
            ax.text(index, value, f'{value:,} ({pct:.1f}%)', ha='center', va='bottom')

        # Set the title of the plot
        ax.set_title('Distribution of TARGET Variable')

        # Show the plot
        st.pyplot(plt)

        #st.write("Exploratory Data Analysis: Target Variable Analysis")
        #st.write("-------------------------------------------------\n")

        # Exploratory Analysis on Target Variable
        Y_unique = Y["TARGET"].unique()
        formatted_value = f"{Y_unique[0]:,} and {Y_unique[1]:,}"
        st.write(f"Unique Values of Target Variable are: {formatted_value}")
        st.write(f"Count of missing values in Target Variable is: {Y['TARGET'].isnull().sum()}")

        #st.success(f"\nStep 3 completed!!!\n")

    except Exception as e:
        st.write(f"An error occurred during target variable analysis: {e}")


# Function 2: Exploratory Data Analysis(EDA) - Input Variable Analysis
##############################################################################
 
def input_variable_EDA():
    """
    Perform EDA on input variables from the application dataset.

    Extracts input variables (excluding 'TARGET'), analyzes feature types, visualizes counts, 
    assesses missing values, identifies features with >20% missing data, and checks for duplicates.

    Globals:
        X (DataFrame): Input variables from 'application_train'.
        missing_percentage (Series): Percentage of missing values per feature.
        files_dict (dict): Loaded DataFrames from CSV files.

    Returns:
        None
    """
    try:
        # Split the input variable from the application_train dataset
        globals()["X"] = data.files_dict["dataset"].copy()

        ###-------- Exploratory Analysis on numeric and non-numeric input variables-------###
        # Identify numeric and non-numeric features
        numeric_features = X.select_dtypes(include=['number']).columns
        non_numeric_features = X.select_dtypes(exclude=['number']).columns

        # Count the features
        num_numeric = len(numeric_features)
        num_non_numeric = len(non_numeric_features)

        # Create a bar plot distribution of count of numeric and non-numeric features
        labels = ['Numeric Features', 'Non-Numeric Features']
        counts = [num_numeric, num_non_numeric]

        # Set the theme for the plot
        sns.set_theme(style="whitegrid")

        # Plot the distribution
        plt.figure(figsize=(6, 4))
        bars = plt.bar(labels, counts, color=['green', 'orange'])
        plt.title('Count of Numeric and Non-Numeric Features')
        plt.ylabel('Count')
        plt.xlabel('Feature Type')

        # Add the count values on the bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom')  # va: vertical alignment

        ###-------- Exploratory Analysis on count of missing values of input variables-------###
        
        # Calculate the percentage of missing values for each feature
        globals()['missing_percentage'] = (X.isnull().mean() * 100).round(0).sort_values(ascending=True)

        # Count the number of features with more than 20% missing values
        num_features_with_missing_values = len(missing_percentage.loc[missing_percentage > 20])

        # Plot distribution
        fig = plt.figure(figsize=(22, 8))
        ax = missing_percentage.loc[missing_percentage > 20].plot(kind='bar')

        # Annotate each bar with the percentage value
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 10), textcoords='offset points')

        # Set the title of the plot
        ax.set_title('Features with more than 20% missing values', fontsize=20)
        ax.set_ylabel('Percentage of missing values', fontsize=20)

        # Display the plot
        st.pyplot(plt)

        # Check for duplicated records
        duplicated = X.duplicated().sum()

        print("Exploratory Data Analysis: Input Variable Analysis")
        print("-------------------------------------------------\n") 
        # Print the output
        st.write(f'Number of features with more than 20% missing values is: {num_features_with_missing_values}')
        st.write(f"Count of duplicated records is: {duplicated}")
        st.success(f"Input Variable Analysis Completed")

    except Exception as e:
        print(f"An error occurred during input variable analysis: {e}")
        
        
## Module Test Function
def test_func():
    print(" data_exploration module is working fine")


#Function Testing
##############################################################################

if __name__=="__main__":
    test_func()
    
    
