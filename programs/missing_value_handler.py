# Missing Value Handler Module

# This Python module contains functions for handling the missing values in the application train dataset.
# It provides methods to efficiently drops features with over 20% missing values, replaces specific placeholder values 
# (e.g., "XNA", "XAP", "Unknown") and anomalous values (e.g., 365243 in 'DAYS_EMPLOYED') with NaN. 


# Writen By: F315284
# Date: August, 2024

import data_loading as data
import data_exploration as xplore
import numpy as np
import streamlit as st

# Function 0: Create dictionaries to store file names
##############################################################################

def na_handler():
    """
    Handle missing values in the input variables.

    Drops features with over 20% missing values, replaces specific placeholder values 
    (e.g., "XNA", "XAP", "Unknown") and anomalous values (e.g., 365243 in 'DAYS_EMPLOYED') 
    with NaN. Updates the global variable `X_input` with the cleaned dataset.

    Globals:
        X_input (DataFrame): Cleaned DataFrame of input variables.
        missing_percentage (Series): Percentage of missing values per feature.
        X (DataFrame): Original input variables.

    Returns:
        None
    """
    
    try:
        # Identify features with missing values greater than the 20% 
        features_to_drop = (xplore.missing_percentage.loc[xplore.missing_percentage > 20]).index
        
        # Drop features with High Missing Values
        globals()["X_input"] = xplore.X.drop(columns=features_to_drop, axis=1)
        
        # Replace XNAs & XAP values
        X_input.replace(to_replace={"XNA": np.nan, "XAP": np.nan}, inplace=True)
        
        # Replace unknown value 
        X_input["NAME_FAMILY_STATUS"].replace("Unknown", np.nan, inplace=True)
        
        # Replace strange value 
        X_input["DAYS_EMPLOYED"].replace(365243, np.nan, inplace=True)
        
        # Create variable to shape
        input_x_shape = X_input.shape
        
        # Formatted shape
        formatted_shape = f"{input_x_shape[0]:,} rows, and {input_x_shape[1]:,} columns"
        
        #st.info("Preprocessing: Handling Missing Values")
        print("-------------------------------------------------\n") 
        # Print dataset shape
        st.write(f'The shape of the Input variables is: {formatted_shape} after dropping features with over 20% missing values')
        print(f"\nStep 4 completed!!!\n")

    except Exception as e:
        print(f"An error occurred while handling missing values: {e}")
        
        
## Module Test Function
def test_func():
    print(" missing_value_handler module is working fine")


#Function Testing
##############################################################################

if __name__=="__main__":
    test_func()
    
    
    
