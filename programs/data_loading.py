# Loading Data Module

# This Python module contains functions for loading datasets utilized in research analysis.
# It provides methods to efficiently import data from CSV files

# Writen By: F315284
# Date: August, 2024

import os
import pandas as pd
import numpy as np
from stqdm import stqdm
import streamlit as st

# Function 0: Create dictionaries to store file names
##############################################################################

def file_names(path):
    """
    Generate a dictionary mapping unique keys to file names based on the provided path.

    Parameters:
    - path (str): The path to the directory containing the files.

    Returns:
    dict: A mapping of keys to corresponding file names.
   
    Raises:
    FileNotFoundError: If the specified path does not exist.
    ValueError: If the path is not a directory.
    """
    try:
        # Check if the path exists and is a directory
        if not os.path.exists(path):
            raise FileNotFoundError(f"The specified path '{path}' does not exist.")
        if not os.path.isdir(path):
            raise ValueError(f"The specified path '{path}' is not a directory.")

        key, value = [], []
        for file in os.listdir(path):
            # Only consider files with a .csv extension
            if file.endswith('.csv'):
                key.append(file[:-4])  # Exclude the '.csv' extension from the key
                value.append(file)
        return dict(zip(key, value))

    except Exception as e:
        print(f"An error occurred: {e}")
        return {}
    

    
# Function 1: Load Data sets from the specified directory
##############################################################################
files_dict = {}

#st.cache_data
def load_files():    
    
    """
    Load CSV files from the specified directory and store them in a global dictionary.

    The function reads all CSV files in the 'data/' directory, using 'SK_ID_BUREAU' as the index for 
    the 'bureau_balance' file and 'SK_ID_CURR' for all other files. 

    Globals:
        files_dict (dict): A dictionary to store the loaded DataFrames with file names as keys.
    
    Returns:
        None
    """
    
    # Assign variable to path name
    path = "./data/"
    
    # Store loaded files in a dictionary
    #globals()["files_dict"] = {}
    
    
    # Store files names loaded successfully
    files_loaded = []
    
    # Read CSV files
    try:
        files = file_names(path)
        progress_text = "Data loading..."
        #my_bar = st.progress(0, text=progress_text)
        for i, key in stqdm(enumerate(files), desc="Loading Data", total=len(files)):
            #my_bar.progress(i/6, text=progress_text)
            if key == 'bureau_balance':
                files_dict[key] = pd.read_csv(path + files[key], index_col='SK_ID_BUREAU')
            else:
                files_dict[key] = pd.read_csv(path + files[key], index_col='SK_ID_CURR')
            
            files_loaded.append(key)
    
    except Exception as e:
        print(f"An error occurred: {e}")
    
    print(f"\nData Loading: Step 1 completed!!!")
    print("-----------------------------------")
    # Print success message
    for i in files_loaded:
        print(f"{i} data was loaded successfully")

    return 

        
## Module Test Function
def test_func():
    print(" data_loading module is working fine")


#Function Testing
##############################################################################

if __name__=="__main__":
    test_func()
    

