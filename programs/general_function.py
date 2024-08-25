# This Python module contains functions used by other modules


# Writen By: F315284
# Date: August, 2024

import data_loading as data
import pandas as pd
import feature_engineering as feat_eng
import missing_value_handler as mvh
import data_exploration as xplore
import gc
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import numpy as np
import category_encoders as ce
from stqdm import stqdm
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import KMeansSMOTE, BorderlineSMOTE
from imblearn.pipeline import make_pipeline as imb_make_pipeline
import streamlit as st

# Function 0: Free Memory to optimize computing resource
##############################################################################
def free_memory():
    """
    Free memory by deleting the global dataset dictionary and invoking garbage collection.

    Globals:
        files_dict (dict): Dictionary storing loaded datasets.

    Returns:
        None
    """
    
    # Declare files_dict as global
    global files_dict
    
    # Delete dictionary storing datasets
    del data.files_dict
    
    # Garbage collect
    gc.collect()
    
    print("Memory space cleared successfully")
    

# Function 1: Split Train Test Dataset function
##############################################################################    
def split_train_test():
    """
    Split the dataset into training and testing sets.

    Global variables:
        X_train (DataFrame): Training features.
        X_test (DataFrame): Testing features.
        y_train (Series): Training labels.
        y_test (Series): Testing labels.

    Returns:
        None
    """
    
    # Create Global variables
    global X_train, X_test, y_train, y_test
    
    # Split dataset to train and test set
    X_train, X_test, y_train, y_test = train_test_split(mvh.X_input, xplore.Y, 
                                                        test_size = 0.3,
                                                        stratify = xplore.Y,
                                                        random_state =42)   
    
    # Output message
    print("Preprocessing: Splitting Dataset to train and test dataset")
    print("-------------------------------------------------\n") 
    
    # Create name for the dataset using dictionary
    names = ["dataset", "trainset", "testset"]
    datas = [mvh.X_input, X_train, X_test]
    names_dict = dict(zip(names,datas))
    # Dataset shape
    for name, data in stqdm(names_dict.items(), desc="Splitting dataset"):
        shape_i = data.shape
        formatted_i_shape = f"{shape_i[0]:,} rows, and {shape_i[1]:,} columns"
        st.write(f"The shape of the {name} is: {formatted_i_shape}")
        

    st.info("Dataset has been split successful, using 70% for trainset and 30% for test set")
    # Process Completion Message
    print(f"\nStep 5 completed!!!\n")

    
# Class 1: Create custom class for simple imputer and Target encoding
##############################################################################     
## Custom SimpleImputer Transformer
class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.imputer_median = SimpleImputer(strategy='median')
        self.imputer_most_frequent = SimpleImputer(strategy='most_frequent')
        self.numeric_cols = None
        self.non_numeric_cols = None

    def fit(self, X, y=None):
        X = X.replace([np.inf, -np.inf], np.nan)
        self.numeric_cols = X.select_dtypes(include=[np.number]).columns
        self.non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns

        self.imputer_median.fit(X[self.numeric_cols])
        self.imputer_most_frequent.fit(X[self.non_numeric_cols])
        return self

    def transform(self, X):
        X = X.replace([np.inf, -np.inf], np.nan)
        
        X[self.numeric_cols] = self.imputer_median.transform(X[self.numeric_cols])
        X[self.non_numeric_cols] = self.imputer_most_frequent.transform(X[self.non_numeric_cols])
        return X

## Custom Target Encoder Transformer
class CustomTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols
        self.encoder = None
    
    def fit(self, X, y=None):
        if self.cols is None:
            self.cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        self.encoder = ce.TargetEncoder(cols=self.cols)
        self.encoder.fit(X[self.cols], y)
        return self
    
    def transform(self, X):
        if self.encoder is None:
            raise ValueError("This CustomTargetEncoder instance is not fitted yet.")
        
        X_encoded = X.copy()
        X_encoded[self.cols] = self.encoder.transform(X[self.cols])
        
        return X_encoded

    
    
# Function 2: Preprocess the dataset
##############################################################################
def preprocess_dataset():
    """
    Preprocess the training and test datasets for modeling.

    This function applies a preprocessing pipeline that includes imputation, encoding, 
    feature selection, and SMOTE for the training dataset, and transforms the test dataset 
    without applying SMOTE.

    Returns:
        None
    """
    
    # Declare global variables
    
    global X_train_processed_df, y_train_processed, X_test_processed_df
    
    # Preprocess the dataset- Train dataset
    preprocess_pipeline = imb_make_pipeline(CustomImputer(),
                                            CustomTargetEncoder(),
                                            SelectKBest(f_classif, k=25),
                                            KMeansSMOTE(random_state=42,cluster_balance_threshold=0.1)
                                           )

    with stqdm(desc = "Preprocessing data", total =1) as pbar:
        X_train_processed, y_train_processed = preprocess_pipeline.fit_resample(X_train, y_train)
        pbar.update(1)

    # Preprocess the dataset Get feature names after selection
    selector = preprocess_pipeline.named_steps['selectkbest']
    selected_indices = selector.get_support(indices=True)  # Get indices of selected features

    # Create a DataFrame from the selected features
    X_train_processed_df = pd.DataFrame(X_train_processed, columns=X_train.columns[selected_indices])
    
    # Preprocess the test dataset
    X_test_processed = preprocess_pipeline[:-1].transform(X_test)  # Exclude SMOTE from transformation

    # Convert to dataframe
    X_test_processed_df = pd.DataFrame(X_test_processed, columns=X_train.columns[selected_indices])
    
        
    print("Preprocessing: Data preprocessed for modelling")
    print("---------------------------------------------------------------\n")
    
    # Create name for the dataset using dictionary
    names = ["processed_trainset", "processed_testset"]
    datas = [X_train_processed_df, X_test_processed_df]
    names_dict = dict(zip(names,datas))
    # Dataset shape
    for name, data in names_dict.items():
        shape_i = data.shape
        formatted_i_shape = f"{shape_i[0]:,} rows, and {shape_i[1]:,} columns"
        st.write(f"The shape of the {name} is: {formatted_i_shape}")

    ## Display preprocessed dataset
    st.info("Display top 20rows of preprocessed dataset")
    st.dataframe(X_train_processed_df.head(20))
    # Process Completion Message
    print(f"\nStep 9 completed!!!\n")
    
    
    
    
## Module Test Function
def test_func():
    print(" general_function module is working fine")
    
#Function Testing
##############################################################################

if __name__=="__main__":
    test_func()
