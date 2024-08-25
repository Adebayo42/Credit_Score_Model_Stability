# This Python module contains functions for feature selection

## It takes about 5hrs to complete running


# Writen By: F315284
# Date: August, 2024


import pandas as pd
import numpy as np
from stqdm import stqdm
import general_function as gen_func
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import make_pipeline as imb_make_pipeline
from imblearn.over_sampling import KMeansSMOTE
import streamlit as st

# Function 0: Feature Selection
##############################################################################
@st.cache_data
def feature_selection():
    """
    Select important features using RandomizedSearchCV.

    This function applies a pipeline with SMOTE and feature selection to identify the best 
    number of features (`k`) based on accuracy through randomized search.

    Returns:
        None
    """
    
    # Initialice pipeline to preprate data for feature selection
    pipeline_balancing = imb_make_pipeline(gen_func.CustomImputer(),
                                           gen_func.CustomTargetEncoder(),
                                           KMeansSMOTE(random_state=42,cluster_balance_threshold=0.1))
    X_balanced, y_balanced = pipeline_balancing.fit_resample(gen_func.X_train, gen_func.y_train)
    
    # Initialize the feature selection process with a pipeline
    pipeline_feature_selection = imb_make_pipeline(SelectKBest(f_classif),
                                                   RandomForestClassifier(n_estimators=30,random_state=42))

    # Define the parameter distribution
    param_dist = {
        'selectkbest__k': np.arange(10, 30),  # Range of k values from 10 to 30 of the number of features
    }

    # Run randomized search
    random_search = RandomizedSearchCV(pipeline_feature_selection,
                                       param_distributions=param_dist,
                                       n_iter=10,
                                       cv = StratifiedKFold(n_splits=3,shuffle=True, random_state=42),
                                       scoring='accuracy',
                                       random_state=42,
                                       n_jobs = -1)

    # Fit the randomized search
    for iter_ in stqdm(range(random_search.n_iter), desc = "Fitting RandomizedSearchCV"):
        random_search.fit(X_balanced, y_balanced.values.ravel())

    ## convert the search result to dataframe and rank the result
    results_df = pd.DataFrame(random_search.cv_results_)
    results_df = results_df[['param_selectkbest__k', 'mean_test_score','std_test_score','rank_test_score']]
    result_df = results_df.sort_values('rank_test_score')

    print("Feature Selection: Comparison of K-Value for Feature Selection")
    print("---------------------------------------------------------------\n") 
    # Display the result
    st.dataframe(result_df)
    
    # Process Completion Message
    print(f"\nStep 8 completed!!!\n")
    
    
    
## Module Test Function
def test_func():
    print(" feature_selection module is working fine")
    
#Function Testing
##############################################################################

if __name__=="__main__":
    test_func()   
