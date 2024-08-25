# This Python module contains functions used for the SMOTE class balancing variants evaluation


# Writen By: F315284
# Date: August, 2024

import numpy as np
import matplotlib.pyplot as plt
import general_function as gen_func
import pandas as pd
from stqdm import stqdm
from imblearn.over_sampling import KMeansSMOTE, BorderlineSMOTE
from imblearn.pipeline import make_pipeline as imb_make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score
import streamlit as st



# Function 0: SMOTE Variants Evaluation
##############################################################################
def class_balancing():
    """
    Evaluate and compare SMOTE variants for class balancing.

    This function applies different SMOTE techniques to balance classes in the dataset and 
    calculates the balanced accuracy for each variant.

    Returns:
        None
    """
    
    # Create dictionary for initialized smote variants
    smote = {"KMeansSMOTE": KMeansSMOTE(random_state=4, cluster_balance_threshold=0.1),
             "BorderlineSMOTE":BorderlineSMOTE(random_state=42)}

    ### Create dictionary to store accuracy
    balanced_accuracy = {}

    ## Loop the dictionary to evalauate each smote variants
    for key, value in stqdm(smote.items(), desc = "Class balancing"):

        ## Pipeline for Sythetic Minority Over-sampling technique
        pipeline = imb_make_pipeline(gen_func.CustomImputer(),
                                     gen_func.CustomTargetEncoder(),
                                     value,
                                     DecisionTreeClassifier(random_state=42))
        # Fit and predict using pipeline
        pipeline.fit(gen_func.X_train,gen_func.y_train)
        prediction = pipeline.predict(gen_func.X_test)

        # balanced accuracy score
        bal_acc = balanced_accuracy_score(gen_func.y_test,prediction)    
        balanced_accuracy[key] = bal_acc.round(4)


    ## Convert Dictionary to dataframe and display
    smote_accuracy_df = pd.DataFrame(balanced_accuracy, index=['Balanced Accuracy'])
    
    # Output message
    print("Class Balancing: Evaluation of SMOTE Variants")
    print("-------------------------------------------------\n") 
    st.dataframe(smote_accuracy_df)
    
    # Process Completion Message
    print(f"\nStep 6 completed!!!\n")

    
    
# Function 1: SMOTE Application
##############################################################################
def smote_application():
    """
    Apply Synthetic Minority Oversampling Technique (SMOTE) to balance class distribution.

    This function visualizes the distribution of the target variable before and after 
    applying SMOTE to the training dataset.

    Returns:
        None
    """
    
    #### Syhthetic Minority Oversampling Technique (SMOTE)

    # Create subplots
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    # Distribution of Target variable
    value_counts = gen_func.y_train.value_counts()
    value_counts_pct = (value_counts / len(gen_func.y_train)) * 100

    # Create the bar plot for original data
    value_counts.plot(kind='bar', ax=ax[0], rot=0, color=['blue', 'red'])

    # Annotate each bar with the value and the percentage rounded to 1 decimal place
    for index, (value, pct) in enumerate(zip(value_counts, value_counts_pct)):
        ax[0].text(index, value, f'{value:,} ({pct:.1f}%)', ha='center', va='bottom', fontsize=10)  

    # Set the title of the first subplot with reduced font size
    ax[0].set_title('Distribution of TARGET Variable (Original)', fontsize=10)

    # Set y-axis limits for the original distribution
    ax[0].set_ylim(0, 250000)

    # Instantiate SMOTE
    ## Pipeline for Sythetic Minority Over-sampling technique
    smote = imb_make_pipeline(gen_func.CustomImputer(),
                              gen_func.CustomTargetEncoder(),
                              KMeansSMOTE(random_state=42,cluster_balance_threshold=0.1))


    # Apply SMOTE to generate synthetic samples
    with stqdm(total=1, desc="Applying SMOTE...") as pbar:
        # Apply SMOTE to generate synthetic samples
        X_resampled, y_resampled = smote.fit_resample(gen_func.X_train, gen_func.y_train)
        pbar.update(1) 


    # Visualize Target feature distribution after SMOTE
    value_counts_resampled = y_resampled.value_counts().sort_index()
    value_counts_pct_resampled = (value_counts_resampled / len(y_resampled)) * 100

    # Create the bar plot for resampled data
    value_counts_resampled.plot(kind='bar', ax=ax[1], rot=0, color=['blue', 'red'])

    # Annotate each bar with the value and the percentage rounded to 1 decimal place
    for index, (value, pct) in enumerate(zip(value_counts_resampled, value_counts_pct_resampled)):
        ax[1].text(index, value, f'{value:,} ({pct:.1f}%)', ha='center', va='bottom', fontsize=10)  

    # Set the title of the second subplot with reduced font size
    ax[1].set_title('Distribution of TARGET Variable (After SMOTE)', fontsize=10)

    # Set y-axis limits for the resampled distribution
    ax[1].set_ylim(0, 250000)

    # Set x and y labels with reduced font size
    ax[0].set_xlabel('Classes', fontsize=10)
    ax[0].set_ylabel('Count', fontsize=10)
    ax[1].set_xlabel('Classes', fontsize=10)
    ax[1].set_ylabel('Count', fontsize=10)

    # Reduce the fontsize of the tick labels on both axes
    ax[0].tick_params(axis='both', labelsize=10)  
    ax[1].tick_params(axis='both', labelsize=10)  

    # Show the plot
    plt.tight_layout()
    # Output message
    st.markdown("*Training Dataset Class Distribution Before and After Borderline SMOTE Application*")
    print("-----------------------------------------------------------------------------------\n") 
    st.pyplot(plt)
    
    # Process Completion Message
    print(f"\nStep 7 completed!!!\n")

    
## Module Test Function
def test_func():
    print(" class_balancing module is working fine")
    
#Function Testing
##############################################################################

if __name__=="__main__":
    test_func()    