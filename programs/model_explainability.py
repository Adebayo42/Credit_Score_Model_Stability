# Model Explainabilty Module

# This Python module contains functions for performing model explanations
# using SOBOL Sensitiy Analysis and SHApley exPlanations (SHAP) values.

# Writen By: F315284
# Date: August, 2024

import model_development as model_dev
import sythetic_data_augmentation as sda
import general_function as gen_func
from SALib.sample import sobol
from SALib.analyze import sobol as sobol_analyze
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from stqdm import stqdm
import streamlit as st


# Function 0: SOBOL Sensitivity Analysis
##############################################################################

def sensitivity_analysis():
    """
    Perform sensitivity analysis on the 'Best Stable' classifier 
    using the Sobol method to evaluate feature importance.

    Returns:
        None
    """
    
    ### Sensitivity Analysis Using SOBOL Method

    stable_classifiers = model_dev.classifiers[f"{sda.most_stable_model}"]
    X_train_processed_df = gen_func.X_train_processed_df

    # Define the problem for Sobol analysis
    problem = {
        'num_vars': X_train_processed_df.shape[1],
        'names': X_train_processed_df.columns.tolist(),
        'bounds': [[X_train_processed_df[col].min(), X_train_processed_df[col].max()] for col in X_train_processed_df.columns]
    }

    # Sample input space using Sobol sequence
    param_values = sobol.sample(problem, 1024, seed=42)  

    # Convert param_values to DataFrame
    param_values_df = pd.DataFrame(param_values, columns=X_train_processed_df.columns)

    # Predict probabilities for class 1
    probabilities = stable_classifiers.predict_proba(param_values_df)[:, 1]

    # Analyze using Sobol
    Si = sobol_analyze.analyze(problem, probabilities, print_to_console=False)
    S1 = Si['S1']

    # Create sorted DataFrame
    df = pd.DataFrame({
        'Features': X_train_processed_df.columns,
        'Sensitivity_Index': S1
    })
    df_sorted = df.sort_values(by='Sensitivity_Index', ascending=False)

    # Define colors for the bars and Ttle
    colors = ['darkgreen' if value > 0 else 'red' for value in df_sorted['Sensitivity_Index']]
    titles = [f'{sda.most_stable_model} Sensitivity Index']

    ### Visualize the Sensitivity Analysis

    plt.figure(figsize=(8,6)) 
    plt.barh(df_sorted['Features'], df_sorted['Sensitivity_Index'], color=colors)  

    # Set title and labels with increased font sizes
    plt.title(titles[0], fontsize=14)  
    plt.xlabel('Sensitivity Index', fontsize=12)  
    plt.ylabel('Features', fontsize=12)  

    # Increase the font size of x-ticks and y-ticks
    plt.tick_params(axis='x', rotation=0, labelsize=10) 
    plt.tick_params(axis='y', labelsize=10)
    
    # display output message
    st.info(f"{sda.most_stable_model} Classifier Model Explanation: Highlighting Contributions of Features to Model Output Variance")
    print("---------------------------------------------------------------------------------------------------------------------")

    # Adjust layout for better spacing
    plt.tight_layout()
    st.pyplot(plt)

    
    
    
# Function 1: SHApley exPlanations (SHAP) values
##############################################################################
def shap_explaination():
    """
    Generate SHAP (SHapley Additive exPlanations) plots to explain the 
    predictions of the 'Best Stable' classifier, visualizing feature 
    importance and model output relationships.

    Returns:
        None
    """
    
    ## Model Explanability
    model_stable_clf = model_dev.classifiers[f'{sda.most_stable_model}']


    ## sample the testset
    X_train_shap, _,y_train_shap,_ = train_test_split(gen_func.X_test_processed_df, 
                                                      gen_func.y_test, train_size = 100,
                                                      stratify = gen_func.y_test, 
                                                      random_state =42 )

    # Explain predictions using SHAP
    explainer = shap.Explainer(model_stable_clf.predict,X_train_shap)
    shap_values = explainer(X_train_shap)
    
    st.info("Summary plots to explain the relationship between input features and model output")
    print("---------------------------------------------------------------------------------")

    # Visualize the SHAP values using summary plot
    plt.figure()
    shap.summary_plot(shap_values,plot_size =(8,6))
    st.pyplot(plt.gcf())

    # Visualize the SHAP values using bar plot
    plt.figure()
    shap.plots.bar(shap_values)
    st.pyplot(plt.gcf())

    # Visualize the SHAP values using waterfall plot to showcase prediction for class 0
    plt.figure()
    shap.plots.waterfall(shap_values[0])
    st.pyplot(plt.gcf())


    # Visualize the SHAP values using waterfall plot to showcase prediction for class 1  

    plt.figure()
    shap.plots.waterfall(shap_values[44])
    st.pyplot(plt.gcf())

    
    
    
## Module Test Function
def test_func():
    print(" model_explainability module is working fine")
    
#Function Testing
##############################################################################

if __name__=="__main__":
    test_func()