# Sythentic Data Augmentation Module

# This Python module contains functions for performing sythentic data augmentation to represent dataset shift.
# The synthetic augmented dataset will you used by the modl to evaluate the model stability on the augmented dataset.

# Writen By: F315284
# Date: August, 2024

import general_function as gen_func
import model_development as model_dev
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stqdm import stqdm
from sklearn.metrics import accuracy_score
import streamlit as st

# Function 0: Synthetic Data Augmentation
##############################################################################

def sythentic_data():
    """
    Augment test data synthetically and visualize the original 
    and augmented feature distributions.

    Globals:
        X_test_aug (DataFrame): Augmented test features.

    Returns:
        None
    """
    
    ### Synthetic data augmentation
    
    # create global variable
    global X_test_aug

    # Create a copy of the processed test set for augmentation
    X_test_processed = gen_func.X_test_processed_df.copy()


    # Function to generate synthetic data
    def augment(data, mu=3, sigma=5, factor=3.5):
        augmented_data = data.copy()
        # Set seed
        np.random.seed(42)
        
        for column in augmented_data.columns:
            augmented_data[column] = (augmented_data[column] 
                                      + np.random.normal(mu, sigma, augmented_data[column].shape)) * factor
        return augmented_data


    # Apply augmentations to all features
    X_test_aug = augment(X_test_processed)

    # Number of features to plot
    num_features = X_test_aug.shape[1]

    # Create a figure with subplots arranged in a grid
    fig, axs = plt.subplots(5, 5, figsize=(25, 22))

    # Flatten axs for easier iteration
    axs = axs.flatten()

    for i, feature in stqdm(enumerate(X_test_aug.columns), desc = "Generating sythentic augmented dataset", total=25):
        # Plot augmented data
        axs[i].hist(X_test_aug[feature], bins=30, alpha=0.5, label='Augmented', color='lightblue', edgecolor='black')

        # Plot original data with enhanced visibility
        axs[i].hist(X_test_processed[feature], bins=30, label='Original', 
                    color='red', alpha=0.9, edgecolor='black', linewidth=1.5)  # Edge color and line width

        axs[i].set_title(feature)
        axs[i].legend()

    # Display Output message
    st.info("Comparison of Feature Distributions: Original Test Data vs. Augmented Test Data")
    print("--------------------------------------------------------------------------------")
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    # Show the plots
    st.pyplot(plt)


# Function 1: Model Stability Evaluation
##############################################################################    
def model_stability_evaluation():
    """
    Evaluate the stability of classifiers by comparing 
    their accuracy on original and augmented test data.

    Returns:
        None
    """
    
    ## Model Stability Evaluation

    global stability_evaluation_df, most_stable_model,most_stable_model_score,most_stable_model_acc_score,min_acc

    # Datasets
    datasets = {
        "original": (gen_func.X_test_processed_df, gen_func.y_test),
        "augmented": (X_test_aug, gen_func.y_test)
    }

    # Function to evaluate already fitted models
    def evaluate_model(clf, X, y):
        pred = clf.predict(X)
        accuracy = accuracy_score(y, pred)
        return accuracy

    # Function to evaluate model stability
    def accuracy_stability(original_accuracy, augmented_accuracy):
        # Stability metrics
        stability_score = 1 - abs(original_accuracy - augmented_accuracy) / original_accuracy
        return stability_score

    # Initialize lists to store results for plotting
    classifier_names = []
    accuracies = []
    augmented_data_accuracies = []
    stability_scores = []
    colors = ['skyblue', 'darkgreen', 'coral', 'darkblue', 'gold', 'darkred' ]

    # Evaluate models and collect results
    for i, (clf_name, clf) in stqdm(enumerate(model_dev.classifiers.items())):
        # Evaluate accuracy and stability score using * to unpack datasets
        original_accuracy = evaluate_model(clf, *datasets["original"])
        augmented_accuracy = evaluate_model(clf, *datasets["augmented"])

        # Calculate stability score
        stability_score = accuracy_stability(original_accuracy, augmented_accuracy)

        # Store results for plotting
        classifier_names.append(clf_name)
        accuracies.append(original_accuracy)
        augmented_data_accuracies.append(augmented_accuracy)
        stability_scores.append(stability_score)

    stability_evaluation_df = pd.DataFrame({"Classifiers":classifier_names,
                                   "Accuracy (Original testset)":accuracies,
                                   "Accuracy (Augmented testset)":augmented_data_accuracies,
                                   "Stability Score":stability_scores})
    
    ## Identify the top stable model and the stability score
    model_stability_sorted = stability_evaluation_df.sort_values(by="Stability Score", ascending= False)
    most_stable_model = model_stability_sorted.iloc[0]["Classifiers"]
    most_stable_model_score = model_stability_sorted.iloc[0]["Stability Score"]
    most_stable_model_acc_score = model_stability_sorted.iloc[0]["Accuracy (Original testset)"]
    min_acc = model_stability_sorted["Accuracy (Original testset)"].min()

    # Plotting
    plt.figure(figsize=(6, 4))
    for i, clf_name in enumerate(model_dev.classifiers.keys()):
        plt.scatter(stability_scores[i], accuracies[i], marker='o', color=colors[i], label=clf_name)

    # Add labels and title
    plt.xlabel('Stability Score')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Stability Score')

    # Add legend
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)

    st.info("Accuracy and Stability Scores of Classifiers on Original vs. Augmented Test Sets")
    print("--------------------------------------------------------------------------------")
    st.dataframe(stability_evaluation_df)




    
    

## Module Test Function
def test_func():
    print(" sythetic_data_augmentation module is working fine")
    
#Function Testing
##############################################################################

if __name__=="__main__":
    test_func()