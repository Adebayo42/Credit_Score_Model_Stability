# This Python module contains functions used for model development


# Writen By: F315284
# Date: August, 2024

# Importing required libraries
import numpy as np
import pandas as pd
from stqdm import stqdm
from sklearn.ensemble import RandomForestClassifier,VotingClassifier,StackingClassifier,GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import general_function as gen_func
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, ConfusionMatrixDisplay,confusion_matrix
import streamlit as st


# Function 0: Model Development
##############################################################################
classifiers = {}

# @st.cache_resource
def model_development():
    """
    Develop and evaluate multiple classifiers using cross-validation.

    This function initializes several classifiers, including individual models and 
    ensemble methods (voting and stacking), and performs cross-validation to evaluate 
    their accuracy on the training dataset.

    Returns:
        None
    """
    
    ## Model Development
    # declare global variables
    # global classifiers

    # Initialize the single classifiers
    rf_clf = RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=-1)
    lg_clf = LogisticRegression(solver='sag',max_iter=1000,random_state=42, n_jobs=-1)
    dt_clf = DecisionTreeClassifier(random_state=42)
    hgbt_clf = HistGradientBoostingClassifier(random_state=42)

    # Voting Ensemble using Hard voting strategy
    vt_clf = VotingClassifier(estimators=[
        ('logistic', lg_clf), 
        ('random_forest', rf_clf), 
        ('decision_tree', dt_clf)],
        voting='soft',
        n_jobs=-1  
    )

    # Stacking Ensemble
    st_clf = StackingClassifier(estimators=[
        ('logistic', lg_clf), 
        ('random_forest', rf_clf), 
        ('decision_tree', dt_clf)],
        final_estimator=GradientBoostingClassifier(random_state=42),
        n_jobs=-1  
    )
    
   
    # Create Dictionary for the classifiers
    classifier = dict(zip(['RandomForest', 'LogisticRegression', 'DecisionTree',
                            'HistGradientBoosting', 'VotingEnsemble', 'StackingEnsemble'],
                           [rf_clf, lg_clf, dt_clf, hgbt_clf, vt_clf, st_clf]))


    # Define a cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    classifier_name_cv = []
    score_cv = []
    # Iterate cross-validation
    with stqdm(desc="Training Models", total=6) as pbar:
        for  label,clf in classifier.items():
            # Perform cross-validation
            scores = cross_val_score(clf, gen_func.X_train_processed_df, gen_func.y_train_processed.values.ravel(), cv=cv, n_jobs=-1)
            classifier_name_cv.append(label)
            score_cv.append(f"{scores.mean():.4f} ± {scores.std():.4f}")
            #print(f'Accuracy of {label} is: {scores.mean():.4f} ± {scores.std():.4f}')  # Mean and std of accuracy

            # Fit the classifier on the entire dataset
            clf.fit(gen_func.X_train_processed_df, gen_func.y_train_processed.values.ravel())
            classifiers[label] = clf
            pbar.update(1)
    
    st.info("Model Development: Cross-Validation Results on Training Set")
    print("-------------------------------------------------------------\n")
    cv_results = pd.DataFrame({"Classifier":classifier_name_cv, "Accuracy":score_cv})
    st.dataframe(cv_results)

    # Process Completion Message
    print(f"\nStep 10 completed!!!\n")
    
    
# Function 1: Model Evaluation
##############################################################################
def model_evaluation():
    """
    Evaluate classifiers on test data, visualize confusion matrices, 
    and consolidate classification reports into a DataFrame.

    Globals:
        classifiers (dict): Trained classifiers.
        X_test_processed_df (DataFrame): Processed test features.
        y_test (Series): True target values.

    Returns:
        None
    """
    
    ### Model Evaluation on Test Data

    # Initialize the plot
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))  
    axes = axes.flatten()  

    classification_reports = []

    for idx, (label, clf) in (enumerate(classifiers.items())):
        # Predict using the classifier
        prediction = clf.predict(gen_func.X_test_processed_df)
        classification_report_ = classification_report(gen_func.y_test, prediction, output_dict=True)  

        # Convert the report to a DataFrame and add a column for the classifier label
        report_df = pd.DataFrame(classification_report_).T
        report_df['model'] = label

        # Filter out 'macro avg' and 'weighted avg'
        report_df = report_df[~report_df.index.isin(['macro avg', 'weighted avg'])]

        # Drop the 'support' column
        report_df.drop(columns='support', inplace=True)

        # append the report for each classifier to the list
        classification_reports.append(report_df)

        ## Display Confusion Matrix
        cm = confusion_matrix(gen_func.y_test, prediction)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=None) 
        disp.plot(ax=axes[idx])  
        axes[idx].set_title(f'Confusion Matrix for {label}',fontsize=8)
        axes[idx].grid(False)
        

    # Adjust layout
    plt.tight_layout()
    st.pyplot(plt)
    #plt.gca().grid(False)

    # Concatenate all DataFrames into a single DataFrame
    classification_report_df = pd.concat(classification_reports)

    # Reset index for better display
    classification_report_df.reset_index(inplace=True)
    classification_report_df.rename(columns={'index': 'class'}, inplace=True)

    # Display the DataFrame
    st.info("### Model Evaluation Report on Test Data ###")
    st.write(classification_report_df)

    
# Function 2: Model Stability Evaluation
##############################################################################    
def model_stability_evaluation():
    """
    Evaluate the stability of classifiers by comparing 
    their accuracy on original and augmented test data.

    Returns:
        None
    """
    
    ## Model Stability Evaluation

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
    for i, (clf_name, clf) in stqdm(enumerate(classifiers.items())):
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

    # Plotting
    plt.figure(figsize=(6, 4))
    for i, clf_name in enumerate(classifiers.keys()):
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

    st.subheader("Accuracy and Stability Scores of Classifiers on Original vs. Augmented Test Sets")
    print("--------------------------------------------------------------------------------")
    st.dataframe(stability_evaluation_df)
    
    
    
## Module Test Function
def test_func():
    print(" model_development module is working fine")
    
#Function Testing
##############################################################################

if __name__=="__main__":
    test_func()  