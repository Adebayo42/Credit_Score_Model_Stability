# Import libraries
# from IPython.display import display

import gc
import warnings
import os
import time
from tqdm import tqdm
import streamlit as st

# Import program modules
import data_loading as data
import data_exploration as xplore
import missing_value_handler as mvh
import feature_engineering as feat_eng
import general_function as gen_func
import class_balancing as cb
import feature_selection as feat_select
import model_development as model_dev
import sythetic_data_augmentation as sda
import model_explainability as model_explainer

############ Web Development #################

import streamlit as st
import data_loading as data
import data_exploration as xplore
import missing_value_handler as mvh
import feature_engineering as feat_eng
import general_function as gen_func
import class_balancing as cb
import feature_selection as feat_select
import model_development as model_dev
import sythetic_data_augmentation as sda
import model_explainability as model_explainer

## Set the page layout
st.set_page_config(
    layout="wide"  # Set layout to wide
)


# Title and Introduction
st.title("Enhancing Credit Score Modelling Stability")
st.write("""
    This web application supports the research dissertation titled "Enhancing Credit Score Modelling Stability through Data Science Techniques."
    Navigate through the different sections to follow the steps of the research.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
sections = ["Introduction", "Load Data", "Exploratory Data Analysis", "Preprocessing", 
            "Model Development", "Model Stability Evaluation", "Model Explainability", "Conclusion"]
choice = st.sidebar.radio("Go to", sections)

# Section 1: Introduction
if choice == "Introduction":
    st.header("Introduction")
    st.markdown("""
This Python notebook is dedicated to the research dissertation titled \
            **"Enhancing Credit Score Modelling Stability through Data Science Techniques.\
            "** The aim of the research is to develop a robust and accurate credit score model\
             for predicting customer behaviour based on a large and imbalanced dataset. This \
            research focuses on improving model performance and explainability through effective \
            data preprocessing, synthetic data generation techniques for class balancing, feature \
            selection, evaluating model stability under augmented data conditions, and model explainability.

In this notebook, we will perform:

- **Class Balancing**: Apply the Synthetic Minority Oversampling Technique (SMOTE) to balance \
            the imbalanced credit dataset, enhancing model training and performance.
- **Feature Selection**: Use univariate ANOVA F-tests to identify and select the most \
            important features that contribute to the prediction of the target variable, \
            reducing computational complexity and enhancing model interpretability.
- **Model Development and Evaluation**: Build at least three artificial intelligent systems \
            for credit scoring models.
- **Synthetic Data Augmentation**: Generate synthetic data using specified augmentation \
            techniques to simulate real-world data variations to represent dataset shifts.
- **Model Stability Evaluation**: Evaluate the stability scores of trained models under \
            augmented data conditions to measure their resilience, robustness, and generalizability. 
- **Model Explainability**: Conduct Sobol sensitivity analysis and SHAP to determine the \
            influence of individual features on prediction outcomes and understand \
            the reasoning behind model predictions.

##### Author: F315284 | Date: August 2024
            """)
    

# Section 2: Load Data
elif choice == "Load Data":
    st.header("Load Data")
    st.write("Welcome to the Data Loading page of our research study on credit scoring modelling.\
              We use the Home Credit Default Risk dataset from Kaggle (Anna et al., 2018), which \
             includes multiple interconnected tables covering loan applications, borrower demographics, and credit histories. \
             This dataset's richness and complexity make it ideal for exploring credit scoring modelling, \
             with seven related tables offering diverse data for analysis.")
    st.image("../data/Picture.jpg")
    
    if st.button("Load Dataset"):
        data.load_files()
        st.success("Data Loaded Successfully")
    
# Section 3: Exploratory Data Analysis
elif choice == "Exploratory Data Analysis":
    st.header("Exploratory Data Analysis")
    st.write("Exploratory Data Analysis (EDA) is our initial dive into the dataset,\
              crucial for uncovering patterns, identifying anomalies, and gaining insights to guide\
              our modeling. Here, we'll examine the dataset's structure, summarize key characteristics, \
             and visualize important relationships, laying the groundwork for advanced analysis.")
    
    if st.button("Analyze Dataset Shape" ):
        st.subheader("Dataset Shape Analysis")
        xplore.dataset_shape()
        st.success("Dataset Shape Analysis Completed")

    if st.button("Analyze Target Variable"):
        st.subheader("Target Variable Analysis")
        xplore.target_variable_EDA()
        st.success("Target Variable Analysis Completed")

    if st.button("Analyze Input Variables"):
        st.subheader("Input Variable Analysis")
        xplore.input_variable_EDA()
        #st.write("Input Variable Analysis Completed")

# Section 4: Preprocessing
elif choice == "Preprocessing":
    st.header("Preprocessing Operations")
    st.markdown("""
                
Data preprocessing is crucial for effective machine learning model development. Key steps include:
-	**Data Cleaning**: Removed features with over 20% missing values, replaced invalid entries with NaN, and used median or most-frequent imputation for remaining missing values. No duplicate records were found. Categorical variables were encoded using target encoding to handle high-dimensional features.
-	**Feature Engineering**: Created 322 new features through statistical aggregations, mathematical transformations, and integration of supplementary datasets to enhance model performance.
-	**Data Splitting**: The dataset was split into 70% training and 30% test sets using a stratified approach to maintain target class distribution.
-	**Class Balancing**: Evaluated SMOTE variants, including Borderline SMOTE and KMeans SMOTE, and adopting best performing variant to address class imbalance and improve model performance by generating synthetic samples.
-	**Feature Selection**: Used F-test univariate selection to refine the feature set from 392 to the most relevant features, improving model efficiency and interpretability.

                """)
    
    if st.button("Handle Missing Values"):
        mvh.na_handler()
        st.success("Missing Values Handled")
    
    if st.button("Perform Feature Engineering"):
        st.subheader("Feature Engineering")
        feat_eng.feature_engineering()
        st.success("Feature Engineering Completed")

    # if st.button("Free Memory"):
    #     st.subheader("Optimize Memory Usage")
    #     gen_func.free_memory()
    #     st.success("Memory space cleared successfully")

    if st.button("Split Data into Train and Test"):
        st.subheader("Dataset Splitting")
        gen_func.split_train_test()
        st.success("Data Split into Train and Test Sets")

    if st.button("SMOTE Variant Evaluation"):
        st.subheader("Class Balancing: Evaluation of SMOTE Variants")
        cb.class_balancing()
        st.success("SMOTE Variants Evaluation Completed")

    if st.button("SMOTE Application on Dataset"):
        st.subheader("Class Balancing: Application of SMOTE on Dataset")
        cb.smote_application()
        st.success("SMOTE Applied")

    if st.button("Feature Selection Analysis"):
        st.subheader("Feature Selection - Optimal k-value search using RandomizedSearchCV")
        feat_select.feature_selection()
        st.success("Feature Selection completed")

    if st.button("Data Preprocessing"):
        st.subheader("Preprocessing: Data preprocessed for modelling")
        gen_func.preprocess_dataset()
        st.success("Data processing completed!!!")


# Section 5: Model Development
elif choice == "Model Development":
    st.header("Model Development and Evaluation")
    st.write("The Model Development page is where we build and refine predictive\
              models based on our processed data. This phase involves building multiple\
              models using different algorithms, training models, and evaluating \
             the performance of the model on the test dataset.")
    st.markdown("""
                
### **Model Selection**:
-	**Logistic Regression**: Simple and efficient for binary classification, providing probabilities between 0 and 1.
-	**Decision Tree**: Captures non-linear relationships and offers high interpretability through a tree-like structure.
-	**Random Forest**: An ensemble method that reduces overfitting by averaging predictions from multiple decision trees, enhancing accuracy and robustness.
-	**Histogram-based Gradient Boosting**: Uses boosting and histogram-based techniques for high accuracy and speed, especially with large datasets.
-	**Voting Ensemble**: Combines multiple classifiers using a soft voting strategy to improve overall performance and robustness.
-	**Stacking Ensemble**: Stacks predictions from various classifiers for enhanced accuracy, using a meta-learner for final predictions.
### **Model Training and Evaluation**:
Models were trained using stratified 5-fold cross-validation to ensure balanced class distribution in each fold. Performance was assessed based on average accuracy and standard deviation across folds to evaluate accuracy and stability.

                
                """)
    
    if st.button("Develop Models"):
        st.subheader("Model Development using 5-fold Cross-Validation")
        model_dev.model_development()
        st.success("Model Development Completed")

    if st.button("Evaluate Models"):
        st.subheader("Model Evaluation")
        model_dev.model_evaluation()
        st.success("Model Evaluation Completed")
    
# Section 6: Model Stability Evaluation
elif choice == "Model Stability Evaluation":
    st.header("Data Augmentation and Model Stability Evaluation")
    st.write("The Model Stability Evaluation page assesses how well our models\
              perform under real-world conditions. We test their robustness by using\
              a synthetic, augmented dataset to simulate potential shifts that may occur\
              during deployment. This ensures that our models remain reliable and \
             effective in dynamic environments.")
    st.markdown("""
### Synthetic Data Augmentation

To assess model stability under potential dataset shifts, we generated synthetic data to mimic the original dataset while introducing variability.

#### Augmentation Function
A function was used to add normally distributed noise to each feature, controlled by parameters: `mu` (mean), `sigma` (standard deviation), and `factor` (scaling factor). Synthetic datasets were created with these initial parameters.

#### Model Stability Analysis

- **Model Evaluation**: Accuracy was measured on both original and augmented datasets.

- **Stability Evaluation**: Stability scores were computed to assess performance consistency on the models on the augmented dataset.

- **Stability Matrix Plot**: Accuracy versus stability was visualized with a scatter plot, showing stability scores on the x-axis and accuracies on the y-axis. This helps in comparing classifier performance under different conditions.
""")

    
    if st.button("Sythentic Data Augmentation"):
        st.subheader("Generate synthetic augmented data and plot the distribution")
        sda.sythentic_data()
        st.success("Sythentic Data Augmentation Completed")

    if st.button("Model Stability Evaluation"):
        st.subheader("Model Stability Evaluation")
        sda.model_stability_evaluation()
        st.success("Model Stability Evaluation Completed!!!")

# Section 7: Model Explainability
elif choice == "Model Explainability":
    st.header("Model Explainability")
    st.write("The Model Explainability page focuses on understanding and interpreting \
             the most stable model from our analysis. We use Sobol sensitivity analysis\
              and SHAP values to uncover how different features influence model predictions. \
             This helps ensure transparency and provides insights into the factors driving model decisions.")
    st.markdown("""

Model interpretability is essential for transparency and understanding in credit scoring. This study employs \
                Sensitivity Analysis and SHapley Additive exPlanations (SHAP) to enhance model explainability. \
                Various SHAP visualizations were employed to clearly convey model behavior and enhance understanding in credit scoring.
                
                

#### Model Sensitivity Analysis
We computed first-order sensitivity indices to gauge how individual input features affect model output variability. Using Sobol's method (Sobol′, 2001), we assessed model robustness and performance across different conditions.

#### Shapley Additive Explanations
SHAP, a cooperative game theory approach, was used to explain feature contributions to predictions from our best-performing model. This method supports our goal of improving model transparency.


                """)
    
    if st.button("Perform Sensitivity Analysis"):
        st.subheader("Sobol Sensitivity Analysis: First order indices for Hist-GradientBoosting Decition Tree (Best Stable model)")
        model_explainer.sensitivity_analysis()
        st.success("Sensitivity Analysis Completed")

    if st.button("SHAP Explanations"):
        st.subheader("SHapley exPlanations Values: Explanations for Hist-GradientBoosting Decition Tree")
        model_explainer.shap_explaination()
        st.success("SHAP Explanations Generated")

# Section 8: Conclusion
elif choice == "Conclusion":
    st.header("Conclusion")
    st.write("In conclusion, this research advances credit scoring by developing a \
             robust model that effectively predicts customer behavior using a large, \
             imbalanced dataset. We addressed class imbalance with KMeans SMOTE and \
             identified the top 25 features through univariate ANOVA F-tests, improving \
             model interpretability and reducing complexity. We trained six supervised \
             learning models—Logistic Regression, Decision Tree, Random Forest, \
             Histogram-based Gradient Boosting, Voting Ensemble, and Stacking \
             Ensemble—using stratified 5-fold cross-validation. Model stability was \
             evaluated with synthetic augmented data to simulate real-world shifts. \
             The Histogram-based Gradient Boosting Decision Tree showed the highest stability with a score of 1.0 \
             and an accuracy of 0.89, while the Stacking Ensemble achieved the best accuracy of 0.915. All models performed well, \
             with accuracies above 0.844. Model interpretability was enhanced through Sobol sensitivity analysis and SHAP values, \
             particularly for the Histogram-based Gradient Boosting model. This deepened our understanding of feature impacts on predictions. \
             Overall, our methods proved effective in enhancing predictive performance and stability, with the Random Forest and Stacking \
             Ensemble models surpassing previous benchmarks. These approaches offer a valuable framework for future research in predictive \
             analytics and credit risk management.")




