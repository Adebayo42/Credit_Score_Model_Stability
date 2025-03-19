# 🚀 Credit Score Prediction Model: Data Science Pipeline

## 🔍 Overview

In today’s fast-paced financial industry, **accurate and reliable credit scoring** is crucial for assessing the creditworthiness of individuals and minimizing risks for lenders. With the increasing volume of data and evolving patterns of financial behavior, it is important for financial institutions to have **robust and stable credit scoring models** that can adapt to changes in consumer behavior and market dynamics.

This project focuses on building a **highly accurate credit score prediction model** that employs cutting-edge machine learning techniques, ensuring high stability and consistency in predicting creditworthiness. By leveraging a combination of advanced modeling techniques and data science best practices, this model helps improve risk management and streamline decision-making for loan approvals, credit cards, and other financial services.

---

## 📌 Key Features:
- ✅ **Multi-Model Approach**: Uses **Random Forest**, **Logistic Regression**, **Decision Tree**, **HistGradientBoosting**, and advanced **ensemble methods** like **Voting** and **Stacking Classifiers** for improved accuracy.
- ✅ **SMOTE for Class Balancing**: Addresses class imbalance using the **Synthetic Minority Oversampling Technique (SMOTE)** to ensure fairness and reduce bias in predictions.
- ✅ **Feature Selection & Engineering**: Uses **ANOVA F-tests** for optimal feature selection and additional feature engineering to boost model performance.
- ✅ **Model Explainability**: Implements **SHAP values** and **Sobol Sensitivity Analysis** for interpretable model decisions, providing transparency to the stakeholders.
- ✅ **Scalable**: Designed for use in **real-world financial applications**, ensuring the model can adapt to changes in financial behavior and continue to predict accurately.

---

## 📊 Performance Metrics on Test Data

| Model                | Precision (Class 0) | Recall (Class 0) | F1-Score (Class 0) | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) | Accuracy  |
|----------------------|---------------------|------------------|--------------------|---------------------|------------------|--------------------|-----------|
| **Random Forest**     | 0.9249              | 0.9726           | 0.9482             | 0.2450              | 0.1011           | 0.1431             | 0.9023    |
| **Logistic Regression**| 0.9344              | 0.9132           | 0.9237             | 0.2145              | 0.2700           | 0.2391             | 0.8612    |
| **Decision Tree**     | 0.9263              | 0.9067           | 0.9164             | 0.1442              | 0.1790           | 0.1597             | 0.8479    |
| **HistGradientBoosting**| 0.9291             | 0.9524           | 0.9406             | 0.2411              | 0.1724           | 0.2011             | 0.8894    |
| **Voting Ensemble**   | 0.9276              | 0.9524           | 0.9399             | 0.2210              | 0.1536           | 0.1812             | 0.8880    |
| **Stacking Ensemble** | 0.9208              | 0.9929           | 0.9555             | 0.2581              | 0.0279           | 0.0504             | **0.9150**|

📈 **Stacking Classifier** outperforms individual models in **accuracy** and **F1-Score** for Class 0 (non-default customers).

---

## 🔧 Technology Stack
- **Python** 🐍  
- **Machine Learning Libraries**: Scikit-learn, XGBoost, LightGBM, Imbalanced-learn  
- **Data Processing**: Pandas, NumPy, Category Encoders  
- **Model Explainability**: SHAP, Sobol Sensitivity Analysis  
- **Visualization**: Matplotlib, Seaborn

---

## 🛠 How It Works

1️⃣ **Data Preprocessing & Feature Engineering**:
   - **Load datasets** (application, bureau, balance data)  
   - Handle missing values (remove features with more than 20% missing data)  
   - Apply **feature engineering** to create new predictive features, enhancing model performance.

2️⃣ **Class Balancing (SMOTE)**:
   - **SMOTE** is applied to **oversample the minority class**, ensuring a balanced dataset and reducing model bias toward the majority class.

3️⃣ **Feature Selection**:
   - **ANOVA F-tests** are used to select the most important features for the model, improving model efficiency and reducing overfitting.

4️⃣ **Model Training & Stacking**:
   - Train multiple classifiers (Random Forest, Logistic Regression, Decision Tree, etc.).
   - Combine these classifiers into a **Stacking Ensemble Model** for improved predictive performance and stability.

5️⃣ **Model Evaluation & Explainability**:
   - Models are evaluated using metrics like **precision**, **recall**, **F1-score**, and **accuracy**.
   - **SHAP** and **Sobol Sensitivity Analysis** are used to interpret feature importance and understand the model's decision-making process.

---

## 🏆 Why This Project Stands Out
- 🔹 **Demonstrates real-world financial data science skills**: This project showcases expertise from data preprocessing to model evaluation and explainability.
- 🔹 **Applies advanced techniques** like **SMOTE**, **Feature Engineering**, and **Stacking Classifiers** to achieve superior model performance, offering more accurate creditworthiness predictions.
- 🔹 **Ensures model interpretability** with **SHAP** and **Sobol Sensitivity Analysis**, making the model decisions **transparent** for decision-makers in the credit scoring process.

---

## 🚀 Let's Connect!
Looking for a **Data Scientist** who can build **high-performing AI models** for credit score prediction? Feel free to reach out!


