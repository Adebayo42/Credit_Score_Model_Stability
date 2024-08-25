# Feature Engineering Module

# This Python module contains functions for performing feature engineering
# on the application train dataset and supplementary datasets.

# Writen By: F315284
# Date: August, 2024

import data_loading as data
import missing_value_handler as mvh
import pandas as pd
import numpy as np
from stqdm import stqdm
import gc
import streamlit as st

# Function 0: Feature Engineering - application dataset
##############################################################################

def application_feature(df):
    """
    Perform feature engineering on the application dataset.

    This function adds aggregated features and handcrafted features to the input dataframe.

    Parameters:
    df (pd.DataFrame): Input dataframe containing application data.

    Returns:
    pd.DataFrame: Dataframe with new engineered features.
    """
    index_name = 'SK_ID_CURR'
    
    
    ## for the below aggregation, the idea and code was adapted from James Dellinger Solution 
    ## at: https://www.kaggle.com/code/jamesdellinger/home-credit-putting-all-the-steps-together/notebook
    
    # Sample aggregation recipes
    aggregation_recipes = [
        # Aggregation by gender and education type
        (['CODE_GENDER', 'NAME_EDUCATION_TYPE'], [
            ('AMT_CREDIT', 'sum'), ('AMT_CREDIT', 'mean'), 
            ('AMT_ANNUITY', 'mean'), ('AMT_ANNUITY', 'sum'), 
            ('EXT_SOURCE_3', 'mean'), ('EXT_SOURCE_3', 'sum'), 
            ('EXT_SOURCE_2', 'mean'), ('EXT_SOURCE_2', 'sum')
        ]),

        # Aggregation by income type and organization type
        (['NAME_INCOME_TYPE', 'ORGANIZATION_TYPE'], [
            ('AMT_ANNUITY', 'mean'), ('AMT_ANNUITY', 'sum'), 
            ('AMT_INCOME_TOTAL', 'mean'), ('AMT_INCOME_TOTAL', 'sum'), 
            ('EXT_SOURCE_3', 'median'), ('EXT_SOURCE_2', 'mean'), 
            ('EXT_SOURCE_3', 'mean'),('AMT_CREDIT', 'mean')
        ]),

        # Aggregation by family status and city work status
        (['NAME_FAMILY_STATUS', 'REG_CITY_NOT_WORK_CITY'], [
            ('AMT_ANNUITY', 'mean'), ('CNT_CHILDREN', 'mean'), 
            ('AMT_ANNUITY', 'sum')
        ]), 

        # Aggregation by contract type, education type, income type, and city work status
        (['NAME_CONTRACT_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE', 'REG_CITY_NOT_WORK_CITY'], [
            ('EXT_SOURCE_3', 'median'), ('EXT_SOURCE_2', 'mean'), ('EXT_SOURCE_3', 'mean')
        ]),


        # Aggregation by income type
        (['NAME_INCOME_TYPE'], [
            ('AMT_ANNUITY', 'mean'), ('CNT_CHILDREN', 'mean'), ('DAYS_EMPLOYED', 'mean'), 
            ('DAYS_BIRTH', 'mean'), ('DAYS_ID_PUBLISH', 'mean'), 
            ('DAYS_REGISTRATION', 'mean'), ('EXT_SOURCE_3', 'median'), 
            ('EXT_SOURCE_2', 'mean'), ('EXT_SOURCE_3', 'mean')
        ])
    ]

    # Reset index
    if df.index.name == index_name:
        df = df.reset_index()

    # Generate aggregated features based on the aggregation recipes
    for groupby_cols, spec in aggregation_recipes:
        groupby_object = df.groupby(groupby_cols)

        for feature, aggr in spec:
            groupby_aggregation_name = '{}_{}_{}'.format('_'.join(groupby_cols), aggr.upper(), feature)
            diff_stat = '{}_DIFF'.format(groupby_aggregation_name)
            abs_diff_stat = '{}_ABS_DIFF'.format(groupby_aggregation_name)

            aggregated_df = groupby_object[feature].agg(aggr).reset_index().rename(columns={feature: groupby_aggregation_name})
            df = pd.merge(df, aggregated_df, on=groupby_cols, how='left')

            # Calculate difference statistics
            df[diff_stat] = df[feature] - df[groupby_aggregation_name]
            df[abs_diff_stat] = np.abs(df[diff_stat])

            # Clean up memory
            del aggregated_df
            gc.collect()

    # Handcrafted engineered features
    df['NEW_INC_PER_CHLD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
    df['NEW_EMPLOY_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['NEW_SOURCES_PROD'] = df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['NEW_EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['NEW_SCORES_STD'] = df[['EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    df['NEW_SCORES_STD'] = df['NEW_SCORES_STD'].fillna(df['NEW_SCORES_STD'].mean())
    df['EXT_SOURCES_WEIGHTED_SUM'] = df['EXT_SOURCE_3'] * 5 + df['EXT_SOURCE_2'] * 1
    df['EXT_SOURCES_WEIGHTED_AVG'] = (df['EXT_SOURCE_3'] * 5 + df['EXT_SOURCE_2'] * 1) / 2
    df['NEW_PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
    df['NEW_PHONE_TO_EMPLOYED_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
    df['NEW_CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['AMT_PAY_YEAR'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['AGE_PAYOFF'] = -df['DAYS_BIRTH'] / 365.25 + df['AMT_PAY_YEAR']
    df['AMT_ANNUITY_INCOME_RATE'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['AMT_DIFF_CREDIT_GOODS'] = df['AMT_CREDIT'] - df['AMT_GOODS_PRICE']
    df['AMT_CREDIT_GOODS_PERC'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['AGE_EMPLOYED'] = df['DAYS_EMPLOYED'] - df['DAYS_BIRTH']
    df['AMT_INCOME_OVER_CHILD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
    df['CNT_ADULT'] = df['CNT_FAM_MEMBERS'] - df['CNT_CHILDREN']

    return df.set_index(index_name)



# Function 1: Feature engineering - Bureau dataset
############################################################################

def bureau_feature(df):
    """
    Perform feature engineering on the bureau dataset.

    This function adds aggregated features and handcrafted features to the input dataframe based on the bureau data.

    Parameters:
    df (pd.DataFrame): Input dataframe containing bureau data.

    Returns:
    pd.DataFrame: Dataframe with new engineered features.
    """
    index_name = 'SK_ID_CURR'
    
    # Reset index if necessary
    if df.index.name == index_name:
        df = df.reset_index()
    
    # Bureau Balance transformation
    bureau_bal_ = data.files_dict["bureau_balance"].copy()
    # Replace STATUS values with corresponding DPD values
    bureau_bal_['DPD'] = bureau_bal_['STATUS'].replace(['C', 'X', '0', '1', '2', '3', '4', '5'],
                                                       [0, 0, 0, 30, 60, 90, 120, 180])
    # Reset index of bureau dataset
    bureau_v = data.files_dict["bureau"].reset_index()
    
    # Merge bureau balance data with bureau data on SK_ID_BUREAU
    bureau_ = bureau_v.merge(
        bureau_bal_.groupby('SK_ID_BUREAU')['DPD'].agg('mean').reset_index(),
        on='SK_ID_BUREAU', 
        how='left'
    ).set_index('SK_ID_CURR')
    
    # Define aggregation recipes
    aggregate_recipes = [('SK_ID_BUREAU', 'count')]
    for aggr in ['mean', 'sum', 'max', 'min', 'median']:
        for feature in [
            'AMT_CREDIT_MAX_OVERDUE', 'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT',
            'AMT_CREDIT_SUM_LIMIT', 'AMT_CREDIT_SUM_OVERDUE', 'AMT_ANNUITY',
            'DAYS_CREDIT', 'CREDIT_DAY_OVERDUE', 'DPD'
        ]:
            aggregate_recipes.append((feature, aggr))
    
    aggregate_recipes = [('SK_ID_CURR', aggregate_recipes)]
    
    # Apply aggregation recipes
    for groupby_col, spec in aggregate_recipes:
        groupby_object = bureau_.groupby(groupby_col)
        
        for feature, aggr in spec:
            groupby_aggregate_name = '{}_{}'.format(aggr.upper(), feature)
            
            # Merge the aggregated feature back to the original dataframe
            df = df.merge(
                groupby_object[feature].agg(aggr).reset_index()
                .rename(columns={feature: groupby_aggregate_name})
                [[groupby_col, groupby_aggregate_name]], 
                on=groupby_col, 
                how='left'
            )
    
    return df.set_index('SK_ID_CURR')

# Function 2: Feature engineering - Previous Application
############################################################################

def previous_feature(df):
    """
    Perform feature engineering on the previous application dataset.

    This function adds aggregated features to the input dataframe based on previous application data.

    Parameters:
    df (pd.DataFrame): Input dataframe containing previous application data.

    Returns:
    pd.DataFrame: Dataframe with new engineered features.
    """
    # Reset index to ensure 'SK_ID_CURR' is not the index
    df = df.reset_index()
    
    # Replace 'XNA' and 'XAP' with NaN in the previous_application dataset
    prev_appl = data.files_dict["previous_application"].replace(to_replace={'XNA': np.nan, 'XAP': np.nan})
    
    # Define initial aggregation recipes
    aggregate_recipes = [('SK_ID_PREV', 'count')]
    
    # Add aggregation recipes for various statistical measures
    for aggr in ['sum', 'min', 'max', 'mean', 'median']:
        for feature in [
            'AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_GOODS_PRICE',
            'HOUR_APPR_PROCESS_START', 'DAYS_DECISION', 'DAYS_FIRST_DRAWING',
            'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE',
            'DAYS_TERMINATION', 'AMT_DOWN_PAYMENT'
        ]:
            aggregate_recipes.append((feature, aggr))
    
    aggregate_recipes = [('SK_ID_CURR', aggregate_recipes)]
    
    # Apply aggregation recipes
    for groupby_col, spec in aggregate_recipes:
        groupby_object = prev_appl.groupby(groupby_col)
        
        for feature, aggr in spec:
            groupby_aggregate_name = '{}_{}'.format(aggr.upper(), feature)
            
            # Merge the aggregated feature back to the original dataframe
            df = df.merge(
                groupby_object[feature].agg(aggr).reset_index()
                .rename(columns={feature: groupby_aggregate_name})
                [[groupby_col, groupby_aggregate_name]], 
                on='SK_ID_CURR', 
                how='left'
            )
    
    return df.set_index('SK_ID_CURR')


# Function 3: Feature engineering - Credit Card data
############################################################################
def credit_card_feature(df):
    """
    Perform feature engineering on the credit card balance dataset.

    This function adds aggregated features to the input dataframe based on credit card balance data.

    Parameters:
    df (pd.DataFrame): Input dataframe containing credit card balance data.

    Returns:
    pd.DataFrame: Dataframe with new engineered features.
    """
    # Reset index to ensure 'SK_ID_CURR' is not the index
    df = df.reset_index()
    
    # Define initial aggregation recipes
    aggregate_recipes = [('SK_ID_PREV', 'count')]
    
    # Add aggregation recipes for various statistical measures
    for aggr in ['mean', 'median', 'sum', 'max', 'min']:
        for feature in [
            'AMT_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_DRAWINGS_ATM_CURRENT',
            'AMT_DRAWINGS_CURRENT', 'AMT_DRAWINGS_OTHER_CURRENT', 'AMT_DRAWINGS_POS_CURRENT',
            'AMT_INST_MIN_REGULARITY', 'AMT_PAYMENT_CURRENT', 'AMT_PAYMENT_TOTAL_CURRENT',
            'AMT_RECEIVABLE_PRINCIPAL', 'AMT_RECIVABLE', 'AMT_TOTAL_RECEIVABLE',
            'CNT_DRAWINGS_ATM_CURRENT', 'CNT_DRAWINGS_CURRENT', 'CNT_DRAWINGS_OTHER_CURRENT',
            'CNT_DRAWINGS_POS_CURRENT', 'CNT_INSTALMENT_MATURE_CUM', 'SK_DPD', 'SK_DPD_DEF'
        ]:
            aggregate_recipes.append((feature, aggr))
    
    aggregate_recipes = [('SK_ID_CURR', aggregate_recipes)]
    
    # Apply aggregation recipes
    for groupby_col, spec in aggregate_recipes:
        groupby_object = data.files_dict["credit_card_balance"].groupby(groupby_col)
        
        for feature, aggr in spec:
            groupby_aggregate_name = '{}_{}'.format(aggr.upper(), feature)
            
            # Merge the aggregated feature back to the original dataframe
            df = df.merge(
                groupby_object[feature].agg(aggr).reset_index()
                .rename(columns={feature: groupby_aggregate_name})
                [[groupby_col, groupby_aggregate_name]], 
                on='SK_ID_CURR', 
                how='left'
            )
    
    return df.set_index('SK_ID_CURR')


# Function 4: Feature engineering - Installment payments and POS Balance
############################################################################

def installment_pos_bal_feature(df):
    """
    Perform feature engineering on the installments payments and POS_CASH_balance dataset.

    This function adds aggregated features to the input dataframe based on installments payments data
    and merges aggregated features from the POS_CASH_balance table.

    Parameters:
    df (pd.DataFrame): Input dataframe containing installments payments data.

    Returns:
    pd.DataFrame: Dataframe with new engineered features.
    """
    # Reset index to ensure 'SK_ID_CURR' is not the index
    df = df.reset_index()
    
    # Create a copy of the installments_payments dataframe
    installments_payments_ = data.files_dict["installments_payments"].copy()
    # Calculate difference between actual payment date and expected payment date
    installments_payments_['DAYS_INSTALMENT_DIF'] = installments_payments_['DAYS_ENTRY_PAYMENT'] - installments_payments_['DAYS_INSTALMENT']
    # Calculate difference between installment amount and paid amount
    installments_payments_['AMT_INSTALMENT_DIF'] = installments_payments_['AMT_INSTALMENT'] - installments_payments_['AMT_PAYMENT']
    
    # Define initial aggregation recipes
    aggregate_recipes = []
    
    # Add aggregation recipes for mean and sum of the difference features
    for aggr in ['mean', 'sum']:
        for feature in ['DAYS_INSTALMENT_DIF', 'AMT_INSTALMENT_DIF']:
            aggregate_recipes.append((feature, aggr))
    
    aggregate_recipes = [('SK_ID_CURR', aggregate_recipes)]
    
    # Apply aggregation recipes
    for groupby_col, spec in aggregate_recipes:        
        groupby_object = installments_payments_.groupby(groupby_col)
        
        for feature, aggr in spec:
            groupby_aggregate_name = '{}_{}'.format(aggr.upper(), feature)
            
            # Merge the aggregated feature back to the original dataframe
            df = df.merge(
                groupby_object[feature].agg(aggr).reset_index()
                .rename(columns={feature: groupby_aggregate_name})
                [[groupby_col, groupby_aggregate_name]], 
                on='SK_ID_CURR', 
                how='left'
            )
    
    # Merge aggregated features from the POS_CASH_balance table
    df = df.merge(
        data.files_dict["POS_CASH_balance"].groupby('SK_ID_CURR')
        .agg(
            CNT_INSTALMENT_MEAN=('CNT_INSTALMENT', 'mean'),
            CNT_INSTALMENT_FUTURE_MEAN=('CNT_INSTALMENT_FUTURE', 'mean'),
            SK_DPD_DEF_MEAN=('SK_DPD_DEF', 'mean'),
            SK_DPD_MEAN=('SK_DPD', 'mean')
        ).reset_index(), 
        on='SK_ID_CURR',
        how='left'
    )
    
    return df.set_index('SK_ID_CURR')


# Function 5: Feature engineering 
############################################################################

def feature_engineering():
    """
    Perform feature engineering on the input dataframe by sequentially applying
    various feature generation functions.

    Parameters:
    df (pd.DataFrame): Input dataframe containing application data.

    Returns:
    pd.DataFrame: Dataframe with engineered features.
    """
    
    # assign the dataframe from cleaned dataframed processed b missing_value_handler as mvh
    df= mvh.X_input
    # Create global variable
    global dataset
    # List of feature functions
    feature_functions = [
        application_feature,
        bureau_feature,
        previous_feature,
        credit_card_feature,
        installment_pos_bal_feature,
    ]

    # Iterate through the feature functions with a progress bar
    for feature_func in stqdm(feature_functions, desc="Generating Features"):
        try:
            df = feature_func(df)  # Apply each feature function
        except Exception as e:
            print(f"Error occurred while processing {feature_func.__name__}: {e}")
            
            return df

    dataset = df
    st.info("Display Top 20rows of the dataset")
    return st.dataframe(df.head(20))  # Return the final dataframe


## Module Test Function
def test_func():
    print(" feature_engineering module is working fine")
    
#Function Testing
##############################################################################

if __name__=="__main__":
    test_func()
    