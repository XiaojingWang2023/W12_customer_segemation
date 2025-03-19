
# build_features.py
import pandas as pd

# create dummy features
def create_dummy_vars(df):
    
    # Create dummy variables for ‘Gender' column
    df = pd.get_dummies(df, columns=['Gender'], dtype=int)
 
    # Drop 'Customer_ID' because no need for clustering
    df = df.drop('Customer_ID', axis=1)
    
    # store the processed dataset in data/processed
    df.to_csv('data/processed/processed_mallCustomers.csv', index=None)

    return df
