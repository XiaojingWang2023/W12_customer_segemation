# load_data.py

import pandas as pd

def load_data(data_path):

    # Import the data from 'mall_customers.csv'
    df = pd.read_csv(data_path)

    return df
