import pandas as pd


def preprocess_data(df):
    df = df.drop(columns=['CustomerID'])
    df['MonthlyOverTotalCharges'] = df['MonthlyCharges'] / df['TotalCharges']

    numerical_features = sorted([
        'Tenure', 'MonthlyCharges', 'TotalCharges', 'MonthlyOverTotalCharges'
    ])
    categorical_features = sorted(
        list(set(df.columns) - set(numerical_features)))

    df[numerical_features] = df[numerical_features].astype(float)
    df[categorical_features] = df[categorical_features].astype('category')

    processed_df = df[numerical_features + categorical_features]
    return processed_df


def prepare_df(data_ls):
    raw_data_df = pd.DataFrame(data_ls)
    processed_df = preprocess_data(raw_data_df)
    return processed_df