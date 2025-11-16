import numpy as np
import pandas as pd
import os


def load_data():
    data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/evaluation_set/test.csv'
    print(data_path)
    df = pd.read_csv(data_path)
    df['employment_type'].replace({'Full Time': 'Full-time', 'FULL_TIME': 'Full-time', 'Fulltime': 'Full-time', 'FT': 'Full-time',
                                   'SELF_EMPLOYED': 'Self-employed', 'Self Emp': 'Self-employed', 'Self Employed': 'Self-employed',
                                   'PART_TIME': 'Part-time', 'PT': 'Part-time', 'Part Time': 'Part-time',
                                   'Contractor': 'Contract', 'CONTRACT': 'Contract'}, inplace=True)

    df['loan_type'].replace({'Personal Loan': 'Personal', 'personal': 'Personal', 'PERSONAL': 'Personal',
                                'MORTGAGE': 'Mortgage', 'mortgage': 'Mortgage',
                                'CreditCard': 'Credit Card', 'credit card': 'Credit Card', 'CC': 'Credit Card'}, inplace=True)

    df['loan_purpose'].replace({'Mortgage': 'Home Loan'}, inplace=True)

    columns_to_fix = ['loan_amount', 'monthly_income', 'existing_monthly_debt', 'monthly_payment', 
                        'revolving_balance', 'credit_usage_amount', 'available_credit', 'total_debt_amount',
                        'monthly_free_cash_flow', 'annual_income', 'total_monthly_debt_payment']

    for col in columns_to_fix:
        df[col] = df[col].astype(str).str.replace(r"[$,]", "", regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')



    df.drop(columns=['customer_id', 'application_id', 'random_noise_1', 'recent_inquiry_count',
                    'oldest_credit_line_age', 'loan_officer_id', 'previous_zip_code', 
                    'marketing_campaign', 'referral_code', 'account_status_code', 'state',
                    'revolving_balance'], inplace=True)


    df['num_delinquencies_2yrs'].fillna(0, inplace=True)
    df['employment_length'].fillna(df['employment_length'].median(), inplace=True)

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    data_path_save = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/evaluation_set/final_test.csv'

    df.to_csv(data_path_save, index=False)
    return df

load_data()