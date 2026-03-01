import pandas as pd
import numpy as np

def clean_data():
    # load data
    print("Loading data...")
    df = pd.read_csv('data/original/22zpallagi.csv', dtype={'zipcode': str})
    
    # remove invalid zips
    df = df[~df['zipcode'].isin(['00000', '0', 0, '99999'])]
    # pad zips
    df['zipcode'] = df['zipcode'].astype(str).str.zfill(5)
    # convert state
    df['STATE'] = df['STATE'].astype(str)
    
    # cast numerics
    numeric_cols = ['N1', 'A00100', 'A19700', 'N19700', 'agi_stub']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    # drop missing
    df = df.dropna(subset=['N1', 'A00100', 'A19700', 'N19700'])
    
    # aggregate brackets
    if 0 not in df['agi_stub'].unique():
        print("agi_stub 0 not found. Aggregating ZIP code totals from brackets 1-6...")
        df = df.groupby(['STATE', 'zipcode'], as_index=False)[['N1', 'A00100', 'A19700', 'N19700']].sum()
    else:
        df = df[df['agi_stub'] == 0][['STATE', 'zipcode', 'N1', 'A00100', 'A19700', 'N19700']]
    
    # filter positive sums
    df = df[(df['N1'] >= 100) & (df['A00100'] > 0) & (df['A19700'] > 0) & (df['N19700'] > 0)]
    
    # calc metrics
    df['generosity_index'] = df['A19700'] / df['A00100']
    df['participation_rate'] = df['N19700'] / df['N1']
    
    # constrain bounds
    df = df[df['generosity_index'].between(0, 1, inclusive='right')]
    df = df[df['participation_rate'].between(0, 1, inclusive='right')]
    
    # drop inf
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['generosity_index'])
    
    # save data
    print(f"Retained {len(df)} ZIP codes after cleaning.")
    df.to_csv('data/22zpallagi_cleaned.csv', index=False)
    print("Cleaned data exported to data/22zpallagi_cleaned.csv")

if __name__ == "__main__":
    # run cleaner
    clean_data()
