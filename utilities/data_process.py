import pandas as pd

from config import MAIN_CSV_PATH, FILTERED_CSV_PATH



# Main dataframe creation
df = pd.read_csv(MAIN_CSV_PATH)

# Creating a dataframe copy for a restricted area (around Paris) and transactions in 2022
filtered_df = df[(df.departement.isin([75, 77, 78, 91, 92, 93, 94, 95])) & (df.date_transaction.str.startswith('2022-'))].copy()


# Data cleaning/processing
filtered_df = filtered_df.drop('Unnamed: 0', axis=1)   # dropping useless column
filtered_df = pd.get_dummies(filtered_df, columns=['type_batiment'], dtype=int, prefix='type')   # processing 'type_batiment' into numerical values

filtered_df['date_transaction'] = pd.to_datetime(filtered_df['date_transaction'])   # processing 'transaction_date' into numerical values
reference_date = pd.to_datetime('1970-01-01')
filtered_df['days_since_epoch'] = (filtered_df['date_transaction'] - reference_date).dt.days
filtered_df = filtered_df.drop('date_transaction', axis=1)

surface_cols = [c for c in df.columns if 'surface_' in c and c != 'surface_habitable']   # cleaning dataframe from transactions with farmland, commercial spaces, etc.
for c in surface_cols:
    filtered_df[c + '_sum'] = filtered_df[c].apply(lambda x: sum(eval(x)) if 'NULL' not in x else 0)
    
filtered_df = filtered_df[filtered_df[[c + '_sum' for c in surface_cols]].sum(axis=1) == 0]

filtered_df['prix_m2'] = filtered_df['prix'] / filtered_df['surface_habitable']   # adding a 'prix_m2' column


filtered_df.to_csv(FILTERED_CSV_PATH, index=False)