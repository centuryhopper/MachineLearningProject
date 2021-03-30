import pandas as pd




# Bin glucose and insulin
# continous ---> categorical (discrete)

# 10 features
# 116 rows/entries
df = pd.read_csv('data.csv')
# print(df.head())
# might need a labels list for glucose and insulin???
glucose_bins = [60, 88, 116, 144, 172, 201]
df['Glucose'].describe()
df['glucose_range'] = pd.cut(df['Glucose'], glucose_bins, include_lowest=True)
df['glucose_range'].isna().any()

insulin_bins = [2, 13, 24, 35, 46, 60]
# df['Insulin'].describe()
df['insulin_range'] = pd.cut(df['Insulin'], insulin_bins, include_lowest=True)
df['insulin_range'].isna().any()

# print(df)

