import pandas as pd


df = pd.read_csv('../app/1_data/raw_X_batter.csv', index_col=0)

df = df[['dk_salary', 'fd_salary']]
df = df.dropna()
df['diff'] = (df['dk_salary'] - df['fd_salary'])
print(df['diff'].mean())
# print(df)
print(max(df['dk_salary']))
print(min(df['dk_salary']))
print(max(df['fd_salary']))
print(min(df['fd_salary']))
print(df['dk_salary'].mean() - df['fd_salary'].mean())
print(df['diff'].mean())
