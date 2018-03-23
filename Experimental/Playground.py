import pandas as pd


# df = pd.read_csv('data/batter_data_all.csv')


# # id_count = 0
# # prev_id = None
# # counts = []
# # for i, row in df.iterrows():
# #     if prev_id is None:
# #         prev_id = row['mlb_id']
# #     else:
# #         if row['mlb_id'] != prev_id:
# #             counts.append(id_count)
# #             id_count = 0
# #             prev_id = row['mlb_id']
# print(df['mlb_id'].value_counts().mean())

'''
data = {1: [1,2,3], 2: [4,5,6,7,8,9]}

df = pd.DataFrame.from_dict(data, orient='index', dtype=None)
columns = [n for n in range(1,7)]
df.columns = columns
print(df.fillna(value='--'))
'''
df = pd.read_csv('../app/1_data/batter_data.csv', index_col=0)
# df = df.iloc[:, [17,18]]
# df = df[[17,18,46]]
# print(df.head())
print(df.columns)
