import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from data.FormatData import one_hot

from pandas.plotting import scatter_matrix


df = pd.read_csv('data/raw_rotoguru_data.csv', index_col=0)
df = df[['fd_pos', 'date', 'dk_points']]
df = df.loc[df['date'] == 20170718]
positions = df.fd_pos.unique().tolist()
pos_map = {
    1: "P",
    2: "C",
    3: '1B',
    4: '2B',
    5: '3B',
    6: 'SS',
    7: 'OF'
}
for pos in positions:
    total = df['dk_points'].sum()
    if pos in pos_map:
        pos_sum = int(df.loc[df['fd_pos'] == pos, ['dk_points']].sum())
        print('{}:   {}   -- {:.1%}'.format(pos_map[pos], pos_sum, pos_sum / total))
# x = df['dk_salary']
# y = df['dk_points']
# df_sub = df[['avg', 'dk_salary', 'fd_salary', 'adi', 'dk_points']]
# df_sub = df[['order','w_speed','dk_points']]
# cols = ['pos']
# cat = ['pos']
# # df_sub = one_hot(df, cols, cat)
# # df_sub = df['pos']
# df = df.loc[df['dk_points'] > 0]
# plt.figure(figsize=(16,8))
# ax1 = plt.subplot(121, aspect='equal')
# df.plot(kind='pie', y = 'dk_points', ax=ax1, autopct='%1.1f%%',
#  startangle=90, shadow=False, labels=df['pos'], legend = False, fontsize=14)

# plt.pie(x=df_sub)

# df_sub.hist()
plt.show()

# df_sub = df_sub.sample(n=100, random_state=7)
# scatter_matrix(df_sub)
# plt.show()
