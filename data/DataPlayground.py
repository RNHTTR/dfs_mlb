'''
Test ideas related to data
'''

import sys

import numpy as np
import pandas as pd

# '''
# Make use of avg points last 30
# NOTE - how can we configure this to use all data sets?
     # NOTE - ANSWER! Sort the DF by mlb_id
# df = pd.read_csv('batter_data_all.csv', index_col=0)
df = pd.read_csv('batter_data_all_2.csv')
# df = df.drop(['hand','oppt_pitch_hand'], axis=1)

row_count = df.shape[0]

start = 20170618
stop  = 20170718

current_id = None
for index, row in df.iterrows():
    if current_id == None:
        current_id = row['mlb_id']
    if row['mlb_id'] != current_id:
        avg_points_last_30 = df.loc[(df['mlb_id'] == current_id) & (df['date'] < stop) & (df['date'] >= start)]
        avg_points_last_30 = avg_points_last_30['dk_points'].mean()

        df.loc[df['mlb_id'] == current_id, 'avg_points_last_30'] = avg_points_last_30

        current_id = row['mlb_id']

    sys.stdout.write('record {} of {} records           \r'.format(index, row_count))
    sys.stdout.flush()

df = df.dropna(axis=0)
print(df)
df.to_csv('20170718_batter_data_2.csv', index=False)
# '''

# batter_df = pd.read_csv('batter_data_2015.csv', index_col=0)
# batter_df = batter_df.drop(['hand', 'oppt_pitch_hand'], axis=1)
# print(batter_df.columns)
# print(batter_df.head())
# all_df = pd.read_csv('batter_data_all.csv')
# print(all_df.head())
# df_all = pd.concat([all_df, batter_df])
# df_all = df_all.sort_values(by=['mlb_id'])
# print(df_all.tail())
# df_all.to_csv('batter_data_all_2.csv', index=False)
#
# batter_df = pd.read_csv('batter_data_all-bkp.csv', index_col=0)
# batter_df = batter_df.drop(['B', 'L_', 'R_', 'L_.1', 'R_.1'], axis=1)
# batter_df = batter_df.sort_values(by=['mlb_id'])
# batter_df.to_csv('batter_data_all.csv', index=False)

# df_2016 = pd.read_csv('batter_data_2016.csv', index_col=0)
# df_2017 = pd.read_csv('batter_data_2017.csv', index_col=0)
# df_2017 = df_2017.drop(columns=['none'])
# df_2016 = df_2016.drop(columns=['unknown'])
# df_all  = pd.concat([df_2016, df_2017])
# # print(df_all.head())
# df_all.to_csv('batter_data_all.csv')
