'''
Test ideas related to data
'''

import sys

import numpy as np
import pandas as pd


df = pd.read_csv('batter_data.csv', index_col=0)
df = df.drop(['hand','oppt_pitch_hand'], axis=1)

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
df.to_csv('20170718_batter_data')
