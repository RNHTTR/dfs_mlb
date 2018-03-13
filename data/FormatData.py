import requests
from io import StringIO

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

# from GetData import get_rotoguru_data
from GetData import read_fangraphs_pitcher_data
# from OpponentPitcherData import read_pitcher_data


def one_hot(df, columns, cat_columns):
    '''
    One hot encode categorical variables
    '''
    df = df[columns]
    for col in cat_columns:
        one_hot = pd.get_dummies(df[col])
        df = df.drop(col, axis=1)
        df = df.join(one_hot)

    return df


def player_data(position, columns, cat_cols):
    '''
    Args:
        position (str)      : P for pitchers, H for hitters
        columns  (list[str]): All columns to be read into the DataFrame
        cat_cols (list[str]): Categorical columns that need to be one-hot encoded.

    Returns:
        df (Pandas DataFrame): Player data for batters or pitchers
    '''
    position = position.upper()

    assert position in ['P', 'H'], \
        "Position must be given as a single letter, P for pitchers or H for " \
        "hitters."

    # df = get_rotoguru_data()
    df = pd.read_csv('raw_rotoguru_data.csv')
    df = one_hot(df, columns, cat_cols)
    df = df.loc[df['p/h'] == position]
    df = df.drop('p/h', axis=1)

    fangraphs_pitcher_df = read_fangraphs_pitcher_data()

    if position == 'H':
        df = df.merge(fangraphs_pitcher_df, on='oppt_pitch_name', how='left')
        df = df.drop('oppt_pitch_name', axis=1)
    else:
        fangraphs_pitcher_df.rename(index=str, columns={"oppt_pitch_name": "name_last_first"}, inplace=True)
        df = df.merge(fangraphs_pitcher_df, on='name_last_first')
        # df = df.drop('name_last_first', axis=1)

    df = df.dropna(axis=0)

    return df


if __name__ == '__main__':
    '''
    '''
    batter_columns=['p/h','mlb_id','date','hand','oppt_pitch_hand','condition','adi',
                    'order','w_speed','w_dir','oppt_pitch_name','dk_salary',
                    'fd_salary','fd_pos','dk_points']
    batter_cat_cols=['condition','w_dir']
    # #
    batter_df = player_data("h", batter_columns, batter_cat_cols)
    batter_df['hand_advantage'] = np.where(batter_df['hand'] == batter_df['oppt_pitch_hand'], 0, 1)
    batter_df.to_csv('batter_data.csv')
    # df = pd.read_csv('batter_data_full.csv', index_col=0)
    # df = df.drop('hand', axis=1)
    # df = df.drop('oppt_pitch_hand', axis=1)
    # print(df)
    # # df['hand_advantage'] = np.where(df['hand'] == df['oppt_pitch_hand'], 0, 1)
    # df.to_csv('batter_data_full.csv')
    # print(df['hand_advantage'])

    # pitcher_columns=['p/h','name_last_first','mlb_id','condition','adi','w_speed','w_dir','dk_salary','fd_salary','dk_points']
    # # pitcher_columns=['condition','adi','w_speed','w_dir','dk_salary','fd_salary','dk_points']
    # pitcher_cat_cols=['condition', 'w_dir']
    # pitcher_df = player_data("p", pitcher_columns, pitcher_cat_cols)
    # # # pitcher_df = pd.read_csv('pitcher_data.csv')
    # # # pitcher_df = one_hot(pitcher_df, pitcher_columns, pitcher_cat_cols)
    # pitcher_df.to_csv('pitcher_data.csv')
