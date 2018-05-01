import sys

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

# from GetData import get_rotoguru_data
from GetData import read_fangraphs_pitcher_data
# from OpponentPitcherData import read_pitcher_data

# QUESTION - Should we consolidate GetData.py, FormatData.py, and PointsLast30.py
#            into one file and use classes?

def one_hot(df, columns, cat_cols):
    '''
    One hot encode categorical variables

    Args:
        df (Pandas DF)      : Player data
        columns  (list[str]): All columns to be read into the DataFrame
        cat_cols (list[str]): Categorical columns that need to be one-hot encoded

    Returns:
        df (Pandas DF): Player data w/categorical features one-hot encoded
    '''
    df = df[columns]
    for col in cat_cols:
        one_hot = pd.get_dummies(df[col])
        df = df.drop(col, axis=1)
        df = df.join(one_hot)

    return df


def player_data(data_file_path, position, columns, cat_cols):
    '''
    Args:
        position (str)      : P for pitchers, H for hitters
        columns  (list[str]): All columns to be read into the DataFrame
        cat_cols (list[str]): Categorical columns that need to be one-hot encoded

    Returns:
        df (Pandas DF): Player data for batters or pitchers
    '''
    position = position.upper()

    assert position in ['P', 'H'], \
        "Position must be given as a single letter, P for pitchers or H for " \
        "hitters."

    # NOTE - This should be done in if __name__ block and passed as a param...
    # df = get_rotoguru_data()
    df = pd.read_csv(data_file_path)
    df = one_hot(df, columns, cat_cols)
    df = df.loc[df['p/h'] == position]
    df = df.drop('p/h', axis=1)

    fangraphs_pitcher_df = read_fangraphs_pitcher_data()

    if position == 'H':
        df = df.merge(fangraphs_pitcher_df, on='oppt_pitch_name', how='left')
        df = df.drop('oppt_pitch_name', axis=1)
        df['hand_advantage'] = np.where(df['hand'] == df['oppt_pitch_hand'], 0, 1)
        df = df.drop(['hand', 'oppt_pitch_hand'], axis=1)
    else:
        fangraphs_pitcher_df.rename(index=str, columns={"oppt_pitch_name": "name_last_first"}, inplace=True)
        df = df.merge(fangraphs_pitcher_df, on='name_last_first')

    df['dk_salary'] = df['dk_salary'].fillna(df['fd_salary'])
    df['fd_salary'] = df['fd_salary'].fillna(df['dk_salary'])

    df = df.dropna(axis=0)

    return df


def main(data_file_path, position, config, output_file_path):
    '''
    '''
    # TODO - Creat config yaml.
    columns = config['columns']
    categorical_columns = config['cat_cols']
    df = player_data(data_file_path, position, columns, categorical_columns)
    # df['w_speed'] = df['w_speed'].map(lambda x: str(x))
    # df['w_speed'] = df['w_speed'].map(lambda x: x.rstrip(' mph'))
    # df['w_speed'] = df['w_speed'].map(lambda x: int(x))

    df.to_csv(output_file_path)

if __name__ == '__main__':
    '''
    '''
    if len(sys.argv) > 1:
        required_parameter_keys = {'data_file_path',
                                   'position',
                                #    'config', # NOTE - This is only temporary
                                   'output_file_path'}
        missing_keys = []
        parameters   = {}
        for arg in sys.argv[1:]:
            split_arg       = arg.split('=')
            key             = split_arg[0].lower()
            value           = split_arg[1].lower()
            parameters[key] = value

        for key in required_parameter_keys:
            if key not in set(parameters):
                missing_keys.append(key)

        assert required_parameter_keys.issubset(set(parameters)), \
            'The following required parameter keys are not present \
            present in sys.argv: {}'.format(missing_keys)

        data_file_path   = parameters['data_file_path']
        position         = parameters['position']
        # NOTE - This is only temporary and this needs to be set in a config file
        config           = {'columns': ['p/h','mlb_id','date','hand','oppt_pitch_hand',
                                        # 'order','w_speed','w_dir','oppt_pitch_name','dk_salary', # For training data
                                        'oppt_pitch_name','dk_salary',
                                        'fd_salary','fd_pos','dk_points'], # For training data
                                        # 'fd_salary','fd_pos'], # For prediction data
                            # 'cat_cols': ['condition','w_dir']
                            'cat_cols': []
                            }
        # config           = parameters['config']

        output_file_path = parameters['output_file_path']
    else:
        print('No command-line parameters')
        data_file_path   = 'raw_rotoguru_data_2015.csv'
        position         = "H"
        config           = {'columns': ['p/h','mlb_id','date','hand','oppt_pitch_hand','condition','adi',
                                        'order','w_speed','w_dir','oppt_pitch_name','dk_salary',
                                        'fd_salary','fd_pos','dk_points'],
                            'cat_cols': ['condition','w_dir']
                            }

        output_file_path = '../../data/test_format_data.csv'

    main(data_file_path, position, config, output_file_path)
    # TODO - Columns should come from a params/config file
    # batter_columns=['p/h','mlb_id','date','hand','oppt_pitch_hand','condition','adi',
    #                 'order','w_speed','w_dir','oppt_pitch_name','dk_salary',
    #                 'fd_salary','fd_pos','dk_points']
    # batter_cat_cols=['condition','w_dir']
    # # #
    # batter_df = player_data("h", batter_columns, batter_cat_cols)
    # batter_df['hand_advantage'] = np.where(batter_df['hand'] == batter_df['oppt_pitch_hand'], 0, 1)
    # batter_df.to_csv('batter_data_2015.csv')

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
