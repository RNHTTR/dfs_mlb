import argparse
import requests
import sys
import yaml
from io import StringIO

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

sys.path.append('../..')
from utils.ObjectChecks import is_int
from utils.ReadConfig import read_config


# TODO: Reformat MLBData to inherit from generic Data class with read/write capabilities,
#       one-hot encoding function,

class MLBData:
    '''Downloads, formats, and prepares data for the neural network.'''

    def get_rotoguru_data(self, link):
        '''Get player data from rotoguru

        Args:
            link (str): Link to rotoguru data

        Returns:
            df (pd.DataFrame): Raw player data
        '''
        page = requests.get(link)

        soup = BeautifulSoup(page.content, 'lxml')

        p  = soup.p.get_text()
        data = p.split('\n\n', 1)[0]
        data = StringIO(data)
        df = pd.read_csv(data, delimiter=':', index_col=False)
        df.columns = df.columns.str.lower()

        return df


    def append_new(self, new_df, old_file_name):
        '''Append new entries if they are not in the batter data

        Args:
            new_df (pd.DataFrame): Newly scraped data from rotoguru
            old_file_name (str)  : Name of existing data set

        Returns:
            df (pd.DataFrame): Existing data set with new data appended
        '''
        old_df = pd.read_csv(old_file_name, index_col=0)
        df = pd.concat([old_df, new_df], ignore_index=True)
        df = df.drop_duplicates()

        return df


    def get_X_data(self, df, date, output_file_name):
        '''Get X Data for neural network

        Args:
            df (pd.DataFrame)     : Player data set
            date (str)            : Date to pass to neural network for predictions
            output_file_name (str): Name of file to pass to neural network
        '''
        df = df.loc[df['date'] == date]
        columns_to_drop = list(df.filter(regex='_points').columns)
        df = df.drop(columns=columns_to_drop)
        df.to_csv(output_file_name)
        print(df)


    def read_fangraphs_pitcher_data(self, pitcher_config):
        '''
        Get opponent pitcher data to be used for batting data
        https://www.fangraphs.com/leaders.aspx?pos=all&stats=pit&lg=all&qual=y&type=1&season=2017&month=0&season1=2017&ind=0&team=0&rost=0&age=0&filter=&players=0

        Args:
            pitcher_config (:obj:`dict` of :obj:obj): Configuration details

        Returns:
            df (pd.DataFrame): Pitcher data
        '''
        df = pd.read_csv(pitcher_config['link'])
        df = df[pitcher_config['columns']]
        df['Name'] = df['Name'].apply(lambda name: ", ".join(name.split(" ")[::-1]))
        df = df.rename(columns = pitcher_config['rename'])

        return df


    def get_data(self, config):
        '''Wrapper to get batter data and save it as a csv

        Args:
            config (:obj:`dict` of :obj:obj): Configuration details
        '''
        get_data_config = config['1_data']['get_data']

        year = get_data_config['year']
        date = get_data_config['date']
        old_file_name = get_data_config['old_file_name']
        new_file_name = get_data_config['new_file_name']
        query_param_2018 = get_data_config['2018_query_param']
        query_param_2017 = get_data_config['2017_query_param']
        query_param_2016 = get_data_config['2016_query_param']

        if is_int(year):
            link = "http://rotoguru1.com/cgi-bin/mlb-dbd-{}.pl".format(year)
        else:
            base_link = "http://rotoguru1.com/cgi-bin/mlb-dbd-"

        if year == 'new':
            link = base_link + '2018.pl?' + query_param_2018
            df = self.get_rotoguru_data(link)
            output_file_name = 'raw_rotoguru_data_{}.csv'.format(year)

            format_config = config['1_data']['format_data']

            data_file_path      = 'raw_rotoguru_data_new.csv'
            position            = format_config['pitcher_or_hitter']
            columns             = format_config['columns']
            categorical_columns = format_config['cat_cols']
            pitcher_config      = format_config['pitcher_data']

            df = self.player_data(data_file_path, position, columns,
                                  categorical_columns, pitcher_config)

            df = self.append_new(df, old_file_name)

            df = df.sort_values(by=['mlb_id', 'date'])

            df.to_csv('batter_data.csv')
            self.get_X_data(df, date, 'raw_X.csv')
            quit()

        elif year == 'all':
            link = base_link + '2018.pl?' + query_param_2018
            df_2018 = self.get_rotoguru_data(link)

            link = base_link + '2017.pl?' + query_param_2017
            df_2017 = self.get_rotoguru_data(link)

            link = base_link + '2016.pl?' + query_param_2016
            df_2016 = self.get_rotoguru_data(link)

            link = base_link + '2015.pl'
            df_2015 = self.get_rotoguru_data(link)

            df = pd.concat([df_2017, df_2016, df_2015], ignore_index=True)
        elif year == '2017':
            link = link + query_param_2017
            df = self.get_rotoguru_data(link)
        elif year == '2016':
            link = link + query_param_2016
            df = self.get_rotoguru_data(link)

        output_file_name = 'raw_rotoguru_data_{}.csv'.format(year)
        df.to_csv(output_file_name)


    def one_hot(self, df, columns, cat_cols):
        '''One hot encode categorical variables

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


    def player_data(self, data_file_path, position, columns, cat_cols, pitcher_config):
        '''Formats data for neural network

        Args:
            data_file_path (str): Raw data
            position (str)      : P for pitchers, H for hitters
            columns  (list[str]): All columns to be read into the DataFrame
            cat_cols (list[str]): Categorical columns that need to be one-hot encoded
            pitcher_config (:obj:`dict` of :obj:obj): Configuration details

        Returns:
            df (Pandas DF): Player data for batters or pitchers
        '''
        position = position.upper()

        assert position in ['P', 'H'], \
            "Position must be given as a single letter, P for pitchers or H for " \
            "hitters."

        df = pd.read_csv(data_file_path)
        df = self.one_hot(df, columns, cat_cols)
        df = df.loc[df['p/h'] == position]
        df = df.drop('p/h', axis=1)

        fangraphs_pitcher_df = self.read_fangraphs_pitcher_data(pitcher_config)

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


    def format_data(self, config):
        '''Wrapper for data formatting. Saves formatted data as a csv

        Args:
            config (:obj:`dict` of :obj:obj): Configuration details
        '''
        format_config = config['1_data']['format_data']

        data_file_path      = format_config['input_file_name']
        position            = format_config['pitcher_or_hitter']
        columns             = format_config['columns']
        categorical_columns = format_config['cat_cols']
        pitcher_config      = format_config['pitcher_data']

        df = self.player_data(data_file_path, position, columns,
                              categorical_columns, pitcher_config)

        output_file_path = format_config['output_file_name']
        df.to_csv(output_file_path)


    def points_last_30(self, config):
        '''Gets players average points for the last n games

        Args:
            config (:obj:`dict` of :obj:obj): Configuration details

        TODO: Test. Implement this if it's useful, delete it otherwise.
        '''
        # Make use of avg points last 30
        # NOTE - how can we configure this to use all data sets?
             # NOTE - ANSWER! Sort the DF by mlb_id
        # df = pd.read_csv('batter_data_all.csv', index_col=0)
        points_last_30_config = config['1_data']['points_last_30']

        start_date       = points_last_30_config['start_date']
        stop_date        = points_last_30_config['stop_date']
        input_file_name  = points_last_30_config['input_file_name']
        output_file_name = points_last_30_config['output_file_name']

        df = pd.read_csv(input_file_name, index_col=0)

        row_count = df.shape[0]

        current_id = None
        for index, row in df.iterrows():
            if current_id == None:
                current_id = row['mlb_id']
            if row['mlb_id'] != current_id:
                avg_points_last_30 = df.loc[(df['mlb_id'] == current_id) & (df['date'] < stop_date) & (df['date'] >= start_date)]
                avg_points_last_30 = avg_points_last_30['dk_points'].mean()

                df.loc[df['mlb_id'] == current_id, 'avg_points_last_30'] = avg_points_last_30

                current_id = row['mlb_id']

            sys.stdout.write('record {} of {} records           \r'.format(index, row_count))
            sys.stdout.flush()

        df = df.dropna(axis=0)
        print(df)
        df.to_csv(output_file_name, index=False)


def main():
    config = read_config('../config.yaml')
    mlb_data = MLBData()
    mlb_data.get_data(config)
    mlb_data.format_data(config)


if __name__ == '__main__':
    # TODO -  Implement argparse to read config file
    main()
