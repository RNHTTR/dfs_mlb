import requests
from io import StringIO

import pandas as pd
from bs4 import BeautifulSoup

from data.PitcherData import read_pitcher_data
# from PitcherData import read_pitcher_data


def get_initial_data():
    '''
    Get player data from rotoguru1 sample data
    '''
    page = requests.get("http://rotoguru1.com/cgi-bin/mlb-dbd-2017.pl")
    # soup = BeautifulSoup(page.content, 'html.parser')
    soup = BeautifulSoup(page.content, 'lxml')
    print(soup.prettify())
    pre  = soup.pre
    data = str(pre).split('<pre>\n', 1)[1]
    data = data.split('\n\n', 1)[0]
    data = StringIO(data)
    df = pd.read_csv(data, delimiter=':', index_col=False)
    df.columns = df.columns.str.lower()
    return df


def get_abreu_data(df, columns):
    '''
    Get dataset with just Abreu
    '''
    # df = df[columns]
    # columns = ['h/a','temp','condition','w_speed','w_dir','order','oppt_pitch_hand','oppt_pitch_mlb_id','dk_points','dk_salary']
    # columns = ['h/a','temp','condition','w_speed','w_dir','order','oppt_pitch_hand','dk_points','dk_salary']
    last_id = None

    n_players = 20
    player = 1

    for index, row in df.iterrows():
        if last_id == None:
            pass
        else:
            if row['mlb_id'] != last_id:
                df = df.loc[df['mlb_id'] == last_id]
                df = df[columns]
                break
        last_id = row['mlb_id']

    return df


def one_hot_abreu(df, columns, cat_columns):
    '''
    One hot encode categorical variables
    '''
    df = df[columns]
    for col in cat_columns:
        one_hot = pd.get_dummies(df[col])
        df = df.drop(col, axis=1)
        df = df.join(one_hot)

    return df


# def abreu_data(columns=['h/a','temp','condition','w_speed','w_dir','order','oppt_pitch_hand','dk_salary','dk_points'],
#          categorical_columns=['h/a','condition','w_dir','oppt_pitch_hand']):
def abreu_data(columns=['oppt_pitch_hand','oppt_pitch_name','dk_salary','dk_points'],
               categorical_columns=['oppt_pitch_hand']):
    '''
    Return final Abreu dataset
    '''
    df = get_initial_data()
    df = get_abreu_data(df, columns)
    df = one_hot_abreu(df, columns, categorical_columns)

    pitcher_data_df = read_pitcher_data()

    df = df.merge(pitcher_data_df, on='oppt_pitch_name', how='left')
    df = df.drop('oppt_pitch_name', axis=1)

    return df.dropna(axis=0)


if __name__ == '__main__':
    print(abreu_data())
    # abreu_data()
