import requests
import sys
from io import StringIO

import pandas as pd
from bs4 import BeautifulSoup


# QUESTION - Should we consolidate GetData.py, FormatData.py, and PointsLast30.py
#            into one file Data.py and use classes?


def get_rotoguru_data(link):
    '''
    Get player data from rotoguru1 sample data

    Returns
    '''
    page = requests.get(link)

    soup = BeautifulSoup(page.content, 'lxml')

    p  = soup.p.get_text()
    data = p.split('\n\n', 1)[0]
    data = StringIO(data)
    df = pd.read_csv(data, delimiter=':', index_col=False)
    df.columns = df.columns.str.lower()

    return df


def append_new(df, old_file_name, new_file_name):
    '''
    Append new entries if they are not in the batter data
    '''
    # TODO - df = df.drop_duplicates (this will drop duplicate rows)
    pass


def get_X_data(df, date, output_file_name):
    '''
    Get X Data for neural network
    '''
    df = df.loc[df['date'] == date]
    columns_to_drop = list(df.filter(regex='_points').columns)
    # columns_to_drop.append(list(df.filter(regex='dd_').columns))
    # columns_to_drop.append(list(df.filter(regex='yh_').columns))
    df = df.drop(columns=columns_to_drop)
    df.to_csv(output_file_name)
    print(df)


def read_fangraphs_pitcher_data():
    '''
    Get opponent pitcher data to be used for batting data
    https://www.fangraphs.com/leaders.aspx?pos=all&stats=pit&lg=all&qual=y&type=1&season=2017&month=0&season1=2017&ind=0&team=0&rost=0&age=0&filter=&players=0
    '''
    df = pd.read_csv('../../data/opponent_pitcher_data.csv')
    df = df[["AVG","Name","K/9","BB/9","K/BB","HR/9","K%","BB%","WHIP","ERA"]]
    df['Name'] = df['Name'].apply(lambda name: ", ".join(name.split(" ")[::-1]))
    df.columns = ['avg', 'oppt_pitch_name',"K/9","BB/9","K/BB","HR/9","K%","BB%","WHIP","ERA"]

    return df


def is_int(value):
  try:
    int(value)
    return True
  except ValueError:
    return False


def main(year):
    if is_int(year):
        link = "http://rotoguru1.com/cgi-bin/mlb-dbd-{}.pl".format(year)
    else:
        base_link = "http://rotoguru1.com/cgi-bin/mlb-dbd-"

    if year == 'new':
        # TODO - Read new data, prepare for formatter
        link = base_link + '2018.pl'
        df = get_rotoguru_data(link)
        get_X_data(df, 20180404, 'raw_X.csv')
        # raise ValueError("Not ready for new yet!")
    elif year == 'all':
        link = base_link + '2017.pl?&user=madrhatter&key=M3487509151'
        df_2017 = get_rotoguru_data(link)

        link = base_link + '2016.pl?&user=madrhatter&key=M6911301251'
        df_2016 = get_rotoguru_data(link)

        link = base_link + '2015.pl'
        df_2015 = get_rotoguru_data(link)

        df = pd.concat([df_2017, df_2016, df_2015], ignore_index=True)
    elif year == '2017':
        # TODO - Should require username and key from command line to access
        link = link + '&user=madrhatter&key=M3487509151'
        df = get_rotoguru_data(link)
    elif year == '2016':
        # TODO - Should require username and key from command line to access
        link = link + '&user=madrhatter&key=M6911301251'
        df = get_rotoguru_data(link)

    output_file_name = 'raw_rotoguru_data_{}.csv'.format(year)
    df.to_csv(output_file_name)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        required_parameter_keys = {'year'}
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

        year = parameters['year']

    else:
        year = '2017'

    main(year)
