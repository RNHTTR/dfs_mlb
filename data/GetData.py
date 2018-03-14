import requests
from io import StringIO

import pandas as pd
from bs4 import BeautifulSoup


def get_rotoguru_data():
    '''
    Get player data from rotoguru1 sample data

    Returns
    '''
    # 2017 link
    # page = requests.get("http://rotoguru1.com/cgi-bin/mlb-dbd-2017.pl?&user=madrhatter&key=M3487509151")
    # 2016 link
    page = requests.get("http://rotoguru1.com/cgi-bin/mlb-dbd-2016.pl?&user=madrhatter&key=M6911301251")
    
    soup = BeautifulSoup(page.content, 'lxml')
    p  = soup.p.get_text()
    data = p.split('\n\n', 1)[0]
    data = StringIO(data)
    df = pd.read_csv(data, delimiter=':', index_col=False)
    df.columns = df.columns.str.lower()

    return df


def read_fangraphs_pitcher_data():
    '''
    Get opponent pitcher data to be used for batting data
    https://www.fangraphs.com/leaders.aspx?pos=all&stats=pit&lg=all&qual=y&type=1&season=2017&month=0&season1=2017&ind=0&team=0&rost=0&age=0&filter=&players=0
    '''
    df = pd.read_csv('opponent_pitcher_data.csv')
    df = df[["AVG","Name","K/9","BB/9","K/BB","HR/9","K%","BB%","WHIP","ERA"]]
    df['Name'] = df['Name'].apply(lambda name: ", ".join(name.split(" ")[::-1]))
    df.columns = ['avg', 'oppt_pitch_name',"K/9","BB/9","K/BB","HR/9","K%","BB%","WHIP","ERA"]

    return df


if __name__ == '__main__':
    df = get_rotoguru_data()
    df.to_csv('raw_rotoguru_data_2016.csv')
