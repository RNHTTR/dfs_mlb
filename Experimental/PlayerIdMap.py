import pandas as pd


df = pd.read_csv('../app/1_data/raw_rotoguru_data_all.csv')
df.columns = df.columns.str.lower()
df = df[['mlb_id', 'name_first_last']]
df = df.drop_duplicates()
df.to_csv('../app/3_selection/player_id_map.csv', index=False)
