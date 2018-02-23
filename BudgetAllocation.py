import pandas as pd


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
        print('-------------\n     {}\n\n{} -- {:.1%}\n-------------'.format(pos_map[pos], pos_sum, pos_sum / total))
