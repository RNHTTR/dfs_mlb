import pandas as pd

# NOTE: Needs to be merged into Knapsack.py
df = pd.read_csv('selections.csv', index_col=0, sep='|')
df = df.apply(lambda x: x.str.lstrip('(').str.rstrip(')').str.split(', '), axis=1)
idx = ['C', '1B', '2B', 'SS', '3B', 'OF']
df['idx'] = idx
df = df.set_index('idx')
df.index.name = None

final_selection = {}
of = []

summ = 0
for i in idx:
    for j in range(6):
        if i == 'OF':
            if len(of) < 3:
                of.append(df.loc[i].iloc[j])
            else:
                for k in range(len(of)):
                    if df.loc[i].iloc[j][2] > of[k][2]:
                        of[k] = df.loc[i].iloc[j]
        else:
            if i in final_selection:
                if df.loc[i].iloc[j] > final_selection[i]:
                    final_selection[i] = df.loc[i].iloc[j]
            else:
                final_selection[i] = df.loc[i].iloc[j]

# Get Infielders
df = pd.DataFrame.from_dict(final_selection, orient='index', dtype=None)
df.columns = ['mlb_id', 'dk_salary', 'dk_points']
df['mlb_id'] = pd.to_numeric(df['mlb_id'], downcast='integer')
player_id_map_df = pd.read_csv('player_id_map.csv')
df = pd.merge(df, player_id_map_df, on='mlb_id', right_index=True)
# print(player_id_map_df.head())

# Get outfielders
of = pd.DataFrame(of, columns=['mlb_id', 'dk_salary', 'dk_points'], index=['OF','OF','OF'])
of['mlb_id'] = pd.to_numeric(of['mlb_id'], downcast='integer')
of = pd.merge(of, player_id_map_df, on='mlb_id', right_index=True)

final_selection = pd.concat([df, of])
print(final_selection)
