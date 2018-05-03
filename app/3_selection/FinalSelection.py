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

df = pd.DataFrame.from_dict(final_selection, orient='index', dtype=None)
df.columns = ['mlb_id', 'dk_salary', 'dk_points']
print(df)
