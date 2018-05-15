import sys

import pandas as pd

sys.path.append('../..')
from utils.ReadConfig import read_config


def main(output_file_name):
    # Read Knapsack.py output and organize into clean DataFrame
    df = pd.read_csv('selections.csv', index_col=0, sep='|')
    df = df.apply(lambda x: x.str.lstrip('(').str.rstrip(')').str.split(', '), axis=1)
    idx = ['C', '1B', '2B', 'SS', '3B', 'OF']
    df['idx'] = idx
    df = df.set_index('idx')
    df.index.name = None

    infield = {}
    outfield = []

    # NOTE: Consider refactoring into well-named functions for easier readability
    # Select players to draft
    summ = 0
    for i in idx:
        for j in range(6):
            # Get a list of four outfielders (one possibly for util position)
            if i == 'OF':
                if len(outfield) < 3:
                    outfield.append(df.loc[i].iloc[j])
                else:
                    # Replace players with players who have a higher projected score
                    for k in range(len(outfield)):
                        if df.loc[i].iloc[j][2] > outfield[k][2]:
                            outfield[k] = df.loc[i].iloc[j]
            else:
                # Select players with highest projected score
                if i in infield:
                    # Replace players with players who have a higher projected score
                    if df.loc[i].iloc[j] > infield[i]:
                        infield[i] = df.loc[i].iloc[j]
                else:
                    infield[i] = df.loc[i].iloc[j]

    # Get Infielders
    df = pd.DataFrame.from_dict(infield, orient='index', dtype=None)
    df.columns = ['mlb_id', 'dk_salary', 'dk_points']
    df['mlb_id'] = pd.to_numeric(df['mlb_id'], downcast='integer')
    player_id_map_df = pd.read_csv('player_id_map.csv')
    df = pd.merge(df, player_id_map_df, on='mlb_id', right_index=True)

    # Get outfielders
    outfield = pd.DataFrame(outfield, columns=['mlb_id', 'dk_salary', 'dk_points'], index=['OF','OF','OF'])
    outfield['mlb_id'] = pd.to_numeric(outfield['mlb_id'], downcast='integer')
    outfield = pd.merge(outfield, player_id_map_df, on='mlb_id', right_index=True)

    all_players = pd.concat([df, outfield])

    all_players.to_csv(output_file_name)


if __name__ == '__main__':
    config = read_config('../config.yaml')['3_selection']['FinalSelection']

    output_file_name = config['output_file_name']

    main(output_file_name)
