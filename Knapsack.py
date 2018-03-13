import sys

import numpy as np
import pandas as pd


def get_salary_and_value(data, salary_index, value_index):
    '''
    Get the value of a list of player_data tuples

    Args:
        data (list[player_data]): List of player_data tuples
        salary_index (int)      : Position in tuple of salary
        value_index (int)       : Position in tuple of value

    Returns:
        total (int): Total value of player_data tuples in data
    '''
    if data:
        value_total = 0
        salary_total = 0
        for player in data:
            value_total  += player[value_index]
            salary_total += player[salary_index]
            # print(player)
        return salary_total, value_total
    else:
        return 0, 0


def knapsack(df, cols, max_weight, n, start):
    '''
    Returns a list that produces the highest value given a maximum capacity
    (weight), and maximum number of items.

    Args:
        df   (Pandas DF): Pandas DataFrame with player data
        cols (list[str]): List of columns to be used in the knapsack problem
        max_weight (int): Maximum combined value of assets (capacity)
        n          (int): Maximum number of assets permitted

    Returns:
        most_valuable (list[*]): List of tuples of assets and their value
    '''
    df = df[cols]
    salary_index = 1
    value_index  = 2
    player_data  = [tuple(row) for row in df.values]

    num_players = len(player_data)

    most_valuable    = []
    value_of_mv      = 0
    current          = []
    value_of_current = 0

    i = 0
    j = 1
    for i in range(start, num_players):
        sys.stdout.write('player {} of {} players\r'.format(i, num_players))
        sys.stdout.flush()
        sys.stdout.flush()
        current.append(player_data[i])
        for j in range(start, num_players):
            if i != j:
                current_salary, current_value = get_salary_and_value(current, salary_index, value_index)
                if current_salary + player_data[j][salary_index] <= max_weight and len(current) < n:
                    current.append(player_data[j])
        current_salary, current_value = get_salary_and_value(current, salary_index, value_index)
        most_salary, most_value = get_salary_and_value(most_valuable, salary_index, value_index)
        if current_value > most_value:
            most_valuable = current
        current = []

    return most_valuable


def run_random(n_iter, df, cols, max_weight, n):
    '''
    Return list of n players that maximizes projected points over n_iter iterations

    Args:
        n_iter     (int): Number of times to generate a random list of players
        df   (Pandas DF): Pandas DataFrame with player data
        cols (list[str]): List of columns to be used in the knapsack problem
        max_weight (int): Maximum combined value of assets (capacity)
        n          (int): Maximum number of assets permitted

    Returns:
        actually_most_valuable (list[*]): List of tuples of assets and their value
    '''
    actually_most_valuable = []

    for i in range(n_iter):
        print('Iteration {} of {}           \r'.format(i, n_iter))
        sys.stdout.flush()
        rand_int = np.random.randint(0, df.shape[0])
        most_valuable = knapsack(df, cols, max_weight, n, rand_int)
        x, value = get_salary_and_value(most_valuable, 1, 2)
        x, super_value = get_salary_and_value(actually_most_valuable, 1, 2)
        if value > super_value:
            actually_most_valuable = most_valuable
            print('New most Valuable: {}'.format(actually_most_valuable))

    return actually_most_valuable


if __name__ == '__main__':
    df = pd.read_csv('batter_predictions.csv', index_col=0)
    df = df.loc[df['fd_pos'] == 7]
    cols = ['mlb_id', 'dk_salary', 'Predictions']

    print('Final most val: {}'.format(run_random(100, df, cols, 15000, 3)))
