'''
TODO: This needs to be more configurable via config file. Need more than just i/o file names.
      Including the number of iterations
TODO: Need to implement reverse seek for optimal players as well as random seek, and allow these to be configurable.
'''

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
        # sys.stdout.flush()
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
        # sys.stdout.flush()
        rand_int = np.random.randint(0, df.shape[0])
        most_valuable = knapsack(df, cols, max_weight, n, rand_int)
        x, value = get_salary_and_value(most_valuable, 1, 2)
        x, super_value = get_salary_and_value(actually_most_valuable, 1, 2)
        if value > super_value:
            actually_most_valuable = most_valuable

    return actually_most_valuable


def main(input_file_name, output_file_name, n_of=6, n_inf=2,
         positions=[2, 3, 4, 5, 6, 7],
         prop_of=30000, prop_inf=10000, n_iter=100):
        # prop_of=30000, prop_inf=10000, n_iter=1):
    '''
    Return list of n players that maximizes projected points over n_iter iterations

    Args:
        input_file_name (str): Input file name -- output from ML process
        n_of (int)           : Number of outfielders to select
        n_inf (int)          : Number of each infield position to select
        positions (list[int]): List of baseball positions by number
        prop_of (int)        : Salary portion for outfielders * 2
        prop_inf (int)       : Salary portion for infielders * 2
        n_iter (int)         : Number of times to generate a random list of players
    '''
    df = pd.read_csv(input_file_name)
    cols = ['mlb_id', 'dk_salary', 'Predictions']

    selections = {}
    for position in positions:
        if position == 7:
            temp_df = df.loc[df['fd_pos'] == position]
            # selections.append(run_random(n_iter, temp_df, cols, prop_of, n_of))
            selections[position] = run_random(n_iter, temp_df, cols, prop_of, n_of)
        else:
            temp_df = df.loc[df['fd_pos'] == position]
            # selections.append(run_random(n_iter, temp_df, cols, prop_inf, n_inf))
            selections[position] = run_random(n_iter, temp_df, cols, prop_inf, n_inf)

    # TODO - What to do with list of selection lists (output)?
    df = pd.DataFrame.from_dict(selections, orient='index', dtype=None)
    columns = [n for n in range(1,n_of+1)]
    df.columns = columns
    df = df.fillna(value='--')
    print(df)
    df.to_csv(output_file_name, sep="|")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        required_parameter_keys = {'input_file_name', 'output_file_name'}
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

        input_file_name = parameters['input_file_name']
        output_file_name = parameters['output_file_name']

    else:
        input_file_name = 'batter_predictions.csv'
        output_file_name = 'selections.csv'

    main(input_file_name, output_file_name)
