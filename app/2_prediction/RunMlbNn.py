import argparse
import math
import sys

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout, BatchNormalization

sys.path.append('../..')
from NeuralNetwork import NeuralNetwork
from utils.ReadConfig import read_config


# NOTE: This should be reformatted into classes for reuse elsewhere. Parent class
#       will use @abstractmethod decorator for compile function.
# TODO: evaluate method should be renamed to avoid confusion. should be renamed to
#       predict or something similar with should_evaluate still as a parameter

def data_to_train(train_data_file_name):
    '''Get the data to train the model

    Args:
        train_data_file_name (str): Name of file for training data

    Returns:
        X (pd.DataFrame): X_train
        y (pd.Series)   : y_train
    '''
    df = pd.read_csv(train_data_file_name)

    X = df.drop('dk_points', axis=1)
    y = df['dk_points']
    return X, y


def predict_data(X_data_file_name, scaler=RobustScaler()):
    '''Get the data to be used to generate predictions

    Args:
        X_data_file_name (str): Name of file for X data

    Returns:
        data (:obj:`dict` of :obj:obj): Contains the scaled X data, ids,
                                        salaries, and position data
    '''
    X = pd.read_csv(X_data_file_name)

    scaled_X = scaler.fit_transform(X)

    ids      = X['mlb_id'].reset_index().drop('index', axis=1)
    salaries = X['dk_salary'].reset_index().drop('index', axis=1)
    pos      = X['fd_pos'].reset_index().drop('index', axis=1)

    data = {'scaled_X': scaled_X,
            'ids'     : ids,
            'salaries': salaries,
            'pos'     : pos}

    return data


def compare_projected_actual(predictions, y_test):
    '''Prints accuracy details based on averages of projected and actual points
    scored

    Args:
        predictions (numpy array): Actual output predictions
        y_test      (numpy array): Test output predictions

    TODO:
        Calculate averages. 4.5 and 7.3 are the averages for predicted data and
        actual DK points data.
    '''
    i = 0
    total = 0
    total_y = 0
    good = 0
    bad = 0
    for pred, y in zip(predictions, y_test):
        i += 1
        diff = abs(pred - y)
        total += diff
        total_y += y
        if pred >= 4.5:
            if y >= 7.3:
                good += 1
            else:
                bad += 1
        elif pred < 4.5:
            if y < 7.3:
                good += 1
            else:
                bad += 1

    print('Predictions average: ', predictions.mean())
    print('Test average: ', y_test.mean())
    print('good!! {}'.format(good))
    print('bad :( {}'.format(bad))
    print('average difference: {}'.format(total / i))


def main(output_file_name, predict_on_unseen, **kwargs):
    '''Triggers the neural network processes based on various parameters.
    Generates predictions and writes the predictions to a csv.

    Args:
        output_file_name (str)  : File name of generated predictions csv
        predict_on_unseen (bool): True if making predictions on unseen data
        kwargs:
            x_file (str)            : File name of X data
            model_file_name (str)   : File name of saved neural network model
            training_file_name (str): File name of training data
            should_save (bool)      : Whether the model should be saved

            # NOTE: Some of these are mutually exclusive. For example, if
                    should_save is True, you should not pass a training_file_name
                    and vice versa.
    '''
    player_nn = NeuralNetwork()

    if predict_on_unseen:
        assert 'x_file' in kwargs and 'model_file_name' in kwargs, \
            'When predicting on unseen data, you must pass an X_file and a\
            model_file_name to load X data and the model.'

        model_details = predict_data(kwargs['x_file'])
        model_details['model'] = load_model(kwargs['model_file_name'])

        ids      = model_details['ids']
        salaries = model_details['salaries']
        pos      = model_details['pos']

        predictions = player_nn.generate_predictions(model_details['model'],
                                                     model_details['scaled_X'],
                                                     should_evaluate=False)
    else:
        assert 'training_file_name' in kwargs, \
            'A path to a training file must be given if you are not predicting \
            results, e.g. batter_data.csv'

        X, y = data_to_train(kwargs['training_file_name'])

        if 'should_save' in kwargs:
            if kwargs['should_save']:
                assert 'model_file_name' in kwargs, \
                    'When saving a model, you must pass a model_file_name.'
                model_file_name = kwargs['model_file_name']
                dim = len(X.columns)
                compiled_model = player_nn.compile(dim, 'mean_squared_logarithmic_error', 'adam')
                player_nn.fit(compiled_model, X=X, y=y, should_save=True, model_file_name=model_file_name)
                raise SystemExit('Model saved to {}\n Exiting program.'.format(model_file_name))

        dim = len(X.columns)
        compiled_model = player_nn.compile(dim, 'mean_squared_logarithmic_error', 'adam')
        model_details = player_nn.fit(compiled_model, X, y)

        ids      = model_details['X_test']['mlb_id'].reset_index().drop('index', axis=1)
        salaries = model_details['X_test']['dk_salary'].reset_index().drop('index', axis=1)
        pos      = model_details['X_test']['fd_pos'].reset_index().drop('index', axis=1)

        y_train       = model_details['y_train']
        y_test        = model_details['y_test']
        scaled_X_test = model_details['scaled_X']

        predictions = player_nn.generate_predictions(compiled_model,
                                                     scaled_X_test,
                                                     model_details['y_test'],
                                                     should_evaluate=True)

        compare_projected_actual(predictions, y_test)

    # Reformat output predictions to write to csv
    predictions = predictions.tolist()
    predictions = pd.Series(predictions).rename("Predictions")
    predictions = predictions.apply(lambda x: x[0])

    output_df = pd.concat([ids, predictions, salaries,pos], axis=1)
    output_df.to_csv(output_file_name, index=False)

if __name__ == '__main__':
    config = read_config('../config.yaml')['2_prediction']['NeuralNetwork']

    required_parameters = config['required']
    optional_parameters = config['optional']

    output_file_name = required_parameters['output_file_name']
    predict = required_parameters['predict_on_unseen']

    main(output_file_name, predict, **optional_parameters)
