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


def compile(dim):
    '''Compile the keras neural network model

    Args:
        dim (int): Input data dimension

    Returns:
        model (keras Sequential model): Compiled neural network model ready to
                                        be trained

    TODO:
        Needs to take in **kwargs params later, e.g. kernel_initializer, activations
    '''
    model = Sequential()

    model.add(Dense(dim, input_dim=dim, kernel_initializer='orthogonal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(math.floor(dim / 2), activation='relu'))
    model.add(BatchNormalization())
    # model.add(Dropout(0.3, seed=7))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='linear'))

    # Fit model, evaluate model, and generate predictions
    # NOTE - Loss improves for about ~40 epochs. Not sure how to interpret mean_squared_logarithmic_error
    model.compile(loss='mean_squared_logarithmic_error', optimizer='adam')

    return model


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


def fit(X, y, should_save=False, model_file_name=None, seed=False, scaler=RobustScaler()):
    '''
    Fit the keras neural network model

    Args:
        X                      : input X data
        y                      : input y data
        scaler (sklearn scaler): sk learn feature scaling object
        # train_test (bool)      : True if testing model with train_test_split
        test                   : True if we want to set a random seed for testing
        should_save (bool)     : True if the model should be saved for future use
        model_file_name (str)  : Must be given if should_save is True

    Returns:
        If should_save:
            None (The model will be saved to model_file_name)
        Else:
            model (keras model): Trained keras model
            scaled_X_train (DF): Scaled X training data
            scaled_X_test (DF) : Scaled X test data
            y_train (DF)       : y training data
            y_test  (DF)       : y test data
            ids (Series)       : mlb_id for each player in the test data
            salaries (Series)  : salaries for each player in the test data
            pos  (Series)      : position for each player in the test data
    '''
    dim = len(X.columns)
    model = compile(dim)

    if seed:
        # Set seed for reproducing results
        np.random.seed(seed)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=seed)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    scaled_X_train = scaler.fit_transform(X_train)
    scaled_X_test  = scaler.transform(X_test)

    model.fit(scaled_X_train, y_train, epochs=10)

    if should_save:
        model.save(model_file_name)
    else:
        ids      = X_test['mlb_id'].reset_index().drop('index', axis=1)
        salaries = X_test['dk_salary'].reset_index().drop('index', axis=1)
        pos      = X_test['fd_pos'].reset_index().drop('index', axis=1)
        return {'model'         : model,
                'scaled_X_train': scaled_X_train,
                'scaled_X'      : scaled_X_test,
                'y_train'       : y_train,
                'y_test'        : y_test,
                'ids'           : ids,
                'salaries'      : salaries,
                'pos'           : pos}


def evaluate(data, should_evaluate=True):
    '''Evaluate model performance and generate predictions

    Args:
        data (:obj:`dict` of :obj:obj): Contains the model, scaled X data, ids,
                                        salaries, and position data
        should_evaluate (bool)        : Whether or not to use the Keras evaluate
                                        method

    Returns:
        predictions (numpy array)     : Generates output predictions
    '''
    scaled_X_test = data['scaled_X']
    model = data['model']

    if should_evaluate:
        y_test = data['y_test']
        model.evaluate(x=scaled_X_test, y=y_test, batch_size=None, verbose=1, sample_weight=None)

    predictions = model.predict(scaled_X_test, verbose=1)

    return predictions


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


def main(output_file_name, predict, **kwargs):
    '''Triggers the neural network processes based on various parameters.
    Generates predictions and writes the predictions to a csv.

    Args:
        output_file_name (str): File name of generated predictions csv
        predict (bool)        : True if making predictions on unseen data
        kwargs:
            x_file (str)            : File name of X data
            model_file_name (str)   : File name of saved neural network model
            training_file_name (str): File name of training data
            should_save (bool)      : Whether the model should be saved

            # NOTE: Some of these are mutually exclusive. For example, if
                    should_save is True, you should not pass a training_file_name
                    and vice versa.

    '''
    if predict:
        assert 'x_file' in kwargs and 'model_file_name' in kwargs, \
            'When making real predictions, (i.e. Predict = True), you must pass \
            an X_file and a model_file_name to load X data and the model.'

        model_details = predict_data(kwargs['x_file'])
        model_details['model'] = load_model(kwargs['model_file_name'])

        predictions   = evaluate(model_details, should_evaluate=False)

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
                fit(X, y, True, model_file_name)
                raise SystemExit('Model saved to {}'.format(model_file_name))
        else:
            model_details = fit(X, y)

        y_train = model_details['y_train']
        y_test  = model_details['y_test']

        predictions = evaluate(model_details, should_evaluate=True)

        compare_projected_actual(predictions, y_test)

    ids            = model_details['ids']
    salaries       = model_details['salaries']
    pos            = model_details['pos']

    # Reformat output predictions to write to csv
    predictions = predictions.tolist()
    predictions = pd.Series(predictions).rename("Predictions")
    predictions = predictions.apply(lambda x: x[0])

    output_df = pd.concat([ids, predictions, salaries,pos], axis=1)
    output_df.to_csv(output_file_name, index=False)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        # NOTE - The predict command line arg is misleading. Predictions are always generated.
        #        Should be renamed to "train" or similar
        # TODO - Reformat to config file
        required_parameter_keys = {'output_file_name', 'predict'}
        optional_parameter_keys = {'training_file_name', 'should_load',
                                   'model_file_name', 'should_save', 'X_file'}
        missing_keys        = []
        parameters          = {}
        optional_parameters = {}
        for arg in sys.argv[1:]:
            split_arg       = arg.split('=')
            key             = split_arg[0].lower()
            value           = split_arg[1].lower()
            if key in required_parameter_keys:
                parameters[key] = value
            else:
                optional_parameters[key] = value

        for key in required_parameter_keys:
            if key not in set(parameters):
                missing_keys.append(key)

        assert required_parameter_keys.issubset(set(parameters)), \
            'The following required parameter keys are not present \
            present in sys.argv: {}'.format(missing_keys)

        output_file_name = parameters['output_file_name']
        predict = True if parameters['predict'] == 'true' else False

        if 'should_load' in optional_parameters:
            optional_parameters['should_load'] = True if optional_parameters['should_load'] == 'true' else False
        if 'should_save' in optional_parameters:
            optional_parameters['should_save'] = True if optional_parameters['should_save'] == 'true' else False
    else:
        input_file_name  = '../../data/20170718_batter_data_2.csv'
        output_file_name = 'batter_predictions_all.csv'

    main(output_file_name, predict, **optional_parameters)
