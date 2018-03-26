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
    '''
    Compile the keras neural network model

    Needs to take in **kwargs params later
    '''
    model = Sequential()

    # dim = len(X.columns)

    model.add(Dense(dim, input_dim=dim, kernel_initializer='orthogonal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(math.floor(dim / 2), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3, seed=7))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='linear'))

    # Fit model, evaluate model, and generate predictions
    # NOTE - Loss improves for about ~40 epochs. Not sure how to interpret mean_squared_logarithmic_error
    model.compile(loss='mean_squared_logarithmic_error', optimizer='adam')

    return model

def fit(input_file_name, scaler=RobustScaler(),
        train_test=True, should_save=False, model_file_name=None):
    '''
    Fit the keras neural network model

    Args:
        input_file_name (str)  : Input data file name
        # output_file_name (str) : I don't think we should have this...
        scaler (sklearn scaler): sk learn feature scaling object
        train_test (bool)      : True if testing model with train_test_split
        should_save (bool)     : True if the model should be saved for future use
        model_file_name (str)  : Must be given if should_save is True

    Returns:
        If should_train:
            model (keras model): Trained keras model
            scaled_X_train (DF): Scaled X training data
            scaled_X_test (DF) : Scaled X test data
            y_train (DF)       : y training data
            y_test  (DF)       : y test data
            ids (Series)       : mlb_id for each player in the test data
            salaries (Series)  : salaries for each player in the test data
            pos  (Series)      : position for each player in the test data
        If not should_train:
            model (keras model): Trained keras model
            scaled_X_train (DF): Scaled X training data
            y_train (DF)       : y training data
    '''
    if should_save:
        assert model_file_name is not None, \
            'If the model is to be saved, you must give a file name for \
            model_file_name.'
    df = pd.read_csv(input_file_name)

    # NOTE - This should not be handled in the neural network file. This should
    #        should be handled somewhere in data prep (/app/1_data)
    X = df.drop('dk_points', axis=1)
    y = df['dk_points']
    dim = len(X.columns)

    if train_test:
        # Set seed for reproducing results
        seed = 7
        np.random.seed(seed)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=7)

        scaled_X_train = scaler.fit_transform(X_train)
        scaled_X_test  = scaler.transform(X_test)

        ids = X_test['mlb_id'].reset_index().drop('index', axis=1)
        salaries = X_test['dk_salary'].reset_index().drop('index', axis=1)
        pos = X_test['fd_pos'].reset_index().drop('index', axis=1)

        model = compile(dim)
        model.fit(scaled_X_train, y_train, epochs=10)

        return {'model'         : model,
                'scaled_X_train': scaled_X_train,
                'scaled_X_test' : scaled_X_test,
                'y_train'       : y_train,
                'y_test'        : y_test,
                'ids'           : ids,
                'salaries'      : salaries,
                'pos'           : pos}

    else:
        X_train = X
        y_train = y

        scaled_X_train = scaler.fit_transform(X_train)

        model = compile(dim)
        model.fit(scaled_X_train, y_train, epochs=10)

        if should_save:
            model.save(model_file_name)

        # QUESTION - Should x, y be written to a csv?
        return {'model': model, 'scaled_X_train': scaled_X_train,
                'y_train': y_train}


def evaluate(data, should_evaluate=True, should_load=False, model_file_name=None):
    '''

    '''
    scaled_X_test = data['scaled_X_test']
    y_test = data['y_test']

    if should_load:
        model = load_model(model_file_name)
    else:
        model = data['model']

    if should_evaluate:
        model.evaluate(x=scaled_X_test, y=y_test, batch_size=None, verbose=1, sample_weight=None)

    predictions = model.predict(scaled_X_test, verbose=1)

    return predictions


def main(input_file_name, output_file_name):
    '''
    '''
    
    model_details = fit(input_file_name)

    model = model_details['model']
    scaled_X_train = model_details['scaled_X_train']
    scaled_X_test = model_details['scaled_X_test']
    y_train = model_details['y_train']
    y_test = model_details['y_test']
    ids = model_details['ids']
    salaries = model_details['salaries']
    pos = model_details['pos']

    # predictions = evaluate(x=scaled_X_test, y=y_test, batch_size=None, verbose=1, sample_weight=None)
    predictions = evaluate(model_details)

    # predictions = model.predict(scaled_X_test, verbose=1)

    # Calculate average difference between projected and actual points scored
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

    print(predictions.mean())
    print(y_test.mean())
    print('good!! {}'.format(good))
    print('bad :( {}'.format(bad))
    print('average difference: {}'.format(total / i))

    # Generate output predictions file for batters
    predictions = predictions.tolist()
    predictions = pd.Series(predictions).rename("Predictions")
    predictions = predictions.apply(lambda x: x[0])

    output_df = pd.concat([ids, predictions, salaries,pos], axis=1)
    output_df.to_csv(output_file_name, index=False)

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

        input_file_name  = parameters['input_file_name']
        output_file_name = parameters['output_file_name']
    else:
        input_file_name  = '../../data/20170718_batter_data_2.csv'
        output_file_name = 'batter_predictions_all.csv'
    # print('inp: {}'.format(input_file_name))
    # print('oup: {}'.format(output_file_name))
    main(input_file_name, output_file_name)
