import math
import sys

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout, BatchNormalization


class NeuralNetwork:
    '''
    '''
    def compile(self, dim, loss, optimizer):
        '''Compile the keras neural network model architecture. Default
        implementation is a one-output-variable linear regression model.

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
        model.add(Dropout(0.3))
        model.add(BatchNormalization())
        model.add(Dense(1, activation='linear'))

        model.compile(loss=loss, optimizer=optimizer)

        return model


    def fit(self, model, X, y, should_save=False, model_file_name=None, seed=False, scaler=RobustScaler()):
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
        '''
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

        return {'model'         : model,
                'scaled_X_train': scaled_X_train,
                'scaled_X'      : scaled_X_test,
                'X_train'       : X_train,
                'X_test'        : X_test,
                'y_train'       : y_train,
                'y_test'        : y_test,
                }


    def generate_predictions(self, model, X, y=None, should_evaluate=True):
        '''Evaluate model performance and generate predictions

        Args:
            model (keras model object): Keras deep learning model
            X (numpy ndarray)         : X data to predict on
            y (numpy ndarray)         : y data to predict on
            should_evaluate (bool)    : Whether or not to use the Keras model
                                        evaluate method

        Returns:
            predictions (numpy array)     : Generates output predictions
        '''
        if should_evaluate:
            assert y is not None, "y_test data is required to evaluate the model"
            model.evaluate(x=X, y=y, batch_size=None, verbose=1, sample_weight=None)

        predictions = model.predict(X, verbose=1)

        return predictions
