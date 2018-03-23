import math

import numpy as np
import pandas as pd

# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, BatchNormalization, LSTM


# TODO - Read from command line parameters
# df = pd.read_csv('data/20170718_batter_data_2.csv')
df = pd.read_csv('data/batter_data_all_2.csv')

df = df.loc[df['mlb_id'] == 116338]
df = df.sort_values(by='mlb_id')
print(df.columns)
df = df.drop(columns=['mlb_id', 'date'])

# Set seed for reproducing results
seed = 7
np.random.seed(seed)

def load_data(df, look_back=30):

    X, y = [], []

    # y_row = df.tail(1).as_matrix()
    # y.append(scale(y_row))

    # print('len - n_prev: {}'.format(len(df) - n_prev))
    # print('count - n_prev: {}'.format(df.mlb_id.count() - n_prev))
    # print('test: {}'.format(len(df.iloc[0:30])))
    # for i in range(len(df) - n_prev):
    #     X.append(scale(df.iloc[i:i+n_prev].as_matrix()))
    #     y.append(scale(df.iloc[i+n_prev].as_matrix()))
    # for i in range(len(df)-look_back+1):
    #     a = df[i:(i+look_back), :]
    #     X.append(a)
    #     y.append(df[i + look_back - 1, :])
    for i in range(len(df)-look_back-1):
        a = df[i:(i+look_back), 0]
        X.append(a)
        y.append(df[i + look_back, 0])
    # print(len(df))
    # for i in range(len(df)):
    #     if i != len(df):
    #         X.append(scale(df.iloc[i].as_matrix()))
    #     else:
    #         y.append(scale(df.iloc[i].as_matrix()))

    return np.array(X), np.array(y)


def scale(x, out_range=(-1, 1)):
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2


def train_test_split(df, test_size=0.1):
    """
    This just splits data to training and testing parts
    """
    # ntrn = round(len(df) * (1 - test_size))
    # print('ntrn: {}'.format(ntrn))
    n_train = 56
    print(len(df))
    X_train, y_train = load_data(df.iloc[0:n_train])
    X_test, y_test = load_data(df.iloc[n_train:])

    print('len X_train: {}\nlen X_test: {}\nlen y_train: {}\nlen y_test: {}'.format(len(X_train),
                                                                                    len(X_test),
                                                                                    len(y_train),
                                                                                    len(y_test)))

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split(df)


dim = 36

model = Sequential()
model.add(LSTM(dim, return_sequences=True, input_shape=(30, dim)))
model.add(BatchNormalization())
model.add(LSTM(dim, return_sequences=True))
model.add(BatchNormalization())
# model.add(Dense(dim, activation='relu'))
model.add(Dense(dim, activation='linear'))
model.compile(loss="mean_squared_error", optimizer="rmsprop", metrics=['accuracy'])

# model.fit(X_train, y_train, batch_size=20, nb_epoch=10, validation_split=0.05)
model.fit(X_train, y_train, batch_size=10, epochs=5)

print('len X_train: {}'.format(len(X_train)))
print('len X_test: {}'.format(len(X_test)))

predicted = model.predict(X_test)
rmse = np.sqrt(((predicted - y_test) ** 2).mean(axis=0))
print('rmse: {}'.format(rmse))
print('len predictd: {}'.format(len(predicted)))
print('len y_test: {}'.format(len(y_test)))
