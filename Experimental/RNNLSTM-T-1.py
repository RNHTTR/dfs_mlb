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
df = df.sort_values(by='date')
# df = df.drop(columns=['mlb_id', 'date'])
df = df[['dk_points', 'dk_salary', 'hand_advantage']]
df['y'] = df['dk_points'].shift(-1)
# print(df[['dk_points', 'y']])
# quit()
df = df.dropna()
values = df.values
values = values.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1))
values = scaler.fit_transform(values)
# print(df.head())
# quit()

n_train_hours = 75
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]

# split into input and outputs
X_train, y_train = train[:, :-1], train[:, -1]
X_test, y_test = test[:, :-1], test[:, -1]

# print('X_train shape: {}\ny_train shape: {}\nX_test shape: {}\ny_test shape: {}'.format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))
# quit()

# reshape input to be 3D [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

print('X_train shape: {}\ny_train shape: {}\nX_test shape: {}\ny_test shape: {}'.format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))
quit()
# model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))

model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(BatchNormalization())
model.add(Dense(1, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='linear'))
model.compile(loss="mean_squared_error", optimizer="rmsprop", metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=1, epochs=50)

# make a prediction
yhat = model.predict(X_test)

X_test = X_test.reshape((X_test.shape[0], X_test.shape[2]))
# # invert scaling for forecast
# inv_yhat = np.concatenate((yhat, X_test[:, 1:]), axis=1)
# inv_yhat = scaler.inverse_transform(inv_yhat)
# print(inv_yhat)
# inv_yhat = inv_yhat[:,0]
# # invert scaling for actual
# y_test = y_test.reshape((len(y_test), 1))
# inv_y = np.concatenate((y_test, X_test[:, 1:]), axis=1)
# inv_y = scaler.inverse_transform(inv_y)
# inv_y = inv_y[:,0]
# # calculate RMSE
# rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
# print('Test RMSE: %.3f' % rmse)

# dim = 36
#
# model = Sequential()
# model.add(LSTM(dim, return_sequences=True, input_shape=(30, dim)))
# model.add(BatchNormalization())
# model.add(LSTM(dim, return_sequences=True))
# model.add(BatchNormalization())
# # model.add(Dense(dim, activation='relu'))
# model.add(Dense(dim, activation='linear'))
# model.compile(loss="mean_squared_error", optimizer="rmsprop", metrics=['accuracy'])

# model.fit(X_train, y_train, batch_size=20, nb_epoch=10, validation_split=0.05)
model.fit(X_train, y_train, batch_size=1, epochs=10)
