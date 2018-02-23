import numpy as np
import pandas as pd
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense, Dropout
# from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
# from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler


df = pd.read_csv('data/pitcher_data.csv', index_col=0)
# df = pd.read_csv('data/batter_data.csv', index_col=0)
# df = df.drop('hand', axis=1)
# df = df.drop('oppt_pitch_hand', axis=1)
# df = df.drop('name_last_first', axis=1)

seed = 7
np.random.seed(seed)

X = df.drop('dk_points', axis=1)
y = df['dk_points']

dim = len(X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=7)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

test_ids = X_test['mlb_id'].reset_index().drop('index', axis=1)


X_train = X_train[['dk_salary', 'fd_salary', 'K%', 'ERA']]
X_test = X_test[['dk_salary', 'fd_salary', 'K%', 'ERA']]
# X_train = X_train[['K%', 'ERA']]
# X_test = X_test[['K%', 'ERA']]
# X_train = X_train[['adi', 'order', 'fd_salary', 'ERA']]
# X_test = X_test[['adi', 'order', 'fd_salary', 'ERA']]

# scaler = MinMaxScaler()
scaler = RobustScaler()
# scaler = StandardScaler()

scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test  = scaler.transform(X_test)

regr = RandomForestRegressor(max_depth=5, random_state=7, verbose=1)
regr.fit(scaled_X_train, y_train)
pred = regr.predict(scaled_X_test)
# score = regr.score(scaled_X_test, y_test, sample_weight=None)
# print(pred)
pred = pd.Series(pred).rename("Predictions")
output_df = pd.concat([test_ids, pred], axis=1)
output_df.to_csv('pitcher_predictions.csv')
print(output_df)


# print(X.columns[1])
# print(X.columns[2])
# print(X.columns[5])
# print(X.columns[34])
# print(X.columns[31])
# print(X.columns[32])

# model = Sequential()
# Shows some hope for hinge, MSLE
# model.add(Dense(dim, input_dim=dim, activation='relu'))
# model.add(Dense(dim, activation='relu'))
# model.add(Dense(dim, activation='relu'))
# model.add(Dense(1, activation='relu'))

# model.add(Dense(dim, input_dim=dim, activation='linear'))
# model.add(Dense(10, activation='linear'))
# model.add(Dense(9, activation='linear'))
# model.add(Dense(8, activation='linear'))
# model.add(Dense(7, activation='linear'))
# # model.add(Dropout(0.5, seed=7))
# model.add(Dense(6, activation='linear'))
# model.add(Dense(5, activation='linear'))
# model.add(Dense(1, activation='linear'))

# model.add(Dense(dim, input_dim=dim, activation='relu'))
# model.add(Dense(dim, activation='relu'))
# # model.add(Dropout(0.1, seed=7))
# model.add(Dense(dim, activation='linear'))
# model.add(Dense(1, activation='linear'))

# Loss improves for about ~40 epochs. NO idea how to interpret mean_squared_logarithmic_error
# model.compile(loss='mean_squared_logarithmic_error', optimizer='adam')
# model.compile(loss='mse', optimizer='adam')

# # model.fit(scaled_X_train, y_train, epochs=50)
# model.fit(scaled_X_train, y_train, epochs=10)

# model.evaluate(x=scaled_X_test, y=y_test, batch_size=None, verbose=1, sample_weight=None)
#
# print('model metrics: {}'.format(model.metrics_names))
# predictions = model.predict(scaled_X_test, verbose=1)
# i = 0
# total = 0
# total_y = 0
# for pred, y in zip(predictions, y_test):
#     # print('pred: {} ... y: {}'.format(pred, y))
#     i += 1
#     diff = abs(pred - y)
#     total += diff
#     total_y += y
#     # print(diff)
#
# print('average difference: {}'.format(total / i))
