import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, BatchNormalization


# TODO - Read from command line parameters
df = pd.read_csv('data/20170718_batter_data.csv', index_col=0)

# Set seed for reproducing results
seed = 7
np.random.seed(seed)

X = df.drop('dk_points', axis=1)
y = df['dk_points']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=7)

test_ids = X_test['mlb_id'].reset_index().drop('index', axis=1)
test_salaries = X_test['dk_salary'].reset_index().drop('index', axis=1)
test_pos = X_test['fd_pos'].reset_index().drop('index', axis=1)

# scaler = MinMaxScaler()
scaler = RobustScaler()

scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test  = scaler.transform(X_test)

model = Sequential()

dim = len(X.columns)

model.add(Dense(dim, input_dim=dim, kernel_initializer='orthogonal', activation='relu'))
model.add(BatchNormalization())
model.add(Dense(dim, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1, seed=7))
model.add(BatchNormalization())
model.add(Dense(dim, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='linear'))

# Fit model, evaluate model, and generate predictions
# NOTE - Loss improves for about ~40 epochs. Not sure how to interpret mean_squared_logarithmic_error
model.compile(loss='mean_squared_logarithmic_error', optimizer='adam')

model.fit(scaled_X_train, y_train, epochs=15)

model.evaluate(x=scaled_X_test, y=y_test, batch_size=None, verbose=1, sample_weight=None)

predictions = model.predict(scaled_X_test, verbose=1)

# Calculate average difference between projected and actual points scored
i = 0
total = 0
total_y = 0
for pred, y in zip(predictions, y_test):
    i += 1
    diff = abs(pred - y)
    total += diff
    total_y += y

print('average difference: {}'.format(total / i))

# Generate output predictions file for batters
predictions = predictions.tolist()
predictions = pd.Series(predictions).rename("Predictions")
output_df = pd.concat([test_ids, predictions, test_salaries,test_pos], axis=1)
output_df.to_csv('batter_predictions.csv')
