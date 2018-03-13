# NOTE - needs work!
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

seed = 7
np.random.seed(seed)

X = df.drop('dk_points', axis=1)
y = df['dk_points']

dim = len(X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=7)

test_ids = X_test['mlb_id'].reset_index().drop('index', axis=1)


X_train = X_train[['dk_salary', 'fd_salary', 'K%', 'ERA']]
X_test = X_test[['dk_salary', 'fd_salary', 'K%', 'ERA']]

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
