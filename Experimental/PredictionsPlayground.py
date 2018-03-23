import pandas as pd

pred = pd.read_csv('batter_predictions_all.csv')
# print(pred.columns)
# pred = pred.drop(['mlb_id','dk_salary','fd_pos'], axis=1)
# pred['Predictions'] = pred['Predictions'].apply(lambda x: x[0])
pred['dk_points'] = pred['dk_points'].apply(lambda x: float(x))
# df['Date'] = df['Date'].apply(lambda x: int(str(x)[-4:]))
print(pred.head())
