import pandas as pd
import numpy as np
import calendar
from datetime import datetime
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer

pd.options.display.width = 0
pd.set_option('display.max_rows', 800)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Append the data into one for feature engineering
data_all = train.append(test)
data_all.reset_index(inplace=True)
data_all.drop('index', inplace=True, axis=1)

# Split datetime into years, months, days, hours, weekdays
data_all['datetemp'] = data_all['datetime'].apply(lambda x: x.split())
data_all['year'] = data_all['datetemp'].apply(lambda x: x[0].split('-')[0])
data_all['month'] = data_all['datetemp'].apply(lambda x: x[0].split('-')[1])
data_all['day'] = data_all['datetemp'].apply(lambda x: x[0].split('-')[2])
data_all['hour'] = data_all['datetemp'].apply(lambda x: x[1].split(':')[0])
data_all['weekday'] = data_all['datetemp'].apply(lambda x: calendar.day_abbr[datetime.strptime(x[0], '%Y-%m-%d').weekday()])

# Replace the data of windspeed with 0 
windspeed0 = data_all[data_all['windspeed'] == 0]
windspeednot0 = data_all[data_all['windspeed'] !=0 ]
wind_rfr = RandomForestRegressor()
cols = ['humidity', 'season', 'temp', 'atemp', 'weather', 'year', 'month', 'day']
wind_rfr.fit(windspeednot0[cols], windspeednot0['windspeed'])
wind_rfr_score = wind_rfr.predict(windspeed0[cols])
windspeed0['windspeed'] = wind_rfr_score

data_all = windspeednot0.append(windspeed0)
data_all.reset_index(inplace=True)
data_all.drop('index', axis=1, inplace=True)

categorical_features = ['holiday', 'season', 'weather', 'workingday', 'year', 'month', 'hour', 'weekday']
numerical_features = ['atemp', 'humidity', 'temp', 'windspeed']

# Map the weekdays into numerical values
weekday_map = {'Mon':0, 'Tue':1, 'Wed':2, 'Thu':3, 'Fri':4, 'Sat':5, 'Sun':6}
data_all['weekday'] = data_all['weekday'].map(weekday_map)

for var in categorical_features:
	data_all[var] = data_all[var].astype("category")

drop_features = ['casual', 'registered', 'datetime', 'count', 'day', 'datetemp']

# Divide the data back into train and testing
train = data_all[pd.notnull(data_all['count'])].sort_values(by=['datetime'])
test = data_all[~pd.notnull(data_all['count'])].sort_values(by=['datetime'])
datetimecol = test['datetime']
y_train = train['count']

# Drop Useless features
train = train.drop(drop_features, axis=1)
test = test.drop(drop_features, axis=1)

# Rmsle scorer
def rmsle(predict, actual):
	predict = np.array(predict)
	actual = np.array(actual)

	plog = np.log(predict + 1)
	alog = np.log(actual + 1)

	dif = (plog-alog)**2
	return np.sqrt(np.mean(dif))

rmsle_scorer = make_scorer(rmsle)

# Cross validation initialization
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

# Modeling
# Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0)
rf_reg.fit(train, y_train)
rf_reg_cvs = cross_val_score(rf_reg, train, y_train, cv=k_fold, scoring=rmsle_scorer).mean()
print(rf_reg_cvs)

"""
predictions = rf_reg.predict(test)

submission = pd.read_csv('sampleSubmission.csv')
submission['count'] = predictions

submission.to_csv("Score_{0:.5f}_submission.csv".format(rf_reg_cvs), index=False)
"""

