import pandas as pd
import quandl
import math
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import datetime
import matplotlib.pyplot as plt
#from matplotlib import style
import pickle

#style.use('ggplot')

df = quandl.get('WIKI/GOOGL')

#print(df.head())

#print(df.iloc[:, 4:9])

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

#print(df.head())

df['HL'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low']
df['OC'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open']

df = df[['Adj. Close', 'HL', 'OC', 'Adj. Volume']]

#print(df.head())

forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.1*len(df)))

#print(forecast_out)
df['label'] = df[forecast_col].shift(-forecast_out)
#print(df)
#df.dropna(inplace=True)
#print(df)

X = np.array(df.drop(['label'], 1))
#print(X)
#print(y)
X = preprocessing.scale(X)
# print(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
# print(X)

df.dropna(inplace=True)
y = np.array(df['label'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)
# clf = svm.SVR()
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)

# saving the model to pickle
with open('saved_model.pickle', 'wb') as file:
	pickle.dump(clf, file)

# loading pickle
pickle_in = open('saved_model.pickle', 'rb')
clf = pickle.load(pickle_in)

confidence = clf.score(X_test, y_test)
#print(confidence)

forecast_set = clf.predict(X_lately)
df['Forecast'] = np.nan
# print(df['label'])
# print(forecast_set)

#df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
next_unix = last_unix + 86400
#print(df)
for i in forecast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += 86400
	df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]
#print(df)

# graph plotting
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=2)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

