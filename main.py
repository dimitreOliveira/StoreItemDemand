import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler


def prepare_data(data):
    list_result = []
    for i in range(len(data)):
        list_result.append(np.asarray(data[i]))
    return np.asarray(list_result)


def build_series(df, serie_size=30):
    for i in range(serie_size):
        df['t%s' % i] = df['id'].shift(i)


train = pd.read_csv('data/db/train.csv').drop(['date', 'dateFormated'], axis=1)
validation = pd.read_csv('data/db/validation.csv').drop(['date', 'dateFormated'], axis=1)
test = pd.read_csv('data/db/test.csv').drop(['date', 'dateFormated'], axis=1)

train_y = train['sales'].values
validation_y = validation['sales'].values
test_ids = test['id']
test = test.drop(['id'], axis=1)
train = train.drop(['sales'], axis=1)
validation = validation.drop(['sales'], axis=1)

build_series(train)
build_series(validation)


train = train.dropna()
validation = validation.dropna()


train = train.iloc[:, (train.shape[1]-30):train.shape[1]].values
validation = validation.iloc[:, (validation.shape[1]-30):validation.shape[1]].values


scaler = MinMaxScaler()
scaler.fit(train)
train_x = scaler.transform(train)
validation_x = scaler.transform(validation)
test_normalized = scaler.transform(test)

shape = train_x.shape[1]
serie_size = len(train_x[0])
n_features = len(train_x[0][0])

model = Sequential()
model.add(GRU(32, return_sequences= input_shape=(serie_size, n_features)))
model.add(Dropout(0.5))
model.add(Dense(64, kernel_initializer='glorot_normal', activation='relu'))
model.add(Dense(1))

model.summary()

adam = optimizers.Adam(0.01)
model.compile(loss='mae', metrics=['mse', 'msle'], optimizer=adam)

history = model.fit(train_x, train_y, validation_data=(validation_x, validation_y), epochs=50, batch_size=128, verbose=2, shuffle=False)

predictions = model.predict(test_normalized)

submission = pd.DataFrame(test_ids)
# submission.columns = ['sales']
# submission['id'] = test_ids
submission['sales'] = predictions
submission['sales'] = submission['sales'].astype(int)
submission.to_csv('submissions/submission.csv', encoding='utf-8', index=False)
