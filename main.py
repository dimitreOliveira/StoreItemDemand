import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler


def build_series(df, serie_size=3):
    for i in range(serie_size):
        for column in df.columns:
            df['%s%s' % (column, i)] = df[column].shift(i)


def prepare_data(df, serie_size=3):
    columns = ['store', 'item', 'day', 'month', 'year', 'weekday', 'weekend', 'monthbegin', 'monthend', 'yearquarter', 'filled_serie', 'rank']
    list_result = []
    for row in range(df.shape[0]):
        row_array = []
        for column in columns:
            for time in range(serie_size):
                row_array.append(df[column+'%s' % time].values)
        list_result.append(np.asarray(row_array))
    return np.asarray(list_result)


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


train = prepare_data(train)
validation = prepare_data(validation)

# train = train.iloc[:, (train.shape[1]-30):train.shape[1]].values
# validation = validation.iloc[:, (validation.shape[1]-30):validation.shape[1]].values


scaler = MinMaxScaler()
scaler.fit(train)
train_x = scaler.transform(train)
validation_x = scaler.transform(validation)
test_normalized = scaler.transform(test)

serie_size = len(train_x[0])
n_features = 1

model = Sequential()
model.add(GRU(32, recurrent_dropout=0.1, return_sequences=True, input_shape=(serie_size, n_features)))
model.add(Dropout(0.3))
model.add(GRU(32, recurrent_dropout=0.1, input_shape=(serie_size, n_features)))
model.add(Dropout(0.3))
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
