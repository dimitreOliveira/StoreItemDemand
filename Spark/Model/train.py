# Databricks notebook source
# MAGIC %md
# MAGIC Notebook for model training.

# COMMAND ----------

from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU
from pyspark.ml.evaluation import RegressionEvaluator

# COMMAND ----------

# MAGIC %run ../utils

# COMMAND ----------

train_transformed = spark.sql("select * from train_transformed")
validation_transformed = spark.sql("select * from validation_transformed")

# COMMAND ----------

train_x, train_y = prepare_collected_data(train_transformed.select('serie', 'sales').collect())
validation_x, validation_y = prepare_collected_data(validation_transformed.select('serie', 'sales').collect())

# COMMAND ----------

n_label = 1
serie_size = len(train_x[0])
n_features = len(train_x[0][0])

# COMMAND ----------

# hyperparameters
epochs = 80
batch = 512
lr = 0.001

# design network
model = Sequential()
model.add(GRU(40, input_shape=(serie_size, n_features)))
model.add(Dense(10, kernel_initializer='glorot_normal', activation='relu'))
model.add(Dense(n_label))
model.summary()

adam = optimizers.Adam(lr)
model.compile(loss='mae', optimizer=adam, metrics=['mse', 'msle'])

history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch, validation_data=(validation_x, validation_y), verbose=2, shuffle=False)

# COMMAND ----------

model_path = '/dbfs/user/model1.json'
weights_path = '/dbfs/user/weights1.npy'
save_model(model_path, weights_path, model)

# COMMAND ----------

predictions = model.predict(validation_x)

# COMMAND ----------

import pandas as pd
ids = validation_y
df = pd.DataFrame(ids, columns=['label'])
df['sales'] = predictions
df_predictions = spark.createDataFrame(df)

# COMMAND ----------

rmse_evaluator = RegressionEvaluator(labelCol="label", predictionCol="sales", metricName="rmse")
mse_evaluator = RegressionEvaluator(labelCol="label", predictionCol="sales", metricName="mse")
mae_evaluator = RegressionEvaluator(labelCol="label", predictionCol="sales", metricName="mae")

# COMMAND ----------

validation_rmse = rmse_evaluator.evaluate(df_predictions)
validation_mse = mse_evaluator.evaluate(df_predictions)
validation_mae = mae_evaluator.evaluate(df_predictions)
print("RMSE: %f, MSE: %f, MAE: %f" % (validation_rmse, validation_mse, validation_mae))
