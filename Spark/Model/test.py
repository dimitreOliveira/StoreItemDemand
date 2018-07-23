# Databricks notebook source
# MAGIC %md
# MAGIC Notebook with test set model predictions

# COMMAND ----------

# MAGIC %run ../utils

# COMMAND ----------

model_path = '/dbfs/user/model4.json'
weights_path = '/dbfs/user/weights4.npy'
model = load_model(model_path, weights_path)

test_transformed = spark.sql("select * from test_transformed")

# COMMAND ----------

test = prepare_collected_data_test(test_transformed.select('serie').collect())

# COMMAND ----------

ids = test_transformed.select('id').collect()

# COMMAND ----------

predictions = model.predict(test)

# COMMAND ----------

import pandas as pd
df = pd.DataFrame(ids, columns=['id'])
df['sales'] = predictions
df_predictions = spark.createDataFrame(df)

# COMMAND ----------

df_predictions = df_predictions.withColumn('sales', df_predictions['sales'].cast('int'))
display(df_predictions)

# COMMAND ----------

# df_predictions = df_predictions.withColumn('sales', df_predictions['sales'].cast('int'))
# from pyspark.sql import types as T
# # df = spark.createDataFrame((ids, unlist(predictions)), T.StructType([T.StructField('ids',T.LongType(), True),T.StructField('sales',T.DoubleType(), True)]))
# df2 = spark.createDataFrame(unlist(predictions), T.StructType([T.StructField('ids',T.FloatType(), True)]))
# display(df2)
# display(df_predictions.select('id', 'sales'))

# COMMAND ----------

display(df_predictions.join(test_transformed, df_predictions['id'] == test_transformed['id']))

# COMMAND ----------

train_data = spark.sql("select * from store_item_demand_train_csv")
display(train_data)
