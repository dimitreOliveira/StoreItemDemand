# Databricks notebook source
# MAGIC %md
# MAGIC ## Notebook for EDA
# MAGIC 
# MAGIC * challenge link: https://www.kaggle.com/c/demand-forecasting-kernels-only

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.functions import when, count, col

# COMMAND ----------

train_data = spark.sql("select * from store_item_demand_train_csv")

# COMMAND ----------

display(train_data)

# COMMAND ----------

train_data.select([count(when(col(c).isNull(), c)).alias(c) for c in train_data.columns]).show()

# COMMAND ----------

print('Row count', train_data.count())
print('Store count', train_data.select('store').distinct().count())
print('Item count', train_data.select('item').distinct().count())
print('Total sales', train_data.agg(F.sum(train_data.sales)).collect()[0][0])

# COMMAND ----------

display(train_data)

# COMMAND ----------

display(train_data)

# COMMAND ----------

display(train_data)

# COMMAND ----------

display(train_data.groupBy('store').count().orderBy('store'))

# COMMAND ----------

display(train_data.groupBy('item').count().orderBy('item'))

# COMMAND ----------

display(train_data.groupBy('store', 'item').count().orderBy('store', 'item'))
