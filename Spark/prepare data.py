# Databricks notebook source
# MAGIC %md
# MAGIC #### Notebook for data preparation.
# MAGIC #### Current pipeline:
# MAGIC * Feature extraction
# MAGIC   * Cast date
# MAGIC   * Extract day
# MAGIC   * Extract Month
# MAGIC   * Extract Year
# MAGIC   * Extract Week day
# MAGIC   * Extract if day is weekend
# MAGIC * Normalize data
# MAGIC   * Min-max scaler
# MAGIC * Create series (each serie is 1 months, series alternate days)

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import MinMaxScaler

# COMMAND ----------

# MAGIC %run ./custom_transformers

# COMMAND ----------

train_data = spark.sql("select * from store_item_demand_train_csv")

# COMMAND ----------

train, validation = train_data.randomSplit([0.8,0.2], seed=1234)

# COMMAND ----------

# Feature extraction
dc = DateConverter(inputCol='date', outputCol='dateFormated')
dex = DayExtractor(inputCol='dateFormated')
mex = MonthExtractor(inputCol='dateFormated')
yex = YearExtractor(inputCol='dateFormated')
wdex = WeekDayExtractor(inputCol='dateFormated')
wex = WeekendExtractor()
mbex = MonthBeginExtractor()
meex = MonthEndExtractor()
yqex = YearQuarterExtractor()

# Data process
#tentar fazer 'day', 'month', 'year', 'weekday', 'weekend' (as colunas derivadas) ficarem de forma dinâmica, no lugar delas ficar a saída de seu respectivo transformer
va = VectorAssembler(inputCols=['store', 'item', 'day', 'month', 'year', 'weekday', 'weekend', 'monthbegin', 'monthend', 'yearquarter'], outputCol="features")
# scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)
scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")

# Serialize data
sm = SerieMaker(inputCol='scaledFeatures', dateCol='date', idCol=['store', 'item'], serieSize=5)

pipeline = Pipeline(stages=[dc, dex, mex, yex, wdex, wex, mbex, meex, yqex, va, scaler, sm])

# COMMAND ----------

pipiline_model = pipeline.fit(train)

# COMMAND ----------

train_transformed = pipiline_model.transform(train)
validation_transformed = pipiline_model.transform(validation)

# COMMAND ----------

# train_transformed = train_transformed.filter(F.col('filled_serie') == 0)
# validation_transformed = validation_transformed.filter(F.col('filled_serie') == 0)

# train_transformed = train_transformed.filter(F.col('rank') % 3 == 0)
# validation_transformed = validation_transformed.filter(F.col('rank') % 3 == 0)

# COMMAND ----------

# pipeline_path = '/dbfs/user/pipeline'
# pipiline_model.save(pipeline_path)
# pipiline_model.load(pipeline_path)

# COMMAND ----------

train_transformed.write.saveAsTable('train_transformed', mode='overwrite')
validation_transformed.write.saveAsTable('validation_transformed', mode='overwrite')

# COMMAND ----------

test_data = spark.sql("select * from store_item_demand_test_csv")
test_transformed = pipiline_model.transform(test_data)
test_transformed.write.saveAsTable('test_transformed', mode='overwrite')

# COMMAND ----------

print('Train raw: %s' % train.count())
print('Validation raw: %s' % validation.count())
print('Test raw: %s' % test_data.count())

# COMMAND ----------

print('Train transformed: %s' % train_transformed.count())
print('Validation transformed: %s' % validation_transformed.count())
print('Test transformed: %s' % test_transformed.count())
