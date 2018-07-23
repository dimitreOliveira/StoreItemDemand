# Databricks notebook source
# MAGIC %md
# MAGIC Notebook with the custom spark ml transformers.

# COMMAND ----------

from pyspark.sql import Window
from pyspark.ml import Pipeline
from pyspark.sql.types import *
from pyspark.sql import types as T
from pyspark.sql.functions import udf
from pyspark.sql import functions as F
from pyspark.ml.pipeline import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol

# COMMAND ----------

days = lambda i: i * 86400
get_weekday = udf(lambda x: x.weekday())
serie_has_null = F.udf(lambda x: reduce((lambda x, y: x and y), x))

# COMMAND ----------

class DateConverter(Transformer):
  def __init__(self, inputCol, outputCol):
    self.inputCol = inputCol
    self.outputCol = outputCol
  
  def this():
    this(Identifiable.randomUID("dateconverter"))

  def copy(extra):
    defaultCopy(extra)
    
  def check_input_type(self, schema):
    field = schema[self.inputCol]
    if (field.dataType != TimestampType()):
      raise Exception('Input type %s did not match input type TimestampType' % field.dataType)

  def _transform(self, df):
    self.check_input_type(df.schema)
    return df.withColumn(self.outputCol, df.date.cast(self.inputCol))

# COMMAND ----------

class DayExtractor(Transformer):
  def __init__(self, inputCol, outputCol='day'):
    self.inputCol = inputCol
    self.outputCol = outputCol
  
  def this():
    this(Identifiable.randomUID("dayextractor"))

  def copy(extra):
    defaultCopy(extra)
    
  def check_input_type(self, schema):
    field = schema[self.inputCol]
    if (field.dataType != DateType()):
      raise Exception('DayExtractor input type %s did not match input type DateType' % field.dataType)

  def _transform(self, df):
    self.check_input_type(df.schema)
    return df.withColumn(self.outputCol, F.dayofmonth(df[self.inputCol]))

# COMMAND ----------

class MonthExtractor(Transformer):
  def __init__(self, inputCol, outputCol='month'):
    self.inputCol = inputCol
    self.outputCol = outputCol
  
  def this():
    this(Identifiable.randomUID("monthextractor"))

  def copy(extra):
    defaultCopy(extra)
    
  def check_input_type(self, schema):
    field = schema[self.inputCol]
    if (field.dataType != DateType()):
      raise Exception('MonthExtractor input type %s did not match input type DateType' % field.dataType)

  def _transform(self, df):
    self.check_input_type(df.schema)
    return df.withColumn(self.outputCol, F.month(df[self.inputCol]))

# COMMAND ----------

class YearExtractor(Transformer):
  def __init__(self, inputCol, outputCol='year'):
    self.inputCol = inputCol
    self.outputCol = outputCol
  
  def this():
    this(Identifiable.randomUID("yearextractor"))

  def copy(extra):
    defaultCopy(extra)
    
  def check_input_type(self, schema):
    field = schema[self.inputCol]
    if (field.dataType != DateType()):
      raise Exception('YearExtractor input type %s did not match input type DateType' % field.dataType)

  def _transform(self, df):
    self.check_input_type(df.schema)
    return df.withColumn(self.outputCol, F.year(df[self.inputCol]))

# COMMAND ----------

class WeekDayExtractor(Transformer):
  def __init__(self, inputCol, outputCol='weekday'):
    self.inputCol = inputCol
    self.outputCol = outputCol
  
  def this():
    this(Identifiable.randomUID("weekdayextractor"))

  def copy(extra):
    defaultCopy(extra)
    
  def check_input_type(self, schema):
    field = schema[self.inputCol]
    if (field.dataType != DateType()):
      raise Exception('WeekDayExtractor input type %s did not match input type DateType' % field.dataType)

  def _transform(self, df):
    self.check_input_type(df.schema)
    return df.withColumn(self.outputCol, get_weekday(df[self.inputCol]).cast('int'))

# COMMAND ----------

class WeekendExtractor(Transformer):
  def __init__(self, inputCol='weekday', outputCol='weekend'):
    self.inputCol = inputCol
    self.outputCol = outputCol
  
  def this():
    this(Identifiable.randomUID("weekdayextractor"))

  def copy(extra):
    defaultCopy(extra)
    
  def check_input_type(self, schema):
    field = schema[self.inputCol]
    if (field.dataType != IntegerType()):
      raise Exception('WeekendExtractor input type %s did not match input type IntegerType' % field.dataType)

  def _transform(self, df):
    self.check_input_type(df.schema)
    return df.withColumn(self.outputCol, F.when(((df[self.inputCol] == 5) | (df[self.inputCol] == 6)), 1).otherwise(0))

# COMMAND ----------

class DataLabeler(Transformer):
  def __init__(self, inputCol, outputCol='label', dateCol='date', idCol=['store', 'item']):
    self.inputCol = inputCol
    self.outputCol = outputCol
    self.dateCol = dateCol
    self.idCol = idCol
  
  def this():
    this(Identifiable.randomUID("datalabeler"))

  def copy(extra):
    defaultCopy(extra)
    
  def check_input_type(self, schema):
    field = schema[self.inputCol]
    if (field.dataType != IntegerType()):
      raise Exception('DataLabeler input type %s did not match input type IntegerType' % field.dataType)

  def _transform(self, df):
    self.check_input_type(df.schema)
    window = Window().partitionBy(self.idCol).orderBy(df[self.dateCol].cast('timestamp').cast('long')).rangeBetween(0, days(30))
    
    return df.withColumn(self.outputCol, F.last(df[self.inputCol]).over(window))

# COMMAND ----------

class SerieMaker(Transformer):
  def __init__(self, inputCol='scaledFeatures', outputCol='serie', dateCol='date', idCol=['store', 'item'], serieSize=30):
    self.inputCol = inputCol
    self.outputCol = outputCol
    self.dateCol = dateCol
    self.serieSize = serieSize
    self.idCol = idCol
  
  def this():
    this(Identifiable.randomUID("seriemaker"))

  def copy(extra):
    defaultCopy(extra)

  def _transform(self, df):
    window = Window.partitionBy(self.idCol).orderBy(self.dateCol)
    series = []   
  
    df = df.withColumn('filled_serie', F.lit(0))
    
    for index in reversed(range(0, self.serieSize)):
      window2 = Window.partitionBy(self.idCol).orderBy(self.dateCol).rowsBetween((30 - index), 30)
      col_name = (self.outputCol + '%s' % index)
      series.append(col_name)
      df = df.withColumn(col_name, F.when(F.isnull(F.lag(F.col(self.inputCol), index).over(window)), F.first(F.col(self.inputCol), ignorenulls=True).over(window2)).otherwise(F.lag(F.col(self.inputCol), index).over(window)))
      df = df.withColumn('filled_serie', F.when(F.isnull(F.lag(F.col(self.inputCol), index).over(window)), (F.col('filled_serie') + 1)).otherwise(F.col('filled_serie')))

    df = df.withColumn('rank', F.rank().over(window))
    df = df.withColumn(self.outputCol, F.array(*series))
    
    return df.drop(*series)

# COMMAND ----------

class MonthBeginExtractor(Transformer):
  def __init__(self, inputCol='day', outputCol='monthbegin'):
    self.inputCol = inputCol
    self.outputCol = outputCol
  
  def this():
    this(Identifiable.randomUID("monthbeginextractor"))

  def copy(extra):
    defaultCopy(extra)
    
  def check_input_type(self, schema):
    field = schema[self.inputCol]
    if (field.dataType != IntegerType()):
      raise Exception('MonthBeginExtractor input type %s did not match input type IntegerType' % field.dataType)

  def _transform(self, df):
    self.check_input_type(df.schema)
    return df.withColumn(self.outputCol, F.when((df[self.inputCol] <= 7), 1).otherwise(0))

# COMMAND ----------

class MonthEndExtractor(Transformer):
  def __init__(self, inputCol='day', outputCol='monthend'):
    self.inputCol = inputCol
    self.outputCol = outputCol
  
  def this():
    this(Identifiable.randomUID("monthendextractor"))

  def copy(extra):
    defaultCopy(extra)
    
  def check_input_type(self, schema):
    field = schema[self.inputCol]
    if (field.dataType != IntegerType()):
      raise Exception('MonthEndExtractor input type %s did not match input type IntegerType' % field.dataType)

  def _transform(self, df):
    self.check_input_type(df.schema)
    return df.withColumn(self.outputCol, F.when((df[self.inputCol] >= 24), 1).otherwise(0))

# COMMAND ----------

class YearQuarterExtractor(Transformer):
  def __init__(self, inputCol='month', outputCol='yearquarter'):
    self.inputCol = inputCol
    self.outputCol = outputCol
  
  def this():
    this(Identifiable.randomUID("yearquarterextractor"))

  def copy(extra):
    defaultCopy(extra)
    
  def check_input_type(self, schema):
    field = schema[self.inputCol]
    if (field.dataType != IntegerType()):
      raise Exception('YearQuarterExtractor input type %s did not match input type IntegerType' % field.dataType)

  def _transform(self, df):
    self.check_input_type(df.schema)
    return df.withColumn(self.outputCol, F.when((df[self.inputCol] <= 3), 0)
                               .otherwise(F.when((df[self.inputCol] <= 6), 1)
                                .otherwise(F.when((df[self.inputCol] <= 9), 2)
                                 .otherwise(3))))
