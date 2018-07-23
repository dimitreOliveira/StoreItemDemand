# Databricks notebook source
# MAGIC %md
# MAGIC Notebook with auxiliary functions.

# COMMAND ----------

# delete file in dbfs
# %fs rm -r file_name

# see dbfs
# %sh ls /dbfs

# COMMAND ----------

import json
import numpy as np
from keras.models import model_from_json

# COMMAND ----------

unlist = lambda x: [float(i[0]) for i in x]

# COMMAND ----------

def prepare_data(data):
  list_result = []
  for i in range(len(data)):
    list_result.append(np.asarray(data[i]))
  return np.asarray(list_result)

# COMMAND ----------

def prepare_collected_data(data):
  list_features = []
  list_labels = []
  for i in range(len(data)):
    list_features.append(np.asarray(data[i][0]))
    list_labels.append(data[i][1])
  return np.asarray(list_features), np.asarray(list_labels)

# COMMAND ----------

def prepare_collected_data_test(data):
  list_features = []
  for i in range(len(data)):
    list_features.append(np.asarray(data[i][0]))
  return np.asarray(list_features)

# COMMAND ----------

def save_model(model_path, weights_path, model):
    """
    Save model.
    """
    np.save(weights_path, model.get_weights())
    with open(model_path, 'w') as f:
      json.dump(model.to_json(), f)

# COMMAND ----------

def load_model(model_path, weights_path):
  """
  Load model.
  """
  with open(model_path, 'r') as f:
     data = json.load(f)

  model = model_from_json(data)
  weights = np.load(weights_path)
  model.set_weights(weights)

  return model
