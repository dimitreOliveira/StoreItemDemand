![](https://s9783.pcdn.co/wp-content/uploads/2018/03/Blog-Optimize-Store-Replenishment.jpg)

# Deep Learning regression with Keras and Spark

## About the repository
The Spark folder of this repository was written using Databricks if you want to replicate or continue the work you can checkout the free version [Databrick community](https://community.cloud.databricks.com/login.html).

The main goal of the repository is to use the Spark structure from [Databricks](https://databricks.com/) clusters, load and process data from the Kaggle competition and train deep learning models distributed.

### What you will find
* Brief EDA of the data set. [[link]](https://github.com/dimitreOliveira/StoreItemDemand/blob/master/Spark/Analysis/EDA.ipynb)
* Creation and usage of custom spark pipelines. [[link]](https://github.com/dimitreOliveira/StoreItemDemand/blob/master/Spark/custom_transformers.ipynb)
* Data preparation. [[link]](https://github.com/dimitreOliveira/StoreItemDemand/blob/master/Spark/prepare%20data.ipynb)
* Model training. [[link]](https://github.com/dimitreOliveira/StoreItemDemand/blob/master/Spark/Model/train.ipynb)
* Model prediction (test set). [[link]](https://github.com/dimitreOliveira/StoreItemDemand/blob/master/Spark/Model/test.ipynb)
* Model evaluation (evaluation of many different models. [[link]](https://github.com/dimitreOliveira/StoreItemDemand/blob/master/Spark/Model/model%20evaluation.ipynb)

### Store Item Demand Forecasting Challenge

link for the Kaggle competition: https://www.kaggle.com/c/demand-forecasting-kernels-only

datasets: https://www.kaggle.com/c/demand-forecasting-kernels-only/data

### Overview
This competition is provided as a way to explore different time series techniques on a relatively simple and clean dataset.

You are given 5 years of store-item sales data, and asked to predict 3 months of sales for 50 different items at 10 different stores.

What's the best way to deal with seasonality? Should stores be modeled separately, or can you pool them together? Does deep learning work better than ARIMA? Can either beat xgboost?

This is a great competition to explore different models and improve your skills in forecasting.

### Dependencies:
* [keras](https://keras.io/)
* [numpy](http://www.numpy.org/)
* [pandas](http://pandas.pydata.org/)
* [sklearn](https://scikit-learn.org/stable/)
* [matplotlib](http://matplotlib.org/)
* [pyspark.ml](http://spark.apache.org/docs/2.2.0/api/python/pyspark.ml.html)
* [pyspark.sql](http://spark.apache.org/docs/2.1.0/api/python/pyspark.sql.html)

### To-Do:
* Persistence of the pipeline classes needs to be fixed.
* Pipeline classes needs revised.
* The data probably needs more feature extraction.
