from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.sql.functions import *
from matplotlib import pyplot as plt

"""
Load data from hdfs
"""
sc = SparkContext()
sqlcontext = SQLContext(sc)
crime_path = 'hdfs://wolf.analytics.private/user/lzp2080/data/crime/Crimes_-_2001_to_present.csv'
crime = sqlcontext.read.csv(crime_path, header=True)

"""
Feature Engineering
"""
crime = crime.withColumn("Month", crime.Date.substr(0, 2))
crime = crime.withColumn("Year-Month", concat(crime.Date.substr(7, 4), crime.Date.substr(0, 2)))

"""
Average crime events by month
"""
cnt_crime = crime.groupby('Year-Month', 'Month').count().groupby('Month').mean().orderBy('Month')

"""
Plot
"""
cnt_crime.toPandas().plot.bar(x="Month", y="avg(count)")
plt.savefig("exercise1.png")
