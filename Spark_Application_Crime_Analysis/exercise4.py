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
Feature engineering
"""
crime = crime.withColumn("Date", from_unixtime(unix_timestamp(crime.Date, 'MM/dd/yyy hh:mm:ss a')))
crime = crime.withColumn("Hour", hour('Date'))
crime = crime.withColumn("DayofWeek", dayofweek('Date'))
crime = crime.withColumn("Month", month('Date'))

"""
Calculate total number of crimes with arrest by hour/day of week/month
"""
hourly_cnt = crime.filter(crime.Arrest == "true").groupby("Hour").count() \
    .filter(col('Hour').isNotNull()).orderBy("Hour").toPandas()
daily_cnt = crime.filter(crime.Arrest == "true").groupby("DayofWeek").count() \
    .filter(col('DayofWeek').isNotNull()).orderBy("DayofWeek").toPandas()
monthly_cnt = crime.filter(crime.Arrest == "true").groupby("Month").count() \
    .filter(col('Month').isNotNull()).orderBy("Month").toPandas()

"""
Write outputs to .txt
"""
file = open("exercise4.txt", "w")

file.write("Total number of crimes with arrest by hour \n")
print("Hour" + ":" + "count")
for index, row in hourly_cnt.iterrows():
    file.write(str(row['Hour']) + ":" + str(row['count']) + "\n")

file.write("Total number of crimes with arrest by day of the week  \n")
print("DayofWeek" + ":" + "count")
for index, row in daily_cnt.iterrows():
    file.write(str(row['DayofWeek']) + ":" + str(row['count']) + "\n")

file.write("Total number of crimes with arrest by month  \n")
print("Month" + ":" + "count")
for index, row in monthly_cnt.iterrows():
    file.write(str(row['Month']) + ":" + str(row['count']) + "\n")

file.close()

"""
Plot
"""
fig, axs = plt.subplots(1, 3, figsize=(20, 5))
axs[0].plot(hourly_cnt["Hour"], hourly_cnt["count"])
axs[1].plot(daily_cnt["DayofWeek"], daily_cnt["count"])
axs[2].plot(monthly_cnt["Month"], monthly_cnt["count"])

plt.setp(axs[0], xlabel='Hours')
plt.setp(axs[1], xlabel='DayofWeeks')
plt.setp(axs[2], xlabel='Months')
plt.setp(axs[0:3], ylabel='Count')
fig.savefig("exercise4.png")

