from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
import math
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd

#######################
# Load Data
#######################
sc = SparkContext()
sqlcontext = SQLContext(sc)
full_data = sqlcontext.read.csv("s3://msia431spark/full_data.csv", header=True)

#######################
# Feature Engineering
#######################

# change data type
full_data = full_data.withColumn("bar_num", col("bar_num").cast('integer'))
full_data = full_data.withColumn("profit", col("profit").cast('double'))

full_data = full_data.withColumn("period", ceil(full_data.bar_num / 10))
full_data.persist()

schema = StructType([StructField('trade_id', StringType(), False),
                     StructField('period', IntegerType(), False),
                     StructField('weighted_avg', DoubleType(), False),
                     StructField('lag1', DoubleType(), False),
                     StructField('lag2', DoubleType(), False),
                     StructField('lag3', DoubleType(), False)],
                    )

results = sqlcontext.createDataFrame([], schema)
max_bar = full_data.groupby().agg(max('bar_num')).collect()[0][0]

for period in range(2, math.ceil(max_bar / 10) + 1):
    last_period = full_data.filter(col('period') < period)
    last_period = last_period.withColumn("length", max('bar_num').over(Window.partitionBy()))
    # weighted average
    last_period = last_period.withColumn("weight", pow(lit(0.95), col('length') - col('bar_num')))
    last_period = last_period.withColumn("weight_profit", col('weight') * col('profit'))
    weighted_avg = last_period.groupby('trade_id').agg(sum('weight').alias('weights'),
                                                       sum('weight_profit').alias('weighted_sum'))
    weighted_avg = weighted_avg.withColumn("weighted_avg", col('weighted_sum') / col('weights'))
    weighted_avg = weighted_avg.withColumn("period", lit(period))
    weighted_avg = weighted_avg[['trade_id', 'period', 'weighted_avg']]
    # the latest three profit from last period
    lags = last_period.filter(col('bar_num') >= (col('length') - 2)).groupby('trade_id').pivot('bar_num').mean('profit')
    new_features = weighted_avg.join(lags, on=['trade_id'], how='inner')
    results = results.union(new_features)

df_with_new_feature = full_data.join(results, on=['trade_id', 'period'], how='inner')

#######################
# Model
#######################

# Change data types

# Date
df_with_new_feature = df_with_new_feature.withColumn("date", to_date(
    from_unixtime(unix_timestamp('time_stamp', 'yyyy-MM-dd HH:mm:ss'))))

# Numeric
none_num = {'profit', 'trade_id', 'date', 'time_stamp', 'period', 'direction', 'bar_num'}
num_features = set(df_with_new_feature.columns) - none_num
for feature in num_features:
    df_with_new_feature = df_with_new_feature.withColumn(feature, col(feature).cast('double'))

# drop na
df_with_new_feature = df_with_new_feature.na.drop()

df_with_new_feature.persist()
full_data.unpersist()

# Random Forest
# PysparkML takes a vector of variable values rather than a list of columns
assembler = VectorAssembler(
    inputCols=list(num_features),
    outputCol='features')

# Define a Random Forest model.
rf = RandomForestRegressor(labelCol="profit", featuresCol="features")

# Chain indexer and GBT in a Pipeline
pipeline = Pipeline(stages=[assembler, rf])

# Get list of Test months
test_months = []
start = datetime.strptime('2008-07-01', '%Y-%m-%d').date()
end = datetime.strptime('2015-07-01', '%Y-%m-%d').date()

while start <= end:
    test_months.append(start)
    start += relativedelta(months=6)

# Train and evaluate
dic = {}
for month in test_months:
    train = df_with_new_feature.filter(
        (months_between(lit(month), col("date")) > 0) & (months_between(lit(month), col("date")) <= 6))
    test = df_with_new_feature.filter(
        (months_between(lit(month), col("date")) <= 0) & (months_between(lit(month), col("date")) > -1))
    model = pipeline.fit(train)
    test = model.transform(test)
    results = test[['profit', 'prediction']].withColumn('mape',
                                                        100 * abs((col('profit') - col('prediction')) / col('profit')))
    mape = results.groupby().mean('mape').collect()[0][0]
    dic[month] = mape

# Save MAPE
mape_all = sc.parallelize(dic.items()).toDF()
mape_all = mape_all.select(col("_1").alias("Year-Month"), col("_2").alias("mape"))
maximum = mape_all.select('mape').groupby().agg(max('mape')).collect()[0][0]
minimum = mape_all.select('mape').groupby().agg(min('mape')).collect()[0][0]
average = mape_all.select('mape').groupby().agg(mean('mape')).collect()[0][0]
newRow = sqlcontext.createDataFrame([('max', maximum), ('min', minimum), ('mean', average)], ['Year-Month', 'mape'])
output = mape_all.union(newRow)
output.toPandas().to_csv("s3://msia431spark/mape_all.csv", index=False)
