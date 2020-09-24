from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.sql.functions import *
from pyspark.sql.types import StringType
from pyspark.sql.window import Window
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import StringIndexer, VectorIndexer, OneHotEncoderEstimator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler


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
crime = crime.withColumn("Date", to_date(from_unixtime(unix_timestamp('Date', 'MM/dd/yyy hh:mm'))))
crime = crime.withColumn("Week", weekofyear(crime.Date).cast(StringType()))
crime = crime.withColumn("Year-Week", concat('Year', 'Week'))


def GBT(df):
    """
    Train Gradient Boosted Tree Model
    :param df: DataFrame(includes both training and test data)
    :return: R square of the model
    """
    weekly_cnt = df.groupby('Beat', "Year-Week", "Week").count().orderBy("Beat", "Year-Week").filter(
        col('Year-Week').isNotNull())  # drop invalid date

    # Feature engineering: create lag of count
    lag_weekly_cnt = weekly_cnt \
        .withColumn("lag1", lag('count').over(Window.partitionBy("Beat").orderBy('Year-Week'))) \
        .withColumn("lag2", lag('lag1').over(Window.partitionBy("Beat").orderBy('Year-Week'))) \
        .withColumn("lag3", lag('lag2').over(Window.partitionBy("Beat").orderBy('Year-Week'))) \
        .withColumn("lag4", lag('lag3').over(Window.partitionBy("Beat").orderBy('Year-Week'))) \
        .withColumn("lag5", lag('lag4').over(Window.partitionBy("Beat").orderBy('Year-Week'))) \
        .withColumn("lag6", lag('lag5').over(Window.partitionBy("Beat").orderBy('Year-Week'))) \
        .withColumn("lag7", lag('lag6').over(Window.partitionBy("Beat").orderBy('Year-Week'))) \
        .withColumn("lag8", lag('lag7').over(Window.partitionBy("Beat").orderBy('Year-Week')))

    lag_weekly_cnt = lag_weekly_cnt.na.drop()

    # Convert categorical variables to Factors
    BeatIdxer = StringIndexer(inputCol='Beat', outputCol='BeatIdx')
    WeekIdxer = StringIndexer(inputCol='Week', outputCol='WeekIdx')

    encoder = OneHotEncoderEstimator(inputCols=["BeatIdx", "WeekIdx"],
                                     outputCols=["BeatVec", "WeekVec"]).setHandleInvalid(
        "keep")  # , handleInvalid = 'keep')

    # PysparkML takes a vector of variable values rather than a list of columns
    assembler = VectorAssembler(
        inputCols=["lag1", "lag2", "lag3", "lag4", "lag5", "lag6", "lag7", "lag8", "BeatVec", "WeekVec"],
        outputCol='features')

    # Split the data into training and test sets (20% held out for testing)
    (trainingData, testData) = lag_weekly_cnt.randomSplit([0.8, 0.2])

    # Train a GBT model.
    gbt = GBTRegressor(labelCol="count", featuresCol="features", maxIter=10)

    # Chain indexer and GBT in a Pipeline
    pipeline = Pipeline(stages=[BeatIdxer, WeekIdxer, encoder, assembler, gbt])

    # Train model.  This also runs the indexer.
    model = pipeline.fit(trainingData)

    # Prediction.
    predictions = model.transform(testData)

    # Model evaluation with out of sample R square.
    evaluator = RegressionEvaluator(
        labelCol="count", predictionCol="prediction", metricName="r2")
    r2 = evaluator.evaluate(predictions)
    return r2


"""
All Crime
"""
all_r2 = GBT(crime)

"""
Violent Crime
"""
crime = crime.withColumn("Violent",
                         when(crime.IUCR.like('01%'), "HOMICIDE")
                         .when(crime.IUCR.like('02%'), "CRIMINAL SEXUAL ASSAULT")
                         .when(crime.IUCR.like('03%'), "ROBBERY")
                         .when(crime.IUCR.like('04%'), "BATTERY")
                         .when(crime.IUCR.like('05%'), "ASSAULT")
                         .when(crime.IUCR.isin(['1010', '1025']), "ARSON")
                         .otherwise("NON-VIOLENT"))
violence_crime = crime.filter(crime.Violent != "NON-VIOLENT")

violent_r2 = GBT(violence_crime)

"""
Write outputs to .txt
"""
file = open("exercise3.txt", "w")
file.write("Out of sample R Square of the Gradient Boosted Tree Model with all data is " + str(all_r2) + "\n")
file.write("Out of sample R Square of the Gradient Boosted Tree Model with violent crime data is " + str(violent_r2) + "\n")

file.close()
