from pyspark import SparkContext
import numpy as np
import re



"""
Load data from hdfs
"""
sc = SparkContext()
myRDD = sc.textFile("hdfs://wolf.analytics.private/user/lzp2080/data/crime/Crimes_-_2001_to_present.csv")
# Split columns
splitted_RDD = myRDD.map(lambda line: line.split(","))
splitted_RDD.persist()

"""
Q2.1 Find the top 10 blocks in crime events in the last 3 years(2017,2018,2019)
"""
Top_10_block = splitted_RDD.filter(lambda rec: rec[17] in ['2017', '2018', '2019']) \
                            .map(lambda rec: (rec[3][0:5], 1)) \
                            .reduceByKey(lambda a, b: a + b) \
                            .sortBy(lambda rec: -rec[1]) \
                            .take(10)

"""
Q2.2 Find the two beats that are adjacent with the highest correlation in the number of crime events
"""
# Calculate annual number of crime events by beats over the last 5 years
# Return: (beat_x,[# of Crime in 2015, # of Crime in 2016, # of Crime in 2017, # of Crime in 2018])
# In this dataset, all beats have at least one crime in the past 5 years. (If not, need to use join and fill 0)
beat_cnt = splitted_RDD.filter(lambda rec: rec[17] in ['2015', '2016', '2017', '2018', '2019'])\
                        .map(lambda rec: ((rec[10], rec[17]), 1))\
                        .reduceByKey(lambda a, b: a+b)\
                        .sortBy(lambda rec: (rec[0][0], rec[0][1]))\
                        .map(lambda rec: (rec[0][0], [rec[1]]))\
                        .reduceByKey(lambda a, b: a+b)

# Convert RDD to dictionary
beat_dict = beat_cnt.collectAsMap()
# Calculate correlation matrix
cor = np.corrcoef(list(beat_dict.values()))
beat = list(beat_dict.keys())

# Store the upper half of the correlation matrix to a list
results = []
for j in range(len(beat)):
    for i in range(len(beat)):
        if i > j:
            results.append([cor[j][i], beat[j], beat[i]])

results.sort(key=lambda x: abs(x[0]), reverse=True)
top_20 = results[:20]

"""
Q2.3 
Establish if the number of crime events is different between Mayor Daley and Emanuel at a granularity of 
<Annual homicide crime number> 
"""

Emanuel = ['2015', '2016', '2017', '2018', '2019']
Daley = ['2006', '2007', '2008', '2009', '2010']

Emanuel_cnt = splitted_RDD.filter(lambda rec: rec[17] in Emanuel)\
                          .filter(lambda rec: re.match("^01", rec[4]) is not None)\
                          .map(lambda rec: (rec[17],1))\
                          .reduceByKey(lambda a, b: a+b)\
                          .take(5)

Daley_cnt = splitted_RDD.filter(lambda rec: rec[17] in Daley)\
                          .filter(lambda rec: re.match("^01", rec[4]) is not None)\
                          .map(lambda rec: (rec[17], 1))\
                          .reduceByKey(lambda a, b: a+b)\
                          .take(5)

splitted_RDD.unpersist()

"""
Write outputs to .txt
"""
file = open("exercise2.txt", "w")
file.write("2(1): Find the top 10 blocks in crime events in the last 3 years; \n")
for i in Top_10_block:
    file.write(str(i) + "\n")

file.write("2(2): Find the two beats that are adjacent with the highest correlation in the number of crime events ("
           "this will require you looking at the map to determine if the correlated beats are adjacent to each other) "
           "over the last 5 years. \n")

for i in top_20:
    file.write(str(i) + "\n")

file.write("2(3): Establish if the number of crime events is different between Mayor Daley and Emanuel at a "
           "granularity of your choice (not only at the city level). Find an explanation of results.  \n")

file.write("Annual homicide crime number during the tenure of Mayor Emanuel.  \n")
for i in Emanuel_cnt:
    file.write(str(i) + "\n")

file.write("Annual homicide crime number during the tenure of Mayor Daley.  \n")
for i in Daley_cnt:
    file.write(str(i) + "\n")

file.close()


