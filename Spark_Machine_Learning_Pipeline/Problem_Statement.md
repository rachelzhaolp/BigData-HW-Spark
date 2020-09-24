# Problem 3

### Chicago crime data
Less obvious fields
1. block = the first 5 characters correspond to the block code and the rest specify
the street location; 
2. IUCR = Illinois Uniform Crime Reporting code; 
3. X/Y coordinates = to visualize the data on a map, not needed in the assignment; 
4. District, Beat = police jurisdiction geographical partition; the region is partitioned in several districts; each district is partitioned in several beats;

### Task
Predict the number of crime events in the next week at the beat level. Violent crime events
represent a greater threat to the public and thus it is desirable that they are forecasted more
accurately. You are encouraged to bring in additional data sets. (extra 10 pts if you mix the existing data
with an exogenous data set) Report the accuracy of your models. You must use Spark dataframes and ML pipelines.