# Problem 1
### Chicago crime data
Less obvious fields
1. block = the first 5 characters correspond to the block code and the rest specify
the street location; 
2. IUCR = Illinois Uniform Crime Reporting code; 
3. X/Y coordinates = to visualize the data on a map, not needed in the assignment; 
4. District, Beat = police jurisdiction geographical partition; the region is partitioned in several districts; each district is partitioned in several beats;

### Task
By using SparkSQL, generate a histogram of average crime events by month. Find an explanation
of results.