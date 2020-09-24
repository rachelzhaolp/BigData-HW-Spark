# Problem 1
### Chicago crime data
Less obvious fields
1. block = the first 5 characters correspond to the block code and the rest specify
the street location; 
2. IUCR = Illinois Uniform Crime Reporting code; 
3. X/Y coordinates = to visualize the data on a map, not needed in the assignment; 
4. District, Beat = police jurisdiction geographical partition; the region is partitioned in several districts; each district is partitioned in several beats;

### Task
By using plain Spark (RDDs): 
1. Find the top 10 blocks in crime events in the last 3 years; 
2. Find the two beats that are adjacent with the highest correlation in the number of crime events (this 
will require you looking at the map to determine if the correlated beats are adjacent to each
other) over the last 5 years 
3. Establish if the number of crime events is different between Majors Daly and Emanuel at a granularity of your choice (not only at the city level). Find an
explanation of results.