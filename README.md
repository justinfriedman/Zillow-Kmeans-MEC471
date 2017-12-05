# Zillow-Kmeans-MEC471
MEC 471 final project work

Stata is the standard statistics software package in the department so I think it is most productive for my peers to maintain a Stata only workflow. I was able to utlize the built in Kmeans clustering, bootstraping and stepwise OLS functionality in Stata to complete my analysis. 


The data set includes k = 5 clusters. 

classifier.py is takes in two .dta files and converts them to pandas dataframes for processing. First, the clustered train data is used to gererate average values for each feature by cluster. The script then uses a brute force means classification method to classify obeservations in the test data set. Two modified sets are then exported. 

Presentation: https://docs.google.com/presentation/d/1iA1iyKvXwrSM-J5rn4DFMRylGOj3-u0fBTjQgB5y_Ts/edit?usp=sharing

![alt text](https://raw.githubusercontent.com/justinfriedman/Zillow-Kmeans-MEC471/master/Pipeline%20Diagram.png "Data Pipeline")
