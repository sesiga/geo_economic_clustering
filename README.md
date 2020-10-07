# Geo-Economic Clustering

In this project we classify all the countries in the world in 10 different groups by means of their GDP per capita and the distance among them.
The process of data gathering, the mathematical formulation and the results can be found in the file **task.pdf**. The file **task.py** contains the 
python implementation. A custom model of *clustering* has been developed, in order to compute the distance among countries given that they lie 
on the surface of a sphere. The files **countries.csv**, **gdp.csv** and **gdp_total** contain the data to feed the clustering algorithm, and results file is 
**geo_economic_clustering.csv**. It has 6 columns:

* *country*: the name of the country.
* *cluster*: the group it belongs to.
* *av_gdp*: the average GDP per capita of the cluster.
* *gdp*: the GDP per capita of the country.
* *gdpt*: the total GDP of the cluster. It is the sum of all the GDPs.
* *cum_gdp*: the fraction of the total GDP that the country has on the group.