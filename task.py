#-----------------------------
#import libraries
#-----------------------------
import pandas as pd 
import numpy as np 
import math

#-----------------------------
#load datasets
#-----------------------------

#gdp per capita file
gdp_path = r'C:\Users\sesig\Documents\tests_for_jobs\infinia_mobile\gdp.csv'
gdp_cols = ['Country Name', '2019 [YR2019]']
gdp_dtype = {'Country Name':str, '2019 [YR2019]':np.float64}
gdp = pd.read_csv(gdp_path, sep=',', usecols=gdp_cols, dtype=gdp_dtype, na_values='..')

#gdp file
gdpt_path = r'C:\Users\sesig\Documents\tests_for_jobs\infinia_mobile\gdp_total.csv'
gdpt_cols = ['Country Name', '2019 [YR2019]']
gdpt_dtype = {'Country Name':str, '2019 [YR2019]':np.float64}
gdpt = pd.read_csv(gdpt_path, sep=',', usecols=gdpt_cols, dtype=gdpt_dtype, na_values='..')

#location file
loc_path = r'C:\Users\sesig\Documents\tests_for_jobs\infinia_mobile\countries.csv'
loc_cols = ['latitude', 'longitude', 'name']
loc_dtype = {'latitude':np.float64, 'longitude':np.float64, 'name':str}
loc = pd.read_csv(loc_path, sep=',', usecols=loc_cols, dtype=loc_dtype)

#-----------------------------
#data pre processing
#-----------------------------

#remove countries with no data about gdp or location
gdp.dropna(inplace=True)
loc.dropna(inplace=True)
gdpt.dropna(inplace=True)

#merge all data frames
gdp_rename_col = {'Country Name':'country', '2019 [YR2019]':'gdp'}
gdp.rename(columns=gdp_rename_col, inplace=True)

loc_rename_col = {'name':'country'}
loc.rename(columns=loc_rename_col, inplace=True)
data_aux = gdp.merge(loc, how='inner', on='country')

gdpt_rename_col = {'Country Name':'country', '2019 [YR2019]':'gdpt'}
gdpt.rename(columns=gdpt_rename_col, inplace=True)
data = data_aux.merge(gdpt, how='inner', on='country')

# #export final data set into csv
# path_out = r'C:\Users\sesig\Documents\tests_for_jobs\infinia_mobile\data_processed.csv'
# data.to_csv(path_out, sep=',', index=False)

#-----------------------------
#function to group the countries
#-----------------------------

def f_group(x, max_iter):

    #-----------------------------
    #definition of arrays and variables
    #-----------------------------

    nrow = x.shape[0]
    earth_perimeter = 20016 #half perimeter

    #data of the 10 clusters
    #columns: gdp, latitude, longitude
    #for reproducibility
    np.random.seed(3)
    cluster = np.random.rand(10,3)
    cluster[:,1] *= 90.
    cluster[:,2] *= 180.

    #distance between countries and the cluster they belong to
    dist = np.zeros(nrow)

    #to which cluster belongs each country
    clas = np.zeros(nrow, dtype=np.int)

    #number of countries per cluster
    no_countries = np.zeros(10, dtype=np.int)

    #-----------------------------
    #data processing
    #-----------------------------

    #convert gps coordinates from degrees to radians
    x[:,1] *= math.pi / 180.
    x[:,2] *= math.pi / 180.

    #normalize gdp
    max_gdp = max(x[:,0])
    min_gdp = min(x[:,0])
    max_min_gdp = max_gdp - min_gdp
    x[:,0] = (x[:,0] - min_gdp) / max_min_gdp

    #coordinates will not be normalized but distance

    #-----------------------------
    #algo
    #-----------------------------

    #whithin sum of squares. distance to minimize
    wss = 1e10
    for k in range(max_iter):

        #compute the distance between clusters and countries. Assign countries to clusters
        no_countries[:] = 0
        for i in range(nrow):

            #latitude and longitude of country
            lat1 = x[i,1]
            long1 = x[i,2]

            #gdp of the country
            gdp1 = x[i,0]

            #compute the distance between countries and clusters to determine to 
            #which cluster belongs each country
            country_dist = 1e10            
            for j in range(10): 

                #latitude and longitude of the cluster
                lat2 = cluster[j,1]
                long2 = cluster[j,2]

                #gdp of the cluster
                gdp2 = cluster[j,0]

                #normalized physical distance of country i to cluster j
                country_gps_dist = 6371. * math.acos( math.sin(lat1) * math.sin(lat2) + 
                                    math.cos(lat1) * math.cos(lat2) * math.cos(long2 - long1) )

                country_gps_dist /= earth_perimeter

                #distance between country and cluster
                country_dist_aux = np.sqrt(country_gps_dist ** 2 + (gdp1 - gdp2) ** 2)

                #if country is closer to cluster update information
                if country_dist_aux < country_dist:

                    #distance between cluster and country
                    country_dist = country_dist_aux

                    #cluster number
                    no_clas = j

            #cluster j for country i
            clas[i] = no_clas

            #sums one more country to cluster j
            no_countries[no_clas] += 1

            #update distance between country i and its cluster
            dist[i] = country_dist

        #recompute the new centroid of the clusters with the countries belonging to them

        #initialize the clusters
        cluster[:,:] = 0.0
        for i in range(nrow):

            #cluster of country i
            country_clas = clas[i]

            #update centroid of cluster
            cluster[country_clas,0] += x[i,0]
            cluster[country_clas,1] += x[i,1]
            cluster[country_clas,2] += x[i,2]
        
        #average all the values with the number of countries
        for j in range(10):
            cluster[j,:] /= max(no_countries[j], 1)

        #compute cost function
        wss_new = np.linalg.norm(dist)

        #check convergence and return array
        change = np.abs(wss_new - wss) / wss

        if change < 1e-4:
            print('Convergence achieved at iteration number {0}'.format(k))
            return cluster, dist, clas, no_countries

        #update value of cost function
        wss = wss_new

        if k == max_iter - 1:
            print('maximum number of iterations reached. Convergence not achieved')
            return cluster, dist, clas, no_countries

#-----------------------------
#call the function and save results
#-----------------------------

#get numpy array with gdp and gps for calling the function     
data_to_function = data.loc[:,['gdp', 'latitude', 'longitude']].values

#maximum number of iterations
max_iter = 1000

#function call
cluster_out, dist_out, clas_out, no_countries_out = f_group(data_to_function, max_iter)

#group the countries in a new dataframe with the information of the clusters
data['cluster'] = clas_out

#-----------------------------
#function add the average gdp of each cluster
#-----------------------------

#computes the average gdp per capita of each cluster
def mean_gdp(x, no_countries):

    #number of rows
    nrow = x.shape[0]

    #initialize arrays
    av_gdp = np.zeros(10)
    av_gdp_data = np.zeros(nrow)

    #compute the mean
    for i in range(nrow):
        
        #cluster number
        no_clas = x.loc[i,'cluster']

        #gdp
        av_gdp[no_clas] += x.loc[i,'gdp']

    for j in range(10):
        
        av_gdp[j] /= max(1, no_countries[j])

    #create an array with the mean gdp of the group for all the country list
    for i in range(nrow):

        #cluster number
        no_clas = x.loc[i,'cluster']

        #av. gdp
        av_gdp_data[i] = av_gdp[no_clas]

    return av_gdp_data

#-----------------------------
#call the function and save results
#-----------------------------

gdp_cluster_out = mean_gdp(data, no_countries_out)

data['av_gdp'] = gdp_cluster_out

#-----------------------------
#function add the cumulative gdp within each cluster
#-----------------------------

data.sort_values(by=['av_gdp', 'gdpt'], ascending=False, inplace=True)
data.reset_index(drop=True, inplace=True)

def cumulative_gdp(x):

    #number of rows
    nrow = x.shape[0]

    #array with the cumulative gdp fraction
    cum_gdp = np.zeros(nrow)

    #group countries by cluster to compute the cumulative gdp
    cluster_grouped = x.groupby(by='cluster')

    #compute cumulative gdp
    for j, group in cluster_grouped:

        #index of the countries of group j on the original data frame
        index = group.index

        #total gdp of cluster clas
        total_gdp = sum(group['gdpt'])

        #compute cumulative gdp of country i in cluster j
        for i in index:
            cum_gdp[i] = x.loc[i,'gdpt'] / total_gdp

    return cum_gdp

#call the function and obtain the cumulative gdp fraction
cum_gdp_out = cumulative_gdp(data)

#update dataframe
data['cum_gdp'] = cum_gdp_out

#-----------------------------
#create final dataframe and write it as csv file
#-----------------------------

#path dataframe out
data_path_out = r'C:\Users\sesig\Documents\tests_for_jobs\infinia_mobile\geo_economic_clustering.csv'

#data frame out
data_out = data[['country', 'cluster', 'av_gdp', 'gdp', 'gdpt', 'cum_gdp']]

#write file
data_out.to_csv(data_path_out, sep=',', index=False)
