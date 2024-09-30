from pathlib import Path
import pandas as pd
import tarfile
import urllib.request


""" 
    this function loads in data as name suggests but it does something interesting
"""
def load_housing_data():
    tarball_path = Path("datasets/housing.tgz") # takes in the 
    if not tarball_path.is_file():
        Path("Datasets").mkdir(parents=True, exist_ok=True) # checks if path is there and if not creates it 
        url = "https://github.com//ageron/data/raw/main/housing.tgz" # specification of the URL
        urllib.request.urlretrieve(url, tarball_path) # looks for requested file at specified url
        
        # opens the file and extracts it into the specified directories
        with tarfile.open(tarball_path) as housing_tarball:     
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing = load_housing_data()

# shows the first 5 lines of the dataframe
# shows the values in each one of the columns for a single record
print("hello world")
print(housing.head())

# shows the columns and their data types allows you to compare missing values as well, can you spot the difference?
housing.info()

# you can check the contents of certain objects, such as the ocean_proximity 
# which we know to be a string by heading the dataset this allows you to find out
# the categories and how many times they occur
print(housing["ocean_proximity"].value_counts())

# next is using the describe function which tells you a lot
# about the different data points in your CSV
# giving the count, mean, standard deviation, min, it gives a lot of info that helps to familiarize the data
print(housing.describe())

# next is plotting the data, this is a histogram of the data
# using a histogram can give you a better feel of the data that you are working with so
# lets import matplotlib and see what's poppin

import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()


# be warned do not look too much into the data or you may be liable of data snooping bias which is when you look too deep into your dataset 
# doing so may cause you to choose a model that does not perform as well as you want it to
# causing many issues, so we should make a test set now now and divide all of the data 

import numpy as np

def shuffle_and_split_data(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

# we will use a 20% test set, which is a common size for a test set
# seperating the dataset like this causes some problems here and there and can result in all of that data set being seen
# one way of stopping this is to use a seed or you can do something where you compare the hash of each sample
# and anything lower than 20% of the max value goes into the test set
train_set, test_set = shuffle_and_split_data(housing, 0.2)
print(len(train_set), "train +", len(test_set), "test")

# the next snippet covers a solution to do this using the hashing method described 
from zlib import crc32

def is_id_in_test_set(identifier, test_ratio):
    return crc32(np.int64(identifier)) < test_ratio * 2**32

def split_train_test_by_id_hash(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

# due to the housing column not having an indentifier we can make the dataset have an index number which would function as the same thing 
housing_with_id = housing.reset_index() # adds an index column to the set

train_set, test_set = split_train_test_by_id_hash(housing_with_id, 0.2, "index") # splits the data in in housing_with_id into 20% based on the index

# the next snippet is a way of splitting the data using the latitude and longitude as the identifier because these are always stable 
# and will always be the same
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id_hash(housing_with_id, 0.2, "id")

# another way of splitting the data is to use the scikit-learn which is capable of doing the same thing and much more
# than the function we defined a bit earlier
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# when using this data we need to make sure that we are representative of the population 
# so we use stratified sampling which is when you divide the population evenly to represent 
# the population as a whole such as doing a survey you may want to represent the population
# based on gender so 51.1 % of the populationis female, you should try to get that many females to take your survey 
# the remaining surveys go to males 
# these groups are called strata

# this is done to avoid sampling bias which is when the sample is not representative of the population
# the next snippet is a way of splitting the data into strata based on the median income
# this is a way of making sure the data is representative of the population
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

# making some graphs to represent the new categories that we just made
housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
plt.xlabel("Income Category")
plt.ylabel("Number of districts")
plt.show()

# now it is time to make a splitter that splits the data into different strata
from sklearn.model_selection import StratifiedShuffleSplit

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
strat_splits=[]
for train_index, test_index in splitter.split(housing, housing["income_cat"]):
    strat_train_set_n = housing.iloc[train_index]
    strat_test_set_n = housing.iloc[test_index]
    strat_splits.append((strat_train_set_n, strat_test_set_n))

# just using the first split we have
strat_train_set, strat_test_set = strat_splits[0]

# another way to accomplish stratifying
#                                                   arrays,  size of val set, the seed, using the income_cat as classifiers
strat_train_set, strat_test_set = train_test_split(housing, test_size=0.2, random_state=42, stratify=housing["income_cat"])

print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))


# now that we have the data we can start to explore it a bit more
# now we will not be using the income_cat we can drop it from the set 
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)  # axis=1 is for columns, axis=0 is for rows

    

