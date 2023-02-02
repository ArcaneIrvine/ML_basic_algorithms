# pandas dataframe
import pandas as pd
# number array
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
# for testing the module
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# load our data
dataset = pd.read_csv('data/KNN_data.csv')

"""
From the dataset we load some values on specific columns are zero when they shouldn't be because
it doesnt make sense for a human to have zero Insulin for example, so from those columns we have 
to remove or rather replace the zeros. We can do that by replacingthese unusable values with NaN's. 
Then replace the NaN's with an integer from the databasse from the column where we skip na's.
By doing so we replace the data for the people from whom the data is missing with the average, the most common data for that.
"""

# remove zeros from these columns
columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']

for item in columns:
    # replace zeros with NaN
    dataset[item] = dataset[item].replace(0, np.NaN)
    # integer from the database from the column where we skip na's
    mean = int(dataset[item].mean(skipna=True))
    # replace all np.NaN's with the mean
    dataset[item] = dataset[item].replace(np.NaN, mean)

