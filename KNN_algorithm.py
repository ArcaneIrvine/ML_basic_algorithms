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
to remove or rather replace the zeros. We can do that by replacing these unusable values with NaN's. 
Then replace the NaN's with an integer from the database from the column where we skip na's.
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


"""
In this part we simply split the date to training and testing
"""
# all columns from 0 to 7 as column 9 in our data is the outcome so not part of the training data
x = dataset.iloc[:, 0:8]
# just the column 8
y = dataset.iloc[:, 8]
# split our data in training data and testing data, 20% of the data put aside for testing later
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2)

"""
In this part we simply scale the data, what that means is basically transforming our data so it fits within a specific range.
In our case we set that range from [-1,1] which is the standard scaling. You usually want to scale your data on methods that
that measure distance or normality. 
"""

# scale the data with standard_scaler() so all the data is inside [-1,1]
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

"""
Define the model using KNeighborsClassifier and fit the training data into the model. The sqrt of the length of our y_test
data equals to 12.4096, 12 is an even number and we don't want to have an even number of neighbors voting so we will make it 
11  and use euclidian metric which is a common one for measuring the distance and works well
"""

# define model
classifier = KNeighborsClassifier(n_neighbors=11, p=2, metric='euclidean')
# fit the model
classifier.fit(x_train, y_train)

"""
In this final step we will evaluate the model by using a confusion matrix that is going to give us the prediction vs the
actual results. It gives us the following matrix in the print. The main diagonal of the matrix consists of 94 and 32 which means
that prediction and the actual agreed (so 94 people don't have diabetes and 32 people have diabetes. On the anti-diagonal 
13 and 15 did not agree (so prediction said 13 of those 94 people did have diabetes and that 15 out of those 32 people didn't have
diabetes.
"""
# predict the test set results
y_predict = classifier.predict(x_test)

# evaluate the model
cm = confusion_matrix(y_test, y_predict)
print("confusion matrix:\n", cm)


"""
Simply print the score and accuracy of our model
"""
print("score: ", f1_score(y_test, y_predict))
print("accuracy: ", accuracy_score(y_test, y_predict))
