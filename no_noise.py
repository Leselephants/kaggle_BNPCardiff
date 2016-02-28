from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import classification_report
from numpy import genfromtxt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
import numpy as np

my_data = genfromtxt('train_f.csv', delimiter=',')

samples=my_data[1:my_data.shape[0],2:my_data.shape[1]]
labels=my_data[1:my_data.shape[0],1]

# Random Forest
clf = RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=365,max_features=50,min_samples_leaf=3,max_depth=20)
scores = cross_val_score(clf, samples, labels)
print scores
"""clf=clf.fit(samples,labels)


# Classify test data
my_data = genfromtxt('test_f.csv', delimiter=',')
samples_tst=my_data[1:my_data.shape[0],1:my_data.shape[1]]
ids=my_data[1:my_data.shape[0],0]
proba=clf.predict_proba(samples_tst)
output=[ids,proba[:,1]]
output=np.transpose(output)
np.savetxt("output.csv", output, delimiter=",")"""
