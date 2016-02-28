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
samples_noise=samples

# Generate new data
for i in range(0,samples.shape[1]):
	std=np.std(samples[:,i])
	s = np.random.normal(0, std/3, samples.shape[0])
	samples_noise[:,i]=samples_noise[:,i]+s

samples=np.concatenate([samples,samples_noise])

labels=my_data[1:my_data.shape[0],1]
labels=np.concatenate([labels,labels])



# train/test split
"""split = StratifiedShuffleSplit(labels, n_iter=1, test_size=0.2, random_state=0)
train_index, test_index = list(split)[0]
X_train, y_train = samples[train_index], labels[train_index]
X_test, y_test = samples[test_index], labels[test_index]"""

# boosting
"""estimator = GradientBoostingClassifier(n_estimators=40, learning_rate=0.1, max_depth=10, random_state=0)
estimator.fit(X_train, y_train)
y_pred = estimator.predict(X_test)
print(classification_report(y_test, y_pred))"""

# Random Forest
clf = RandomForestClassifier(n_estimators=100, criterion="entropy", max_depth=20)
scores = cross_val_score(clf, samples, labels)
print scores
clf=clf.fit(samples,labels)

# Classify test data
my_data = genfromtxt('test_f.csv', delimiter=',')
samples_tst=my_data[1:my_data.shape[0],1:my_data.shape[1]]
ids=my_data[1:my_data.shape[0],0]
proba=clf.predict_proba(samples_tst)
output=[ids,proba[:,1]]
output=np.transpose(output)
np.savetxt("output.csv", output, delimiter=",")
"""scores = cross_val_score(clf, samples, labels)
print scores"""
