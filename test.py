from sklearn import tree
from numpy import genfromtxt
from sklearn.externals.six import StringIO   
import math
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.decomposition import PCA

my_data = genfromtxt('train_f.csv', delimiter=',')
fraction=49

samples=my_data[1:my_data.shape[0],2:my_data.shape[1]]
samples_noise=samples
labels=my_data[1:my_data.shape[0],1]

# Apply dimension reduction
"""pca = PCA(n_components=50)
samples=pca.fit_transform(samples)"""

# Generate new data
for i in range(0,samples.shape[1]):
	std=np.std(samples[:,i])
	s = np.random.normal(0, std/3, samples.shape[0])
	samples_noise[:,i]=samples_noise[:,i]+s

#samples_noise=pca.transform(samples_noise)
samples=np.concatenate([samples,samples_noise])
labels=np.concatenate([labels,labels])

# get data and labels
"""indexes=np.arange(samples.shape[0])
samples_tr=samples[indexes%100<=fraction,:]
labels_tr=labels[indexes%100<=fraction]"""

clf=tree.DecisionTreeClassifier(criterion="entropy",splitter="best",max_depth=20)
#scores = cross_val_score(clf, samples, labels,scoring="mean_squared_error",cv=3)
clf=clf.fit(samples,labels)

# Classify test data
my_data = genfromtxt('test_f.csv', delimiter=',')
samples_tst=my_data[1:my_data.shape[0],1:my_data.shape[1]]
ids=my_data[1:my_data.shape[0],0]
proba=clf.predict_proba(samples_tst)
output=[ids,proba[:,1]]
output=np.transpose(output)
np.savetxt("output.csv", output, delimiter=",")

"""clf=clf.fit(samples_tr,labels_tr)
with open("iris.dot",'w') as f:
	f = tree.export_graphviz(clf, out_file=f)

# get data and labels
samples_tst=samples[indexes%100>fraction,:]
labels_tst=labels[indexes%100>fraction]

labels_test=[]
for i in range(0,samples_tst.shape[0]):
	labels_test.append(clf.predict(samples_tst[i,:]))
"""


