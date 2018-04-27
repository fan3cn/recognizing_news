print("importing packages...")
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
import numpy as np;

print("loading data...")
X = np.loadtxt(open("data/X.train", "rb"));
Y = np.loadtxt(open("data/Y.train", "rb"));
X_validate = np.loadtxt(open("data/X.test", "rb"));
ids = np.loadtxt("data/id.test", dtype=bytes).astype(str)

print(X.shape)
print(Y.shape)
print(X_validate.shape)

print("training model...")
#clf = svm.SVC()
#clf = linear_model.LogisticRegression(C=1e5)
clf = MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32, 16, 8))
clf.fit(X, Y)

print("making predictions...")
y_validate = clf.predict(X_validate);
assert(len(ids) == len(y_validate))
result = [ [ids[i] + "\t"+ str(int(y_validate[i])) + "\t" + "NULL" + "\t" + "NULL"] for i in range(len(y_validate))]
#print(Y_validate)
np.savetxt( "result.txt", result, fmt='%s')

print('Done!')

