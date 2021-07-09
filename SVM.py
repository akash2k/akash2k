from sklearn import svm
import pandas as pd
df = pd.read_csv('creditcard.csv')
print(df.describe())
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
data = pd.read_csv('creditcard.csv')
features = ['V%d' % number for number in range(1, 28)]
target = 'Class'
X = data[features]
y = data[target]
def normalize(X):
   for feature in X.columns:
       X[feature] -= X[feature].mean()
       X[feature] /= X[feature].std()
   return X
model = svm.SVC()
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
for train_indices, test_indices in splitter.split(X, y):
    X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
    X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]
X_train = normalize(X_train)
X_test = normalize(X_test)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy: ",accuracy_score(y_pred, y_test))
