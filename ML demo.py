import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('creditcard.csv')
print(df)
print(df.describe())
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

data = pd.read_csv('creditcard.csv')
features = ['V%d' % number for number in range(1, 28)]
target = 'Class'
def draw_histograms(dataframe, features, rows, cols):
    fig=plt.figure(figsize=(20,20))
    for i, feature in enumerate(features):
        ax=fig.add_subplot(rows,cols,i+1)
        dataframe[feature].hist(bins=20,ax=ax,facecolor='midnightblue')
        ax.set_title(feature+" Distribution",color='DarkRed')
        ax.set_yscale('log')
    fig.tight_layout()  
    plt.show()
draw_histograms(data,data.columns,8,4)
print(data.Class.value_counts())
X = data[features]
y = data[target]

def normalize(X):
    for feature in X.columns:
       X[feature] -= X[feature].mean()
       X[feature] /= X[feature].std()
    return X

model = LogisticRegression()
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
for train_indices, test_indices in splitter.split(X, y): 
    X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
    X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]
X_train = normalize(X_train)
X_test = normalize(X_test)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_pred, y_test))
print('\n')

from sklearn.metrics import roc_curve, auc
logistic_fpr, logistic_tpr, threshold = roc_curve(y_test, y_pred)
auc_logistic = auc(logistic_fpr, logistic_tpr)
plt.figure(figsize=(5, 5), dpi=100)
plt.plot(logistic_fpr, logistic_tpr, marker='.', label='Logistic (auc = %0.3f)' % auc_logistic)
plt.xlabel('False Positive Rate -->')
plt.ylabel('True Positive Rate -->')
plt.legend()
plt.show()
