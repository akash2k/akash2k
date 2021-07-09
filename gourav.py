import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score,auc
dataset = pd.read_csv('creditcard.csv')
#data exploration
dataset.head()
dataset.isnull()
#data preprocessing
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
dataset['normalizedAmount'] = sc.fit_transform(dataset['Amount'].values.reshape(-1,1))
dataset = dataset.drop(['Amount'], axis = 1)
dataset['normalizedTime'] = sc.fit_transform(dataset['Time'].values.reshape(-1,1))
dataset = dataset.drop(['Time'], axis = 1)
dataset.head()
X = dataset.iloc[:, dataset.columns != 'Class'].values
y = dataset.iloc[:, dataset.columns == 'Class'].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
dataset['Class'].value_counts()
from imblearn.over_sampling import SMOTE
#smote = SMOTE(random_state=42, ratio=1.0)
smote = SMOTE(random_state=42, sampling_strategy=1.0)
X_res, y_res = smote.fit_sample(X_train, y_train)
df = pd.DataFrame(data=y_res)
df.count()
df.index.value_counts()
df[0].value_counts()

import keras 
from keras.layers import Dense
from keras.models import Sequential

# Initialising the ANN
classifier = Sequential()
classifier.add(Dense(output_dim = 15, init = 'uniform', activation = 'relu', input_dim = 30))
classifier.add(Dense(output_dim = 15, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_res, y_res, batch_size = 32, nb_epoch = 20)
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
from sklearn.metrics import precision_score,recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
print ('Accuracy:', accuracy_score(y_test, y_pred))