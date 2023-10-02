import pandas
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics


data = pandas.read_csv("data_12000.csv")
df = data.where((pandas.notnull(data)), '')


df['spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
df['blacklisted'] = df['Status'].apply(lambda x: 2 if x == 'blacklisted' else 3)


X_train, X_test, y_train, y_test = train_test_split(df.Message, df.spam)
Z_train, Z_test, C_train, C_test = train_test_split(df.Ip, df.blacklisted)

v = TfidfVectorizer()

X_train_count = v.fit_transform(X_train.values)


#model = SVC(kernel='rbf', random_state = 1)
#model=model.fit(X_train_count, y_train)

X_test_count = v.transform(X_test.values)
#y_pred = model.predict(X_test_count)
#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
'''
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
accuracy = float(cm.diagonal().sum())/len(y_test)
print("\nAccuracy Of SVM For The Given Dataset using Confusion Matrix: ", accuracy)
'''

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y_train = le.fit_transform(y_train)

classifier = SVC(kernel='rbf', random_state = 1)
classifier.fit(X_train_count,Y_train)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
plt.figure(figsize = (7,7))
X_set, y_set = X_train_count, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01), np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

print("X_train:",X_train_count.shape,"\n", X_train_count)
print("Xset: ",X_set.shape,"\n", X_set)
print("X1: ",X1.shape, X1)
print("X1.ravel: ",X1.ravel().shape, X1.ravel())
print("Y_train", Y_train)
print("Yset\n", y_set)
print("X2\n", X2)
print("np.array:", (np.array([X1.ravel(), X2.ravel()]).T).shape)
print("X_train_count: ", X_train_count.shape)

'''
plt.contourf(X1, X2, classifier.predict(X_test_count).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('black', 'white')))
#plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('black', 'white')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'orange'))(i), label = j)
plt.title('Apples Vs Oranges')
plt.xlabel('Weight In Grams')
plt.ylabel('Size in cm')
plt.legend()
plt.show()
'''