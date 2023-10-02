import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics


data = pandas.read_csv("data_12000.csv")
df = data.where((pandas.notnull(data)), '')

df['spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)

X_train, X_test, y_train, y_test = train_test_split(df.Message, df.spam, test_size=0.33, random_state=42)

v = TfidfVectorizer()

X_train_count = v.fit_transform(X_train.values)
X_test_count = v.transform(X_test.values)

from sklearn.svm import SVC
clf = SVC(kernel='linear', gamma='auto', C=32, probability=True)
clf=clf.fit(X_train_count, y_train)
svc_pred = clf.predict(X_test_count)


#       ACCURACY SCORE 
print("Accuracy")
print("SVC:\t",metrics.accuracy_score(y_test, svc_pred))


from sklearn.metrics import precision_score
print('Precision')
print('SVC:\t', precision_score(y_test, svc_pred))

from sklearn.metrics import recall_score
print('Recall')
print('SVM:\t', recall_score(y_test, svc_pred))

from sklearn.metrics import f1_score
print('F1 Score')
print('SVM:\t', f1_score(y_test, svc_pred))

from sklearn.metrics import mean_squared_error
from math import sqrt
print('Root Mean Squared Error')
print('SVM:\t', sqrt(mean_squared_error(y_test, svc_pred)))

#       CONFUSION MATRIX
print("Confusion Matrix")
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
#Generate the confusion matrix
cf_matrix = confusion_matrix(y_test, svc_pred)
print("SVC:")
print(cf_matrix)

#clf = SVC(kernel='linear', gamma='auto', C=32)