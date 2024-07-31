#use standard scal
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
model = SVC(kernel='rbf')
model.fit(X_train, y_train)
scaler = StandardScaler()  # svm is grater in standard scaler
X_train, X_test = scaler.fit_transform(X_train), scaler.fit_transform(X_test)

model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print(classification_report(y_test, y_predict))

all_col = list(set(y))
print(all_col)
cm = confusion_matrix(y_test, y_predict, labels=all_col)
print(cm)
cmd = ConfusionMatrixDisplay(cm, display_labels=all_col)
cmd.plot()
plt.show()