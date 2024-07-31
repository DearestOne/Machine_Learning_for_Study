import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

data = load_digits()
X = data.data
y = data.target # 10 class {0 - 9}
all_class = list(set(y))


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1, train_size= 0.8)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

model_rbf = SVC(kernel="rbf")
model_rbf.fit(X_train, y_train)
y_predict = model_rbf.predict(X_test)
cr = classification_report(y_test, y_predict)
print(cr)

cm = confusion_matrix(y_test, y_predict, labels=all_class)
print(cm)
cmd = ConfusionMatrixDisplay(cm, display_labels= all_class)
cmd.plot()
plt.show()

model_poly = SVC(kernel="poly")
model_poly.fit(X_train, y_train)
y_predict = model_poly.predict(X_test)
cr = classification_report(y_test, y_predict)
print(cr)

cm = confusion_matrix(y_test, y_predict, labels=all_class)
print(cm)
cmd = ConfusionMatrixDisplay(cm, display_labels= all_class)
cmd.plot()
plt.show()

