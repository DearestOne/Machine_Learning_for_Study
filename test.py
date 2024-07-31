import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

dataset = pd.read_csv("Dataset/weather_classification_data.csv")
data = dataset.iloc[:,range(10)]
target = dataset.iloc[:,10]
print(dataset.info())
# print(data.head(3))
# print(target.head(3))

encoder = LabelEncoder()
data.loc[:, 'Cloud Cover'] = encoder.fit_transform(data['Cloud Cover'])
data.loc[:, 'Season'] = encoder.fit_transform(data['Season'])
data.loc[:, 'Location'] = encoder.fit_transform(data['Location'])

# print(dataset.info())
# print(data.head(3))
X_train, X_test, y_train, y_test = train_test_split(data, target, train_size = 0.8, stratify=target, random_state=1)
# series

model = SVC(kernel = 'rbf')
model.fit(X_train, y_train)
y_predict = model.predict(X_test)

# for i in range(5):
#         print(i+1, end = "\t")
#         print(y_test.array[i],end = '\t')
#         print(y_predict[i])

all_class = list(set(target))

matrix = confusion_matrix(y_test, y_predict, labels= all_class)
print(matrix)
print(classification_report(y_test, y_predict))

cmd = ConfusionMatrixDisplay(matrix, display_labels= all_class)
cmd.plot()
plt.show()

scaler = StandardScaler()
X_train, X_test = scaler.fit_transform(X_train), scaler.fit_transform(X_test)

model = SVC(kernel = 'rbf')
model.fit(X_train, y_train)
y_predict = model.predict(X_test)

matrix = confusion_matrix(y_test, y_predict, labels= all_class)
print(matrix)
print(classification_report(y_test, y_predict))