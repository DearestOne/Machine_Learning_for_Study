from time import time
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import LabelEncoder

start = time()

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

clf = MLPClassifier(random_state=1, max_iter=1000, hidden_layer_sizes=(500,500,500,500)).fit(X_train, y_train)
y_predict= clf.predict(X_test)

print(classification_report(y_test, y_predict))
all_class = list(set(target))
cm = confusion_matrix(y_test, y_predict, labels=all_class)
print(cm)
end = time()

print("time used %.2f seconds"  %(end-start))
cmd = ConfusionMatrixDisplay(cm, display_labels= all_class)
cmd.plot()
plt.show()