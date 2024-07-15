from skimage.metrics import mean_squared_error
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split

data = load_diabetes()

# print(data.data.shape)  # sample count
# print(data.target.shape)  # feature count

# print(data.feature_names)  # all feature name

# print(data.target)  # target is continuous sample

# print(data.data)  # print data
# print(data.target)  # print target

# Linear Regression
all_train = [0.99,0.95,0.80,0.20,0.05,0.01]
for i in all_train:
    model = LinearRegression()
    Xtrain, Xtest, ytrain, ytest = train_test_split(data.data, data.target, train_size=i)
    model.fit(Xtrain, ytrain)
    y_predict = model.predict(Xtest)
    mse = mean_squared_error(ytest, y_predict)
    print('LINEAR --> train %.2f : test %.2f, mse = %f' %(i, 1-i, mse))
print('\n\n')

# Ridge Regression
all_train = [0.99,0.95,0.80,0.20,0.05,0.01]
for i in all_train:
    print('RIDGE --> train %.2f : test %.2f | lambda 1-5 \n\t mse = ' %(i, 1-i), end = '')
    xtrain, xtest, ytrain, ytest = train_test_split(data.data, data.target, train_size = i)
    for j in range(1,6):
        model = Ridge(alpha = j)
        model.fit(xtrain, ytrain)
        y_predict = model.predict(xtest)
        mse = mean_squared_error(ytest, y_predict)
        print('%.2f | '%mse, end = '')
    print('\n')
print('\n\n')

# Lasso Regression
all_train = [0.99,0.95,0.80,0.20,0.05,0.01]
for i in all_train:
    print('LASSO --> train %.2f : test %.2f | lambda 1-5 \n\t mse = ' %(i, 1-i), end = '')
    xtrain, xtest, ytrain, ytest = train_test_split(data.data, data.target, train_size = i)
    for j in range(1,6):
        model = Lasso(alpha = j)
        model.fit(xtrain, ytrain)
        y_predict = model.predict(xtest)
        mse = mean_squared_error(ytest, y_predict)
        print('%.2f | '%mse, end = '')
    print('\n')
print('\n\n')

# ElasticNet Regression
all_train = [0.99,0.95,0.80,0.20,0.05,0.01]
for i in all_train:
    print('ELASTICNET --> train %.2f : test %.2f | lambda 1-5 \n\t mse = ' %(i, 1-i), end = '')
    xtrain, xtest, ytrain, ytest = train_test_split(data.data, data.target, train_size = i)
    for j in range(1,6):
        model = ElasticNet(alpha = j)
        model.fit(xtrain, ytrain)
        y_predict = model.predict(xtest)
        mse = mean_squared_error(ytest, y_predict)
        print('%.2f | '%mse, end = '')
    print('\n')